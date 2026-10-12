import os
import sys
import io
import re
import contextlib

# Disable telemetry and set minimal logging levels before any library imports
os.environ["ONELOGGER_DISABLE"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import wave
import tempfile
import warnings
import logging
import time
import threading
import numpy as np
import sounddevice as sd
import keyboard
import pyperclip
import torch

# Filter verbose warnings
warnings.filterwarnings("ignore")
logging.getLogger("nemo_logger").setLevel(logging.ERROR)
logging.getLogger("nemo").setLevel(logging.ERROR)
logging.getLogger("lhotse").setLevel(logging.ERROR)

from huggingface_hub import hf_hub_download

# Silence OneLogger banner during NeMo import
_old_stdout, _old_stderr = sys.stdout, sys.stderr
sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
try:
    import nemo.collections.asr as nemo_asr
    import nemo.utils as nemo_utils
finally:
    sys.stdout, sys.stderr = _old_stdout, _old_stderr

# Completely silence NeMo internal logging/warnings during transcription
nemo_utils.logging.setLevel(logging.ERROR)
nemo_utils.logging.warning = lambda *args, **kwargs: None
nemo_utils.logging.info = lambda *args, **kwargs: None

# --- Model & Hardware Configuration ---
MODEL_NAME = "nectec/Pathumma-stt-th-fastconformer-rnnt-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# --- Decoding Strategy & Beam Search Configuration ---
# Options:
#   BEAM_SIZE = 1  : Greedy decoding (Default, fastest ~80ms, ultra-low latency)
#   BEAM_SIZE = 3  : Beam Search 3 hypotheses (~270ms, recommended balance of speed & accuracy)
#   BEAM_SIZE = 5  : Beam Search 5 hypotheses (~400ms, deeper hypothesis exploration)
BEAM_SIZE = 3

print(f"Loading {MODEL_NAME} on {DEVICE.upper()} ({TORCH_DTYPE})...")
try:
    # First priority: Load instantly from local cache without network latency
    model_path = hf_hub_download(
        repo_id=MODEL_NAME,
        filename="pathumma-stt-th-fastconformer-rnnt-large.nemo",
        local_files_only=True
    )
except Exception:
    try:
        # Second priority: Official Hugging Face Hub
        model_path = hf_hub_download(
            repo_id=MODEL_NAME,
            filename="pathumma-stt-th-fastconformer-rnnt-large.nemo"
        )
    except Exception:
        # Fallback: High-speed HF Mirror
        print("Official Hugging Face failed/timed out. Switching to high-speed HF mirror...")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        model_path = hf_hub_download(
            repo_id=MODEL_NAME,
            filename="pathumma-stt-th-fastconformer-rnnt-large.nemo"
        )
asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(restore_path=model_path)
asr_model.to(DEVICE)
asr_model.to(TORCH_DTYPE)
asr_model.eval()

# Configure Beam Search if BEAM_SIZE > 1
if BEAM_SIZE > 1:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        dec_cfg = asr_model.cfg.decoding
        dec_cfg.strategy = "beam"
        dec_cfg.beam.beam_size = BEAM_SIZE
        dec_cfg.beam.return_best_hypothesis = True
        asr_model.change_decoding_strategy(dec_cfg, verbose=False)

print(f"Decoding strategy: {'Beam Search (size=' + str(BEAM_SIZE) + ')' if BEAM_SIZE > 1 else 'Greedy (size=1)'}")
print("Ready! Hold 'Win + Z' to speak, release to paste text (1-min stats summary active | Press Ctrl+C to exit)")

# --- Audio Recording & Hotkey Setup ---
SAMPLE_RATE = 16000
AUDIO_SPEED_RATE = 0.90  # Slow down speech to 0.90x without altering pitch to improve recognition
WIN_KEYS = {"windows", "left windows", "right windows", "win"}

is_win_pressed = False
is_z_pressed = False
is_recording = False
audio_frames = []
temp_wav_path = os.path.join(tempfile.gettempdir(), f"pathumma_{os.getpid()}.wav")

# --- Thai Number & Math Normalization ---
DIGITS = {
    'ศูนย์': 0, 'หนึ่ง': 1, 'เอ็ด': 1, 'สอง': 2, 'ยี่': 2, 'สาม': 3,
    'สี่': 4, 'ห้า': 5, 'หก': 6, 'เจ็ด': 7, 'แปด': 8, 'เก้า': 9
}
MULTIPLIERS = {
    'สิบ': 10, 'ร้อย': 100, 'พัน': 1000, 'หมื่น': 10000, 'แสน': 100000, 'ล้าน': 1000000
}
ALL_TOKENS_SORTED = sorted(list(DIGITS.keys()) + list(MULTIPLIERS.keys()), key=len, reverse=True)
PATTERN = '|'.join(ALL_TOKENS_SORTED)

OPERATORS = {
    'จุด': '.',
    'บวก': '+',
    'ลบ': '-',
    'ขีด': '-',
    'คูณ': '*',
    'หาร': '/',
    'ยกกำลัง': '^'
}

SINGLE_DIGITS = {
    'ศูนย์': '0', 'หนึ่ง': '1', 'สอง': '2', 'สาม': '3', 'สี่': '4',
    'ห้า': '5', 'หก': '6', 'เจ็ด': '7', 'แปด': '8', 'เก้า': 9
}

def split_into_number_chunks(tokens: list) -> list:
    """
    Splits a stream of number tokens into independent numbers.
    In Thai counting grammar, place values in a single number strictly decrease
    (e.g., ล้าน -> แสน -> หมื่น -> พัน -> ร้อย -> สิบ -> หน่วย).
    When a multiplier is encountered that is >= the previous multiplier in the current number
    (e.g., 'สิบ' followed by 'ยี่สิบ', or 'ร้อย' followed by 'สองร้อย'),
    it represents the start of a new number.
    """
    chunks = []
    current_chunk = []
    last_mult = float('inf')
    pending_digit_idx = None

    for t in tokens:
        if t in MULTIPLIERS:
            m = MULTIPLIERS[t]
            if m >= last_mult:
                if pending_digit_idx is not None and pending_digit_idx > 0:
                    prev_chunk = current_chunk[:pending_digit_idx]
                    new_chunk = current_chunk[pending_digit_idx:] + [t]
                    if prev_chunk:
                        chunks.append(prev_chunk)
                    current_chunk = new_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = [t]
                last_mult = m
                pending_digit_idx = None
            else:
                current_chunk.append(t)
                last_mult = m
                pending_digit_idx = None
        elif t in DIGITS:
            if pending_digit_idx is None:
                pending_digit_idx = len(current_chunk)
            current_chunk.append(t)

    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def evaluate_chunk(tokens: list) -> str:
    """
    Evaluates a single chunk of tokens.
    - If a place-value multiplier appears without any digit on either side
      (e.g., 'ร้อย', 'พัน', 'หมื่น', 'แสน', 'ล้าน'), returns None so it's kept as Thai words.
    - Evaluates composite numbers (e.g. 'หนึ่งร้อยสามสิบสอง' -> '132', 'ยี่สิบ' -> '20', 'สิบ' -> '10').
    - Evaluates consecutive digits >= 2 (e.g. 'ศูนย์ แปด หนึ่ง' -> '081').
    """
    has_digits = any(t in DIGITS for t in tokens)
    has_mults = any(t in MULTIPLIERS for t in tokens)

    # Bare place-value multiplier with no digits on either side:
    if has_mults and not has_digits:
        if tokens != ['สิบ']:
            return None
        # Standalone 'สิบ' evaluates to 10
        return '10'

    if has_mults:
        total = 0
        current = 0
        for t in tokens:
            if t in DIGITS:
                current = DIGITS[t]
            elif t in MULTIPLIERS:
                m = MULTIPLIERS[t]
                if m == 10:
                    if current == 0: current = 1
                    total += current * 10
                    current = 0
                elif m == 1000000:
                    if current == 0 and total == 0: current = 1
                    total = (total + current) * 1000000
                    current = 0
                else:
                    if current == 0: current = 1
                    total += current * m
                    current = 0
        total += current
        return str(total)
    elif len(tokens) >= 2 and all(t in DIGITS and t not in ('เอ็ด', 'ยี่') for t in tokens):
        return ''.join(str(DIGITS[t]) for t in tokens)
    return None

def pass1_thai_numbers(text: str) -> str:
    """
    Pass 1: Converts Thai number phrases:
    - Splits consecutive numbers (e.g. 'สิบ ยี่สิบ สามสิบ' -> '10 20 30').
    - Ignores bare place-values ('หลักร้อย', 'หลักสิบ', 'มีเป็นร้อย').
    - Standalone single digits ('หนึ่ง', 'สอง', ...) are left as Thai words.
    """
    token_regex = re.compile(f'(?:(?:{PATTERN})\\s*)+')
    result = []
    last_idx = 0

    for match in token_regex.finditer(text):
        start, end = match.start(), match.end()
        result.append(text[last_idx:start])
        matched_str = match.group(0)

        # Do not convert if explicitly preceded by 'หลัก' (e.g., 'หลักสิบ', 'หลักร้อย')
        prefix = text[:start].rstrip()
        if prefix.endswith('หลัก'):
            result.append(matched_str)
            last_idx = end
            continue

        tokens = re.findall(PATTERN, matched_str)
        if not tokens:
            result.append(matched_str)
            last_idx = end
            continue

        chunks = split_into_number_chunks(tokens)
        converted_parts = []
        for c in chunks:
            val = evaluate_chunk(c)
            if val is not None:
                converted_parts.append(val)
            else:
                converted_parts.append(' '.join(c))

        lead = matched_str[:len(matched_str) - len(matched_str.lstrip())]
        trail = matched_str[len(matched_str.rstrip()):]
        mid = " ".join(converted_parts)
        result.append(f'{lead}{mid}{trail}')
        last_idx = end

    result.append(text[last_idx:])
    return ''.join(result)

def pass2_math_operators(text: str) -> str:
    """
    Pass 2: Converts math operators ('จุด', 'บวก', 'ลบ', 'ขีด', 'คูณ', 'หาร', 'ยกกำลัง')
    into symbols only when both left and right sides are numeric (Arabic digits or single Thai digits).
    - 'ขีด' converts to '-' with no surrounding spaces (e.g. 113ขีด002 -> 113-002).
    - 'จุด' converts to '.' with no surrounding spaces (e.g. 3 จุด 5 -> 3.5).
    """
    thai_digit_pattern = '|'.join(SINGLE_DIGITS.keys())
    num_pattern = f'(?:\\d+(?:\\.\\d+)?|{thai_digit_pattern})'
    op_pattern = '|'.join(sorted(OPERATORS.keys(), key=len, reverse=True))
    regex = re.compile(f'({num_pattern})\\s*({op_pattern})\\s*({num_pattern})')

    def replacer(m):
        left, op, right = m.group(1), m.group(2), m.group(3)
        left_val = SINGLE_DIGITS.get(left, left)
        right_val = SINGLE_DIGITS.get(right, right)
        symbol = OPERATORS[op]
        if symbol == '.' or op == 'ขีด':
            return f'{left_val}{symbol}{right_val}'
        else:
            return f'{left_val} {symbol} {right_val}'

    prev = ''
    while prev != text:
        prev = text
        text = regex.sub(replacer, text)
    return text

def normalize_text(text: str) -> str:
    """Applies Pass 1 and Pass 2 normalizations in sequence."""
    if not text:
        return text
    step1 = pass1_thai_numbers(text)
    step2 = pass2_math_operators(step1)
    return step2

def change_audio_speed(audio: np.ndarray, rate: float = AUDIO_SPEED_RATE, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Time-stretch audio using WSOLA (Waveform Similarity Overlap-Add).
    Slows down speech (rate < 1.0) while preserving original pitch and formants.
    """
    if rate >= 0.99 and rate <= 1.01:
        return audio
    if len(audio) < int(sample_rate * 0.1):
        return audio

    win_len = int(sample_rate * 0.025)  # 25ms window (400 samples at 16kHz)
    if win_len % 2 != 0:
        win_len += 1
    syn_hop = win_len // 2              # Synthesis hop (200 samples)
    ana_hop = int(round(syn_hop * rate)) # Analysis hop
    delta = win_len // 4                # Search window (+-100 samples)

    window = np.hanning(win_len).astype(np.float32)
    num_syn_frames = int(np.ceil((len(audio) - win_len) / ana_hop))
    out_len = win_len + num_syn_frames * syn_hop
    output = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)

    output[:win_len] += audio[:win_len] * window
    norm[:win_len] += window

    curr_ana_pos = ana_hop
    curr_syn_pos = syn_hop

    for _ in range(1, num_syn_frames):
        if curr_ana_pos + win_len + delta >= len(audio):
            break

        ref = output[curr_syn_pos : curr_syn_pos + win_len]
        min_offset = max(-delta, -curr_ana_pos)
        max_offset = min(delta, len(audio) - curr_ana_pos - win_len)
        target_candidates = audio[curr_ana_pos + min_offset : curr_ana_pos + max_offset + win_len]
        corrs = np.correlate(target_candidates, ref * window, mode="valid")
        best_offset = 0
        if len(corrs) > 0:
            best_offset = min_offset + int(np.argmax(corrs))

        chosen_frame = audio[curr_ana_pos + best_offset : curr_ana_pos + best_offset + win_len]
        output[curr_syn_pos : curr_syn_pos + win_len] += chosen_frame * window
        norm[curr_syn_pos : curr_syn_pos + win_len] += window

        curr_ana_pos += ana_hop
        curr_syn_pos += syn_hop

    mask = norm > 1e-4
    output[mask] /= norm[mask]
    return output

def audio_callback(indata, frames, time_info, status):
    if is_recording:
        audio_frames.append(indata.copy())

def start_recording():
    global is_recording, audio_frames
    if not is_recording:
        audio_frames = []
        is_recording = True
        print("[*] Listening...")

def stop_and_transcribe():
    global is_recording, audio_frames, is_win_pressed, is_z_pressed
    if is_recording:
        is_recording = False
        is_win_pressed = False
        is_z_pressed = False
        print("[*] Processing...")

        if not audio_frames:
            return

        frames_to_process = list(audio_frames)
        audio_frames = []

        # Convert audio data to Float32 1D-array
        audio_data = np.concatenate(frames_to_process, axis=0).flatten()

        # Prevent accidental quick releases (less than 0.3s)
        if len(audio_data) < SAMPLE_RATE * 0.3:
            return

        # Slow down audio speed without altering pitch
        if AUDIO_SPEED_RATE != 1.0:
            audio_data = change_audio_speed(audio_data, rate=AUDIO_SPEED_RATE, sample_rate=SAMPLE_RATE)

        def do_transcribe():
            try:
                # Save audio to temporary 16-bit PCM WAV for NeMo ASR
                int16_audio = (audio_data * 32767).clip(-32768, 32767).astype(np.int16)
                with wave.open(temp_wav_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(int16_audio.tobytes())

                # Transcribe with Pathumma FastConformer-RNNT
                with torch.no_grad(), contextlib.redirect_stderr(io.StringIO()):
                    output = asr_model.transcribe([temp_wav_path], verbose=False)

                text = ""
                if output and len(output) > 0:
                    first = output[0]
                    text = first.text if hasattr(first, "text") else str(first)
                    text = text.strip()

                if text:
                    text = normalize_text(text)
                    with minute_lock:
                        minute_sentences.append(text)
                    print(f">> Output: {text}")

                    # Save existing clipboard content
                    old_clip = pyperclip.paste()

                    # Copy transcribed text and simulate Ctrl+V
                    pyperclip.copy(text + " ")
                    time.sleep(0.05)
                    keyboard.send("ctrl+v")

                    # Restore previous clipboard content
                    time.sleep(0.1)
                    pyperclip.copy(old_clip)

            except Exception as e:
                print(f"[!] Transcription error: {e}")
            finally:
                if os.path.exists(temp_wav_path):
                    try:
                        os.remove(temp_wav_path)
                    except Exception:
                        pass

        threading.Thread(target=do_transcribe, daemon=True).start()

def on_key_event(event):
    global is_win_pressed, is_z_pressed
    name = (event.name or "").lower()

    if name in WIN_KEYS:
        is_win_pressed = (event.event_type == keyboard.KEY_DOWN)
    elif name == "z":
        is_z_pressed = (event.event_type == keyboard.KEY_DOWN)

    # Check if both Win and Z are held down
    win_down = is_win_pressed or keyboard.is_pressed("win")
    z_down = is_z_pressed or keyboard.is_pressed("z")
    both_down = win_down and z_down

    if both_down:
        if not is_recording:
            start_recording()
    else:
        if is_recording:
            stop_and_transcribe()

# --- 1-Minute Statistics Timer ---
minute_sentences = []
minute_lock = threading.Lock()

def minute_timer_worker():
    while True:
        time.sleep(60)
        with minute_lock:
            current_batch = list(minute_sentences)
            minute_sentences.clear()

        # Combine all sentences from the 1-minute window and calculate length
        combined_text = "".join(s.strip() for s in current_batch)
        char_count = len(combined_text)
        result = char_count / 3

        print("\n" + "=" * 55)
        print("[⏱️ 1-Minute Statistics Summary]")
        print(f"- Sentences spoken: {len(current_batch)}")
        print(f"- Total character count: {char_count}")
        print(f"- Result (char count / 3): {result:.2f}")
        if current_batch:
            preview = " ".join(current_batch)
            if len(preview) > 100:
                preview = preview[:97] + "..."
            print(f"- Combined text: \"{preview}\"")
        print("=" * 55 + "\n")

if __name__ == "__main__":
    # Start 1-minute timer thread in background
    timer_thread = threading.Thread(target=minute_timer_worker, daemon=True)
    timer_thread.start()

    # Open audio capture stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback
    )

    # Hook global keyboard hotkey
    keyboard.hook(on_key_event)

    with stream:
        keyboard.wait()
