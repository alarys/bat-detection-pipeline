"""Ultrasonic Audio Resampler (192kHz -> 256kHz) with Pre-Flight Header Validation.
Validates sample rates and file durations before loading into RAM to prevent memory issues.
Outputs errors to STDERR and maintains a persistent skipped/error log.
"""

import os
import sys
import gc
import soundfile as sf
import librosa

INPUT_DIR = "/mnt/tweety/Raw/"
OUTPUT_DIR = "/mnt/c/AudioMoth/Ready_256kHz"
ERROR_LOG_PATH = "resample_errors.log"

EXPECTED_SR = 192000
TARGET_SR = 256000
MAX_DURATION_SEC = 1800  # 30-minute threshold to prevent loading massive single-stream files

# Set DELETE_NON_192KHZ_FILES = True if you want to automatically delete non-192kHz files from /mnt/tweety/
DELETE_NON_192KHZ_FILES = False


def log_error(msg: str):
    """Outputs error messages to both STDERR and a dedicated log file."""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as err_file:
        err_file.write(msg + "\n")


def validate_wav_header(in_path: str, file_name: str) -> bool:
    """Inspects header metadata without loading raw audio into memory."""
    try:
        info = sf.info(in_path)

        # 1. Sample Rate Check
        if info.samplerate != EXPECTED_SR:
            reason = f"[SKIP] Non-192kHz file ({info.samplerate} Hz): {file_name}"
            log_error(reason)

            if DELETE_NON_192KHZ_FILES:
                try:
                    os.remove(in_path)
                    log_error(f"  [DELETED] {in_path}")
                except Exception as del_ex:
                    log_error(f"  [ERROR] Failed to delete {in_path}: {del_ex}")
            return False

        # 2. File Duration / Corruption Check
        if info.duration <= 0 or info.duration > MAX_DURATION_SEC:
            reason = f"[SKIP] Invalid/unsupported duration ({info.duration:.1f}s): {file_name}"
            log_error(reason)
            return False

        return True

    except Exception as ex:
        reason = f"[ERROR] Header read failure for {file_name}: {ex}"
        log_error(reason)
        return False


def main():
    print("=== Ultrasonic Audio Resampler (Header-Validated Sequential Mode) ===", flush=True)

    if not os.path.exists(INPUT_DIR):
        err = f"[FATAL] Input directory missing: {INPUT_DIR}"
        log_error(err)
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wav_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".wav")]

    if not wav_files:
        print(f"No .wav files found in {INPUT_DIR}.", flush=True)
        sys.exit(0)

    print(f"Found {len(wav_files)} total matching files in input directory.\n", flush=True)

    processed_count = 0
    skipped_count = 0

    for idx, file_name in enumerate(wav_files, 1):
        in_path = os.path.join(INPUT_DIR, file_name)
        out_path = os.path.join(OUTPUT_DIR, file_name)

        # Idempotency check: Skip if already resampled in destination
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            continue

        # Pre-flight header inspection
        if not validate_wav_header(in_path, file_name):
            skipped_count += 1
            continue

        print(f"[{idx}/{len(wav_files)}] Resampling: {file_name}", flush=True)

        try:
            # Load audio at native 192kHz
            y, sr = librosa.load(in_path, sr=EXPECTED_SR)

            # Resample to 256kHz for BattyBirdNET
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

            # Preserve explicit PCM_24 dynamic range
            sf.write(out_path, y_resampled, TARGET_SR, subtype="PCM_24")

            processed_count += 1

            # Free RAM immediately
            del y, y_resampled
            gc.collect()

        except Exception as ex:
            fail_msg = f"[ERROR] Resampling execution failed for {file_name}: {ex}"
            log_error(fail_msg)
            skipped_count += 1

    print(f"\nCompleted! Processed: {processed_count} files | Skipped/Failed: {skipped_count} files.", flush=True)
    if os.path.exists(ERROR_LOG_PATH):
        print(f"Skipped/error file log written to: {ERROR_LOG_PATH}", flush=True)


if __name__ == "__main__":
    main()
