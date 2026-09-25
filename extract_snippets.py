"""Production AudioMoth Bat Snippet Extractor.
Parses analysis CSVs, consolidates detection spans via sliding window aggregation (+/- 5s padding),
reads raw 192kHz WAV files directly without upsampling, embeds GUANO/RIFF metadata,
and tracks progress for resume capabilities.
Note on GUANO_BACKUP:
    The `guano` library creates a `GUANO_BACKUP` directory inside OUTPUT_DIR as a safety 
    mechanism while modifying WAV headers. Once an extraction run completes successfully, 
    the files in `GUANO_BACKUP` are redundant and can be safely deleted to free space:
        rm -rf /mnt/c/AudioMoth/Archive/GUANO_BACKUP
"""

import glob
import os
import sys
import guano
import pandas as pd
import soundfile as sf
from mutagen.id3 import ID3, TIT2, TPE1, COMM
from mutagen.wave import WAVE

# Path Configuration
CSV_DIR = "/mnt/tweety/Archive/reports"
RAW_AUDIO_DIR = "/mnt/tweety/bat-files"
OUTPUT_DIR = "/mnt/c/AudioMoth/Archive"
MANIFEST_LOG = os.path.join(OUTPUT_DIR, ".processed_manifest.log")

# Processing Parameters
PADDING_SEC = 5.0  # +/- 5 seconds around detection window
MIN_CONFIDENCE = 0.90  # Detection threshold
TARGET_SUBTYPE = "PCM_24"  # Native 24-bit PCM dynamic range
SLIDING_WINDOW_BOUND = 20.0  # Max raw window length (results in max ~30s snippet with padding)


def validate_nfs_mount(path: str) -> bool:
    """Ensures NFS volume path exists, is accessible, and contains files."""
    if not os.path.exists(path):
        print(f"\n[ERROR] Path does not exist:\n  --> Expected: {path}", flush=True)
        return False
    try:
        contents = os.listdir(path)
        if not contents:
            print(f"\n[ERROR] Mount path is empty:\n  --> {path}", flush=True)
            return False
    except Exception as e:
        print(f"\n[ERROR] Unable to read NFS path {path}: {e}", flush=True)
        return False
    return True


def get_species_code(scientific_name: str) -> str:
    """Generates standard 4-letter species code (e.g., 'Myotis lucifugus' -> 'MYLU')."""
    parts = str(scientific_name).strip().split()
    if len(parts) >= 2:
        return (parts[0][:2] + parts[1][:2]).upper()
    return str(scientific_name)[:4].upper()


def load_processed_manifest() -> set:
    """Loads log of previously processed CSV files to allow seamless resumption."""
    if os.path.exists(MANIFEST_LOG):
        with open(MANIFEST_LOG, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def log_processed_file(csv_filename: str):
    """Appends processed CSV to manifest log."""
    with open(MANIFEST_LOG, "a", encoding="utf-8") as f:
        f.write(f"{csv_filename}\n")


def process_csv_and_extract():
    print("=== Running Pre-Flight Sanity Checks ===")
    if not validate_nfs_mount(RAW_AUDIO_DIR):
        print("\nAborting: Input NFS mount check failed.", flush=True)
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
    processed_set = load_processed_manifest()

    total_csv_files = len(csv_files)
    skipped_no_detection = 0
    already_processed = 0
    files_to_process = []

    print("\n=== Scanning CSV Reports ===")
    for csv_path in csv_files:
        filename = os.path.basename(csv_path)

        if filename == "manifest.csv" or filename.startswith("."):
            continue

        if filename in processed_set:
            already_processed += 1
            continue

        try:
            # Check for "no detection" or empty files efficiently
            if os.path.getsize(csv_path) == 0:
                skipped_no_detection += 1
                log_processed_file(filename)
                continue

            df = pd.read_csv(csv_path)
            if df.empty or "Confidence" not in df.columns:
                skipped_no_detection += 1
                log_processed_file(filename)
                continue

            df_valid = df[df["Confidence"] >= MIN_CONFIDENCE]
            if df_valid.empty:
                skipped_no_detection += 1
                log_processed_file(filename)
                continue

            files_to_process.append((csv_path, df_valid))

        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            skipped_no_detection += 1
            log_processed_file(filename)
        except Exception as e:
            print(f"[WARN] Read failure on {filename}: {e}")

    print(f"Total CSV Files Scanned:        {total_csv_files}")
    print(f"Previously Processed (Skipped): {already_processed}")
    print(f"No Detection CSVs (Skipped):     {skipped_no_detection}")
    print(f"Active Files Pending Extraction: {len(files_to_process)}")

    if not files_to_process:
        print("\nNo pending detections to process. Exiting.")
        return

    total_snippets_created = 0

    print("\n=== Executing Audio Snippet Extraction ===")
    for csv_path, df in files_to_process:
        csv_filename = os.path.basename(csv_path)
        raw_filename = df["Filename"].iloc[0]
        raw_filepath = os.path.join(RAW_AUDIO_DIR, raw_filename)

        if not os.path.exists(raw_filepath):
            print(f"[SKIP] Raw audio missing (will retry next run): {raw_filepath}")
            continue

        try:
            info = sf.info(raw_filepath)
            native_sr = info.samplerate  # Preserve native sample rate (e.g. 192 kHz)
            total_duration = info.duration
        except Exception as e:
            print(f"[ERROR] Reading audio headers failed for {raw_filename}: {e}")
            continue

        # Sort detections by time
        df_sorted = df.sort_values("Start (sec)")

        # Combine contiguous detections using sliding window logic
        windows = []
        current_window = None

        for _, row in df_sorted.iterrows():
            det_start = float(row["Start (sec)"])
            det_end = float(row["End (sec)"])
            species_code = get_species_code(row["Scientific name"])
            common_name = str(row["Common name"])
            scientific_name = str(row["Scientific name"])
            conf = float(row["Confidence"])

            if current_window is None:
                current_window = {
                    "start": det_start,
                    "end": det_end,
                    "species": {species_code},
                    "common_names": {common_name},
                    "scientific_names": {scientific_name},
                    "max_conf": conf,
                }
            else:
                # Extend window if detection falls within padded window boundary
                if (
                    det_start <= current_window["end"] + (PADDING_SEC * 2)
                ) and (
                    (det_end - current_window["start"]) <= SLIDING_WINDOW_BOUND
                ):
                    current_window["end"] = max(current_window["end"], det_end)
                    current_window["species"].add(species_code)
                    current_window["common_names"].add(common_name)
                    current_window["scientific_names"].add(scientific_name)
                    current_window["max_conf"] = max(
                        current_window["max_conf"], conf
                    )
                else:
                    windows.append(current_window)
                    current_window = {
                        "start": det_start,
                        "end": det_end,
                        "species": {species_code},
                        "common_names": {common_name},
                        "scientific_names": {scientific_name},
                        "max_conf": conf,
                    }

        if current_window is not None:
            windows.append(current_window)

        # Slice audio and write snippets
        for w in windows:
            padded_start = max(0.0, w["start"] - PADDING_SEC)
            padded_end = min(total_duration, w["end"] + PADDING_SEC)

            start_frame = int(padded_start * native_sr)
            stop_frame = int(padded_end * native_sr)

            # Read directly as integer samples to preserve exact uncompressed PCM bits
            data, sr = sf.read(
                raw_filepath,
                start=start_frame,
                stop=stop_frame,
                dtype="int32",
            )

            # Build Filename
            timestamp = os.path.splitext(raw_filename)[0]
            species_str = "-".join(sorted(w["species"]))
            conf_int = int(round(w["max_conf"] * 100))

            out_filename = (
                f"{timestamp}_"
                f"{species_str}_"
                f"{padded_start:06.2f}s-{padded_end:06.2f}s_"
                f"conf{conf_int:02d}.wav"
            )
            out_filepath = os.path.join(OUTPUT_DIR, out_filename)

            # Write file (unmodified sample rate, 24-bit PCM)
            sf.write(out_filepath, data, sr, subtype=TARGET_SUBTYPE)

            # Embed GUANO Metadata
            try:
                gfile = guano.GuanoFile(out_filepath)
                gfile["GUANO|Version"] = "1.0"
                gfile["Species Auto"] = ", ".join(sorted(w["common_names"]))
                gfile["Species Manual"] = ", ".join(
                    sorted(w["scientific_names"])
                )
                gfile["Note"] = (
                    f"OriginalFile={raw_filename}; "
                    f"MaxConf={w['max_conf']:.4f}; "
                    f"Window={padded_start:.2f}s-{padded_end:.2f}s"
                )
                gfile["BattyBirdNET|Confidence"] = float(w["max_conf"])
                gfile.write()
            except Exception as e:
                print(f"[WARN] GUANO write failed for {out_filename}: {e}")

            # Embed ID3/RIFF Metadata Tags into WAV
            try:
                audio_meta = WAVE(out_filepath)
                if audio_meta.tags is None:
                    audio_meta.add_tags()

                title_str = ", ".join(sorted(w["common_names"]))
                artist_str = "AudioMoth Bat Detector"
                comment_str = f"Max Conf: {w['max_conf']:.4f}; Source: {raw_filename}"

                # Standard ID3 Frame mapping for WAV containers
                audio_meta.tags.add(TIT2(encoding=3, text=[title_str]))
                audio_meta.tags.add(TPE1(encoding=3, text=[artist_str]))
                audio_meta.tags.add(COMM(encoding=3, lang="eng", desc="", text=[comment_str]))
                
                audio_meta.save()
            except Exception as e:
                print(f"[WARN] WAV metadata write failed for {out_filename}: {e}")

            total_snippets_created += 1

        # Mark CSV as processed in manifest
        log_processed_file(csv_filename)

    print("\n=== Extraction Summary ===")
    print(f"Total CSV Files Evaluated:     {total_csv_files}")
    print(f"Skipped (No Detections/Empty): {skipped_no_detection}")
    print(f"Skipped (Previously Processed): {already_processed}")
    print(f"Active CSV Files Processed:     {len(files_to_process)}")
    print(f"Total Audio Snippets Created:   {total_snippets_created}")
    print(f"Archive Output Destination:     {OUTPUT_DIR}")


if __name__ == "__main__":
    process_csv_and_extract()