"""Module to analyze audio samples optimized for bat CSV detection logging.

================================================================================
USAGE & PIPELINE DOCUMENTATION
================================================================================
Example Command:
  python analyze.py \
    --i /mnt/c/AudioMoth/Ready_256kHz \
    --o /mnt/c/AudioMoth/Results \
    --classifier checkpoints/bats/v1.0/BattyBirdNET-USA-WEST-256kHz.tflite \
    --min_conf 0.99 \
    --rtype csv \
    --threads 1 \
    --batchsize 4
================================================================================
"""

import argparse
import datetime
import json
import operator
import os
import sys
from multiprocessing import Pool, freeze_support

import numpy as np

import audio
import config as cfg
import model
import species
import utils

LOG_SKIP_PATH = "analyze_skips.log"


def log_skip(msg: str):
    """Outputs skip message to stdout and appends to skip log file."""
    print(msg, flush=True)
    with open(LOG_SKIP_PATH, "a", encoding="utf-8") as log_f:
        log_f.write(msg + "\n")


def loadCodes():
    """Loads eBird codes."""
    with open(cfg.CODES_FILE, "r") as cfile:
        return json.load(cfile)


def get_expected_csv_path(fpath: str) -> str:
    """Calculates the target CSV output path for a given audio file path."""
    abs_fpath = os.path.abspath(fpath)
    abs_input_dir = os.path.abspath(cfg.INPUT_PATH)
    abs_output_dir = os.path.abspath(cfg.OUTPUT_PATH)

    if not abs_output_dir.rsplit(".", 1)[-1].lower() in ["txt", "csv"]:
        rel_path = os.path.relpath(abs_fpath, abs_input_dir)
        csv_rel_path = os.path.splitext(rel_path)[0] + ".csv"
        return os.path.join(abs_output_dir, csv_rel_path)
    return abs_output_dir


def saveResultFile(r: dict[str, list], path: str, afile_path: str) -> bool:
    """Saves formatted CSV results to disk.
    
    Returns:
        bool: True if detections were found and written, False if no detections met threshold.
    """
    filename = os.path.basename(afile_path)
    rows = []

    # Sort timestamps strictly ascending
    sorted_timestamps = sorted(r.keys(), key=lambda t: float(t.split("-", 1)[0]))

    for timestamp in sorted_timestamps:
        start, end = timestamp.split("-", 1)
        start_sec = float(start)
        end_sec = float(end)

        for c in r[timestamp]:
            conf = float(c[1])
            if conf >= cfg.MIN_CONFIDENCE and (not cfg.SPECIES_LIST or c[0] in cfg.SPECIES_LIST):
                label = cfg.TRANSLATED_LABELS[cfg.LABELS.index(c[0])]
                scientific_name = label.split("_", 1)[0]
                common_name = label.split("_", 1)[-1]

                rows.append("{},{:.4f},{:.4f},{},{},{:.7f}\n".format(
                    filename,
                    start_sec,
                    end_sec,
                    scientific_name,
                    common_name,
                    conf
                ))

    # Ensure parent output directory exists
    target_dir = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)

    header = "Filename,Start (sec),End (sec),Scientific name,Common name,Confidence\n"
    
    # Save CSV file (contains detections or empty header to mark file as processed)
    with open(path, "w", encoding="utf-8") as rfile:
        rfile.write(header + "".join(rows))

    return len(rows) > 0


def getRawAudioFromFile(fpath: str):
    """Reads audio file and splits into chunks."""
    sig, rate = audio.openAudioFile(fpath, cfg.SAMPLE_RATE)
    return audio.splitSignal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)


def predict(samples):
    """Predicts scores for given audio samples."""
    data = np.array(samples, dtype="float32")
    prediction = model.predict(data)

    if cfg.APPLY_SIGMOID:
        prediction = model.flat_sigmoid(np.array(prediction), sensitivity=-cfg.SIGMOID_SENSITIVITY)

    return prediction


def analyzeFile(item):
    """Analyzes a single audio file."""
    fpath: str = item[0]
    cfg.set_config(item[1])
    results = {}
    start_time = datetime.datetime.now()

    target_csv = get_expected_csv_path(fpath)
    filename = os.path.basename(fpath)

    # Idempotent skip check
    if os.path.exists(target_csv):
        log_skip(f"[SKIP] Existing report found: {os.path.basename(target_csv)}")
        return "SKIPPED", results

    print(f"Analyzing {fpath}", flush=True)

    try:
        chunks = getRawAudioFromFile(fpath)
    except Exception as ex:
        print(f"Error: Cannot open audio file {fpath}", flush=True)
        utils.writeErrorLog(ex)
        return "FAILED", results

    try:
        start, end = 0, cfg.SIG_LENGTH
        samples = []
        timestamps = []

        for chunk_index, chunk in enumerate(chunks):
            samples.append(chunk)
            timestamps.append([start, end])

            start += cfg.SIG_LENGTH - cfg.SIG_OVERLAP
            end = start + cfg.SIG_LENGTH

            if len(samples) < cfg.BATCH_SIZE and chunk_index < len(chunks) - 1:
                continue

            p = predict(samples)

            for i in range(len(samples)):
                s_start, s_end = timestamps[i]
                pred = p[i]

                p_labels = zip(cfg.LABELS, pred)
                p_sorted = sorted(p_labels, key=operator.itemgetter(1), reverse=True)
                
                results[f"{s_start}-{s_end}"] = [[label, str(score)] for label, score in p_sorted]

            samples = []
            timestamps = []

    except Exception as ex:
        print(f"Error: Cannot analyze audio file {fpath}.\n", flush=True)
        utils.writeErrorLog(ex)
        return "FAILED", results

    try:
        has_detections = saveResultFile(results, target_csv, fpath)
    except Exception as ex:
        print(f"Error: Cannot save result for {fpath}.\nDetails: {ex}", flush=True)
        utils.writeErrorLog(ex)
        return "FAILED", results

    delta_time = (datetime.datetime.now() - start_time).total_seconds()

    if has_detections:
        print(f"Finished {fpath} in {delta_time:.2f} seconds [DETECTIONS FOUND]", flush=True)
        return "PROCESSED_WITH_DETECTIONS", results
    else:
        print(f"[NO DETECTION] Zero detections found for: {filename} ({delta_time:.2f}s)", flush=True)
        return "PROCESSED_NO_DETECTIONS", results


def validate_output_target(output_path: str) -> bool:
    """Verifies output destination accessibility and writeability."""
    abs_out = os.path.abspath(output_path)
    parts = abs_out.split(os.sep)
    
    if len(parts) >= 3 and parts[1] == "mnt":
        mount_point = os.sep + os.path.join(parts[1], parts[2])
        if os.path.exists(mount_point):
            is_mounted = os.path.ismount(mount_point)
            is_empty = len(os.listdir(mount_point)) == 0
            if not is_mounted and is_empty:
                print(f"\n[ERROR] Unmounted volume detected: {mount_point}", flush=True)
                return False

    target_dir = abs_out if not os.path.splitext(abs_out)[1] else os.path.dirname(abs_out)
    try:
        os.makedirs(target_dir, exist_ok=True)
        test_file = os.path.join(target_dir, ".write_test.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        print(f"\n[ERROR] Output path is not writeable: {target_dir}\nDetails: {e}", flush=True)
        return False

    return True


if __name__ == "__main__":
    freeze_support()

    parser = argparse.ArgumentParser(description="Analyze audio files with BattyBirdNET")
    parser.add_argument("--i", default="example/", help="Path to input file or folder.")
    parser.add_argument("--o", default="example/", help="Path to output file or folder.")
    parser.add_argument("--lat", type=float, default=-1, help="Latitude (-1 to ignore).")
    parser.add_argument("--lon", type=float, default=-1, help="Longitude (-1 to ignore).")
    parser.add_argument("--week", type=int, default=-1, help="Week of year [1, 48].")
    parser.add_argument("--slist", default="", help="Path to species list.")
    parser.add_argument("--sensitivity", type=float, default=1.0, help="Detection sensitivity [0.5, 1.5].")
    parser.add_argument("--min_conf", type=float, default=0.1, help="Minimum confidence [0.01, 0.99].")
    parser.add_argument("--overlap", type=float, default=0.0, help="Segment overlap [0.0, 2.9].")
    parser.add_argument("--rtype", default="csv", help="Output format.")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads.")
    parser.add_argument("--batchsize", type=int, default=1, help="Batch size.")
    parser.add_argument("--locale", default="en", help="Locale.")
    parser.add_argument("--sf_thresh", type=float, default=0.03, help="Species threshold.")
    parser.add_argument("--classifier", default=None, help="Path to classifier model.")

    args = parser.parse_args()

    errors_found = False

    if not os.path.exists(args.i):
        print(f"\n[ERROR] Input path does not exist: {args.i}", flush=True)
        errors_found = True

    if args.classifier is not None and not os.path.exists(args.classifier):
        print(f"\n[ERROR] Custom classifier model not found: {args.classifier}", flush=True)
        errors_found = True

    if not validate_output_target(args.o):
        errors_found = True

    if errors_found:
        print("\nAborting execution due to missing path or mount errors.\n", flush=True)
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    cfg.MODEL_PATH = os.path.join(script_dir, cfg.MODEL_PATH)
    cfg.LABELS_FILE = os.path.join(script_dir, cfg.LABELS_FILE)
    cfg.TRANSLATED_LABELS_PATH = os.path.join(script_dir, cfg.TRANSLATED_LABELS_PATH)
    cfg.MDATA_MODEL_PATH = os.path.join(script_dir, cfg.MDATA_MODEL_PATH)
    cfg.CODES_FILE = os.path.join(script_dir, cfg.CODES_FILE)
    cfg.ERROR_LOG_FILE = os.path.join(script_dir, cfg.ERROR_LOG_FILE)

    cfg.CODES = loadCodes()
    cfg.LABELS = utils.readLines(cfg.LABELS_FILE)

    if args.classifier is not None:
        cfg.CUSTOM_CLASSIFIER = args.classifier
        cfg.LABELS_FILE = args.classifier.replace(".tflite", "_Labels.txt")
        cfg.LABELS = utils.readLines(cfg.LABELS_FILE)
        args.lat = -1
        args.lon = -1
        args.locale = "en"

    cfg.TRANSLATED_LABELS = cfg.LABELS
    cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK = args.lat, args.lon, args.week
    cfg.LOCATION_FILTER_THRESHOLD = max(0.01, min(0.99, float(args.sf_thresh)))
    cfg.SPECIES_LIST = None

    cfg.INPUT_PATH = os.path.abspath(args.i)
    cfg.OUTPUT_PATH = os.path.abspath(args.o)

    if os.path.isdir(cfg.INPUT_PATH):
        cfg.FILE_LIST = utils.collect_audio_files(cfg.INPUT_PATH)
    else:
        cfg.FILE_LIST = [cfg.INPUT_PATH]

    cfg.MIN_CONFIDENCE = max(0.01, min(0.99, float(args.min_conf)))
    cfg.SIGMOID_SENSITIVITY = max(0.5, min(1.0 - (float(args.sensitivity) - 1.0), 1.5))
    cfg.SIG_OVERLAP = max(0.0, min(2.9, float(args.overlap)))
    cfg.RESULT_TYPE = "csv"

    if os.path.isdir(cfg.INPUT_PATH):
        cfg.CPU_THREADS = max(1, int(args.threads))
        cfg.TFLITE_THREADS = 1
    else:
        cfg.CPU_THREADS = 1
        cfg.TFLITE_THREADS = max(1, int(args.threads))

    cfg.BATCH_SIZE = max(1, int(args.batchsize))

    flist = [(f, cfg.get_config()) for f in cfg.FILE_LIST]
    total_source_files = len(flist)

    # Pre-Flight Report Expectations
    existing_reports = 0
    for entry in flist:
        expected_csv = get_expected_csv_path(entry[0])
        if os.path.exists(expected_csv):
            existing_reports += 1

    expected_new_reports = total_source_files - existing_reports

    print("\n" + "=" * 60, flush=True)
    print("PRE-FLIGHT JOB SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Total Input Audio Files   : {total_source_files}", flush=True)
    print(f"Existing CSV Reports      : {existing_reports}", flush=True)
    print(f"Expected New CSV Reports  : {expected_new_reports}", flush=True)
    print("=" * 60 + "\n", flush=True)

    results_status = []

    if cfg.CPU_THREADS < 2:
        for entry in flist:
            status, _ = analyzeFile(entry)
            results_status.append(status)
    else:
        with Pool(cfg.CPU_THREADS) as p:
            res = p.map(analyzeFile, flist)
            results_status = [r[0] for r in res]

    # Post-Run Audit & Summary Verification
    with_detections = results_status.count("PROCESSED_WITH_DETECTIONS")
    no_detections = results_status.count("PROCESSED_NO_DETECTIONS")
    newly_processed_total = with_detections + no_detections
    skipped_count = results_status.count("SKIPPED")
    failed_count = results_status.count("FAILED")

    total_accounted = newly_processed_total + skipped_count + failed_count
    total_reports_on_disk = existing_reports + newly_processed_total

    print("\n" + "=" * 60, flush=True)
    print("ANALYZE RUN SUMMARY AUDIT", flush=True)
    print("=" * 60, flush=True)
    print(f"Total Input Audio Files      : {total_source_files}", flush=True)
    print(f"Newly Processed (With Hits)  : {with_detections}", flush=True)
    print(f"Newly Processed (Zero Hits)  : {no_detections}", flush=True)
    print(f"Skipped (Report Existed)     : {skipped_count}", flush=True)
    print(f"Failed Analysis Jobs         : {failed_count}", flush=True)
    print("-" * 60, flush=True)
    print(f"Total Accounted Files        : {total_accounted}", flush=True)
    print(f"Total CSV Reports on Disk    : {total_reports_on_disk} / {total_source_files}", flush=True)
    print("=" * 60, flush=True)

    # Verification: Ensure Total Accounted == Source Files AND Total Reports on Disk == Source Files
    if total_accounted != total_source_files or total_reports_on_disk != total_source_files or failed_count > 0:
        print("[ERROR] Mismatch detected or analysis failures occurred!", flush=True)
        print(f"        Expected Total Files  : {total_source_files}", flush=True)
        print(f"        Total Accounted Files : {total_accounted}", flush=True)
        print(f"        Total CSVs on Disk    : {total_reports_on_disk}", flush=True)
        print(f"        Failed Jobs           : {failed_count}", flush=True)
    else:
        print("[SUCCESS] All files accounted for and 1:1 CSV report files present on disk.", flush=True)

    if os.path.exists(LOG_SKIP_PATH):
        print(f"Skip log written to: {LOG_SKIP_PATH}", flush=True)