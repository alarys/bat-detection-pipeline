"""Automated Pipeline Regression Test Suite.
Validates NFS storage access, resampler file expansion, upsampling integrity, 
and ground-truth detection matches (species, count, and interval alignment).
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np

# Storage & Model Configuration
NFS_MOUNT = "/mnt/tweety/verified"
BASELINE_WAV = os.path.join(NFS_MOUNT, "XC883525 - Little Brown Myotis - Myotis lucifugus.wav")
BASELINE_CSV = os.path.join(NFS_MOUNT, "XC883525.BirdNET.results-brown-bat-good.csv")

TEST_DIR = "/tmp/batty_regression"
RESAMPLED_WAV = os.path.join(TEST_DIR, "XC883525 - Little Brown Myotis - Myotis lucifugus.wav")
TEST_CSV = os.path.join(TEST_DIR, "XC883525 - Little Brown Myotis - Myotis lucifugus.csv")
MODEL_PATH = "checkpoints/bats/v1.0/BattyBirdNET-USA-WEST-256kHz.tflite"

# Regression Targets & Tolerances
MIN_EXPANSION_RATIO = 1.25
MIN_CONFIDENCE = 0.99
TIMESTAMP_TOLERANCE_SEC = 0.5  # Seconds allowed for start/end timestamp drift


def run_regression_test():
    print("=== Running Audio Pipeline Regression Test ===")

    # Step 0: Pre-flight Verification
    print("[0/4] Verifying storage mounts and files...")
    if not os.path.exists(BASELINE_WAV):
        print(f"[FAIL] Baseline WAV missing: {BASELINE_WAV}\nEnsure NFS share is mounted.")
        sys.exit(1)
    
    if not os.path.exists(BASELINE_CSV):
        print(f"[FAIL] Baseline ground-truth CSV missing: {BASELINE_CSV}")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"[FAIL] Classifier model missing: {MODEL_PATH}")
        sys.exit(1)

    os.makedirs(TEST_DIR, exist_ok=True)

    # Step 1: Resampler Execution
    print("[1/4] Running audio resampler (192kHz -> 256kHz)...")
    try:
        import resample
        resample.INPUT_DIR = NFS_MOUNT
        resample.OUTPUT_DIR = TEST_DIR
        resample.process_file(os.path.basename(BASELINE_WAV))
    except Exception as e:
        print(f"[FAIL] Resampler execution failed: {e}")
        sys.exit(1)

    # Step 2: Resampling File Expansion Integrity
    print("[2/4] Validating resampled file size expansion...")
    orig_size = os.path.getsize(BASELINE_WAV)
    resampled_size = os.path.getsize(RESAMPLED_WAV)
    ratio = resampled_size / orig_size

    print(f"   Original:  {orig_size / (1024*1024):.2f} MB")
    print(f"   Resampled: {resampled_size / (1024*1024):.2f} MB")
    print(f"   Ratio:     {ratio:.3f}x")

    if ratio < MIN_EXPANSION_RATIO:
        print(f"[FAIL] File size loss or truncation detected ({ratio:.3f}x < {MIN_EXPANSION_RATIO}x).")
        sys.exit(1)

    # Step 3: Model Classification
    print("[3/4] Running BattyBirdNET classifier...")
    cmd = [
        sys.executable, "analyze.py",
        "--i", TEST_DIR,
        "--o", TEST_DIR,
        "--classifier", MODEL_PATH,
        "--min_conf", str(MIN_CONFIDENCE),
        "--rtype", "csv"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)

    if res.returncode != 0:
        print(f"[FAIL] analyze.py execution failed:\n{res.stderr}")
        sys.exit(1)

    if not os.path.exists(TEST_CSV):
        print(f"[FAIL] Output CSV not generated: {TEST_CSV}")
        sys.exit(1)

    # Step 4: Ground-Truth Detection Comparison
    print("[4/4] Comparing output detections against ground-truth CSV...")
    
    df_expected = pd.read_csv(BASELINE_CSV)
    df_actual = pd.read_csv(TEST_CSV)

    # Clean up column names in case of whitespace
    df_expected.columns = df_expected.columns.str.strip()
    df_actual.columns = df_actual.columns.str.strip()

    # 4a. Check Detection Count
    expected_count = len(df_expected)
    actual_count = len(df_actual)
    
    if expected_count != actual_count:
        print(f"[FAIL] Detection count mismatch!")
        print(f"   Expected count: {expected_count}")
        print(f"   Actual count:   {actual_count}")
        sys.exit(1)

    # 4b. Check Species Alignment
    expected_species = df_expected["Common name"].tolist()
    actual_species = df_actual["Common name"].tolist()

    if expected_species != actual_species:
        print(f"[FAIL] Species output mismatch!")
        print(f"   Expected: {expected_species}")
        print(f"   Actual:   {actual_species}")
        sys.exit(1)

    # 4c. Check Start and End Timestamp Alignment
    start_diffs = np.abs(df_expected["Start (sec)"].values - df_actual["Start (sec)"].values)
    end_diffs = np.abs(df_expected["End (sec)"].values - df_actual["End (sec)"].values)

    max_start_diff = np.max(start_diffs) if len(start_diffs) > 0 else 0
    max_end_diff = np.max(end_diffs) if len(end_diffs) > 0 else 0

    if max_start_diff > TIMESTAMP_TOLERANCE_SEC or max_end_diff > TIMESTAMP_TOLERANCE_SEC:
        print(f"[FAIL] Detection timestamps drifted beyond tolerance ({TIMESTAMP_TOLERANCE_SEC}s)!")
        print(f"   Max Start Time Drift: {max_start_diff:.4f}s")
        print(f"   Max End Time Drift:   {max_end_diff:.4f}s")
        sys.exit(1)

    print("\n[PASS] All regression tests passed successfully!")
    print(f"   - Verified Resample Ratio: {ratio:.3f}x")
    print(f"   - Confirmed {actual_count} expected detection(s) for species: {set(actual_species)}")
    print(f"   - Max Timestamp Drift: {max(max_start_diff, max_end_diff):.4f}s")


if __name__ == "__main__":
    run_regression_test()