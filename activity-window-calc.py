"""CSV Activity Window Boundary Calculator.
Analyzes CSV outputs to compute average continuous bat activity duration.
"""

import glob
import os
import pandas as pd

CSV_DIR = "/mnt/c/AudioMoth/Results"
MIN_CONFIDENCE = 0.90


def calculate_average_activity_window():
    csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {CSV_DIR}")
        return

    durations = []

    for csv_path in csv_files:
        if csv_path.endswith("manifest.csv"):
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if df.empty or "Start (sec)" not in df.columns:
            continue

        # Filter detections
        df = df[df["Confidence"] >= MIN_CONFIDENCE].sort_values("Start (sec)")
        if df.empty:
            continue

        # Group overlapping or contiguous detections (gap <= 2.0s)
        current_start = None
        current_end = None

        for _, row in df.iterrows():
            s, e = float(row["Start (sec)"]), float(row["End (sec)"])
            if current_start is None:
                current_start, current_end = s, e
            elif s <= current_end + 2.0:
                current_end = max(current_end, e)
            else:
                durations.append(current_end - current_start)
                current_start, current_end = s, e

        if current_start is not None:
            durations.append(current_end - current_start)

    if durations:
        avg_window = sum(durations) / len(durations)
        print("=== Bat Activity Window Analysis ===")
        print(f"Total Detections Groups Analyzed: {len(durations)}")
        print(f"Mean Activity Window Duration: {avg_window:.2f} seconds")
        print(f"Recommended Sliding Window Bound: {round(avg_window + 10.0, 1)} seconds (includes ±5s padding)")
    else:
        print("No valid detections found to calculate average window.")


if __name__ == "__main__":
    calculate_average_activity_window()