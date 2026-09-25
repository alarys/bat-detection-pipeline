AudioMoth Ultrasonic Bat Detection Pipeline README.md

```
# AudioMoth Ultrasonic Bat Detection Pipeline

This repository contains an end-to-end data processing, resampling, and classification pipeline for analyzing uncompressed 192 kHz [AudioMoth](https://www.openacousticdevices.info/) bat recordings using [BattyBirdNET-Analyzer](https://github.com/rdz-oss/BattyBirdNET-Analyzer) natively inside Windows Subsystem for Linux (WSL2) with GPU acceleration.

---

## Technical Architecture & Workflow Overview

Due to the ultrasonic physical characteristics of bat echolocation signals, raw recordings undergo a multi-stage process before inference:

+------------------------+      +---------------------------+      +-------------------------------+      +-------------------------+
| 1. Ingestion           |      | 2. Resampling Layer       |      | 3. GPU/CPU Inference          |      | 4. Data Aggregation     |
| Raw 192 kHz .wav files | ---> | Upscale to 256 kHz via    | ---> | BattyBirdNET-USA-WEST         | ---> | Structured CSV reports  |
| (AudioMoth v1.2/v2.0)  |      | librosa (24-bit PCM)      |      | TFLite Model Execution        |      | for automated extraction|
+------------------------+      +---------------------------+      +-------------------------------+      +-------------------------+**Ingestion & Dynamic Range:** AudioMoth hardware records raw, uncompressed ultrasonic audio at 192 kHz.
```

1. **Resampling Layer (192 kHz → 256 kHz):** Signals are programmatically resampled to **256 kHz** using `librosa` Kaiser-windowed sinc interpolation. This maps high-frequency bat echolocation into the model's feature space while maintaining 24-bit PCM dynamic range (`subtype="PCM_24"`). *(Downsampling to 48 kHz applies anti-aliasing filters that destroy ultrasonic bat signals).*

2. **Model Inference:** The resampled dataset is evaluated against the fine-tuned `BattyBirdNET-USA-WEST-256kHz.tflite` model.

3. **Data Aggregation & Export:** CSV detection reports are exported to local storage for downstream call audio extraction.

## Step 1: WSL2 System & GPU Pre-Flight Configuration

### 1. Configure WSL2 Memory Limits (`.wslconfig`)

To prevent Out-Of-Memory (OOM) kernel panics or WSL crashes during batch processing, limit WSL2 memory allocation on 16 GB host RAM systems to **12 GB** (leaving 4 GB dedicated to the Windows host).

In Windows, open or create `%USERPROFILE%\.wslconfig` and add:

ini

```
[wsl2]
guiApplications=true
memory=12GB
swap=4GB
```

> **Note:** Do not include `gpuSupport=true` in `.wslconfig`. GPU pass-through is managed automatically by default in WSL2.

Restart WSL from Windows PowerShell:

cmd

```
wsl --shutdown
```

### 

### 2. Bare-Metal CUDA Setup & System Dependencies

Run the following in your WSL2 Ubuntu terminal to set up NVIDIA CUDA 12.4 and audio dependencies:

Bash

```
sudo apt update && sudo apt upgrade -y

# Install standard Linux audio codecs and environment tools
sudo apt install python3-pip python3-venv libsndfile1 plocate ffmpeg -y

# Fetch WSL-Ubuntu CUDA repository pinning profiles
wget [https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin](https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin)
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Add official repository encryption keys
sudo apt-key adv --fetch-keys [https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/3bf863cc.pub](https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/3bf863cc.pub)
sudo add-apt-repository "deb [https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/](https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/) /"
sudo apt-get update

# Install CUDA Toolkit 12.4
sudo apt-get -y install cuda-toolkit-12-4

# Install legacy terminal info dependencies required by underlying runtime layers
wget [http://security.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb](http://security.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb)
sudo apt install ./libtinfo5_6.3-2ubuntu0.1_amd64.deb
```

## Step 2: Virtual Environment Setup & GPU Verification

Clone the repository and build the isolated `batenv` environment:

Bash

```
# Clone the repository framework
git clone [https://github.com/alarys/bat-detection-pipeline.git](https://github.com/alarys/bat-detection-pipeline.git)
cd bat-detection-pipeline

# Initialize virtual environment
python3 -m venv batenv
source batenv/bin/activate

# Upgrade package installer and install requirements
pip install --upgrade pip
pip install -r requirements.txt
pip install librosa soundfile pandas scipy guano mutagen

# Verify python modules are loaded
python3 -c "import guano, mutagen, soundfile, librosa, pandas, numpy; print('All dependencies verified successfully.')"

# Install TensorFlow with CUDA support
pip install "tensorflow[and-cuda]"
```

### Verify GPU Detection

Verify that your GPU device is detected inside `batenv`:

Bash

```
python3 -c "import tensorflow as tf; print('GPUs Available: ', tf.config.list_physical_devices('GPU'))"
```

**Expected Output:**

Plaintext

```
GPUs Available:  [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## Step 3: Pipeline Modules & Scripts

The repository contains four specialized execution scripts located in the project root:

1. **`resample.py`**: Validates WAV headers and resamples 192 kHz AudioMoth recordings to 256 kHz using `librosa` Kaiser-windowed sinc interpolation while maintaining native 24-bit PCM depth.

2. **`analyze.py`**: Executes TFLite model classification, performs mount/writeability checks, and outputs 1:1 audited CSV detection reports.

3. **`extract_snippets.py`**: Parses CSV detection reports, performs sliding-window aggregation with $\pm 5.0$s padding, and embeds dual-layer GUANO and RIFF INFO metadata headers directly into extracted 24-bit WAV snippets.

4. **`test_regression.py`**: Automated pre-flight regression test suite to validate storage mounts, file expansion ratios, upsampling integrity, and species detection ground truth.

## Step 4: Execution Workflow

### 1. Run Pre-Flight Regression Check

Validate storage mounts, resampling expansion ratios, and model accuracy against ground-truth recordings:

Bash

```
python3 test_regression.py
```

### 2. Execute Batch Audio Resampling (192 kHz → 256 kHz)

Resample raw AudioMoth files up to 256 kHz:

Bash

```
python3 resample.py
```

### 3. Run AI Classification

Execute inference using thread limits tuned for system stability:

#### Option 1: Safe Sequential Execution (Recommended)

Uses 1 worker thread to guarantee zero memory exhaustion or WSL crashes on large batch runs (e.g., >5,000 files):

Bash

```
python3 analyze.py \
  --i /mnt/c/AudioMoth/Ready_256kHz \
  --o /mnt/c/AudioMoth/Results \
  --classifier checkpoints/bats/v1.0/BattyBirdNET-USA-WEST-256kHz.tflite \
  --min_conf 0.99 \
  --rtype csv \
  --threads 1 \
  --batchsize 4
```

#### Option 2: Moderate Parallel Execution

Uses 2 worker threads for faster execution while remaining within a 12 GB RAM footprint:

Bash

```
python3 analyze.py \
  --i /mnt/c/AudioMoth/Ready_256kHz \
  --o /mnt/c/AudioMoth/Results \
  --classifier checkpoints/bats/v1.0/BattyBirdNET-USA-WEST-256kHz.tflite \
  --min_conf 0.99 \
  --rtype csv \
  --threads 2 \
  --batchsize 4
```

### 4. Extract Call Audio Snippets & Inject Metadata

Parse CSV detection reports, aggregate adjacent call intervals, write 24-bit snippets, and embed GUANO/RIFF metadata:

Bash

```
python3 extract_snippets.py
```

### 5. Audit Detections via Shell

Summarize species identification counts across all generated CSV reports:

Bash

```
awk -F, 'FNR > 1 {print $5}' /mnt/c/AudioMoth/Results/*.csv | sort | uniq -c | sort -nr
```

## Output CSV Schema & Field Definitions

Each output `.csv` file contains detection rows matching the following structure:

Code snippet

```
Filename,Start (sec),End (sec),Scientific name,Common name,Confidence
20260615_213000.wav,3.0000,6.0000,Myotis lucifugus,Little brown bat,0.9942105
```

| **Column Name**     | **Data Type** | **Description**                                                                            |
| ------------------- | ------------- | ------------------------------------------------------------------------------------------ |
| **Filename**        | `string`      | The source `.wav` recording filename associated with the detection.                        |
| **Start (sec)**     | `float`       | The precise start offset (in seconds) within the audio file where the call was detected.   |
| **End (sec)**       | `float`       | The precise end offset (in seconds) within the audio file where the detection window ends. |
| **Scientific name** | `string`      | The taxonomic binomial nomenclature for the detected species (e.g., *Myotis lucifugus*).   |
| **Common name**     | `string`      | The localized vernacular name of the species (e.g., *Little brown bat*).                   |
| **Confidence**      | `float`       | Model identification confidence score bounded between `0.0000000` and `1.0000000`.         |

## Appendix: Technical Physics, Metadata Specifications, and Ground Truth Baseline

### A1. AudioMoth Hardware Specifications & Signal Dynamics

AudioMoth hardware (v1.2.0 / v2.0) records uncompressed 192 kHz, 24-bit PCM WAV files, capable of capturing acoustic frequencies up to the Nyquist limit of 96 kHz. Because North American insectivorous bat echolocation calls typically range from 20 kHz up to 120 kHz, preserving high-frequency resolution and dynamic range is vital.

- **Bit Depth & Dynamic Range:** AudioMoth captures quiet, distant bat echolocation calls using 24-bit quantization depth. Retaining 24-bit PCM (`PCM_24`) provides a theoretical dynamic range of 144 dB, preserving faint harmonic calls near the noise floor.

- **Upsampling Dynamics (192 kHz → 256 kHz):** Machine learning classifiers like `BattyBirdNET-USA-WEST-256kHz.tflite` require input vectors centered at 256 kHz. Upsampling via `librosa` using Kaiser-windowed sinc interpolation (`resample_type='kaiser_best'`) interpolates time-domain samples without introducing spectral aliasing or modifying physical pitch.

### A2. Snippet Extraction & Metadata Architecture

`extract_snippets.py` uses an advanced extraction workflow engineered for bioacoustic analysis:

- **Native Fidelity Preservation:** Reads raw audio into integer representations (`dtype="int32"`) and writes output clips as 24-bit PCM (`PCM_24`), bypassing floating-point quantization roundoff.

- **Sliding Window Aggregation:** Merges nearby detections separated by less than 10 seconds ($\text{PADDING\_SEC} \times 2$) into a unified window. Windows are capped at 20 seconds of raw detection span (`SLIDING_WINDOW_BOUND = 20.0`), producing final clips under 30 seconds.

- **$\pm 5.0$-Second Temporal Padding:** Adds 5.0 seconds of background context before the start and after the end of each consolidated detection window.

- **Dual-Layer Metadata Tagging:**

- **GUANO Metadata (`guano.GuanoFile`):** Writes standard fields including `GUANO|Version` (`1.0`), `Species Auto` (alphabetical common names), `Species Manual` (alphabetical scientific names), and custom `BattyBirdNET|Confidence` metrics.

- **RIFF INFO Headers (`mutagen.wave.WAVE`):** Injects standard `INAM` (Title/Species), `IART` (Artist / "AudioMoth Bat Detector"), and `ICMT` (Comments / Max Confidence) tags to ensure full compatibility with software like Kaleidoscope, SonoBat, and standard media tools.

### A3. Xeno-Canto Ground Truth Recording (`XC883525`)

Regression testing relies on a verified, high-quality reference recording of a Little Brown Bat (*Myotis lucifugus*) sourced from Xeno-Canto:

- **Reference File:** `XC883525 - Little Brown Myotis - Myotis lucifugus.wav`

- **Baseline Detection Target:** `XC883525.BirdNET.results-brown-bat-good.csv`

- **Validation Criteria:**
1. **NFS Mount Check:** Verifies that `/mnt/tweety/verified/` is mounted and readable.

2. **File Expansion Ratio:** Asserts that resampling yields a file size expansion ratio $\ge 1.25\times$ (verifying that no bit-depth truncation or downsampling occurred).

3. **Classification Integrity:** Asserts that `analyze.py` identifies **Little brown bat** with $\ge 0.99$ confidence.

### A4. Negative Example: Anti-Aliasing & Downsampling Failure Modes

Plaintext

```
WRONG PIPELINE:
Raw Audio (192 kHz) ──> ffmpeg (Default Downsample to 48 kHz) ──> Truncated 16-bit PCM ──> Ultrasonic Signals Wiped Out!

CORRECT PIPELINE:
Raw Audio (192 kHz) ──> librosa Sinc Interpolation (Upsample to 256 kHz) ──> soundfile (PCM_24) ──> Ultrasonic Spectrum Preserved!
```

- **The Failure:** Downsampling 192 kHz ultrasonic audio to 48 kHz using standard utilities (`ffmpeg -ar 48000`) imposes a Nyquist cutoff frequency at 24 kHz. Internal low-pass anti-aliasing filters strip out all acoustic data above 24 kHz.

- **The Result:** Because bat echolocation calls typically span 30 kHz to 120 kHz, downsampled waveforms contain zero bat energy, resulting in 0% detection rates across the entire dataset.

---

### Appendix 2: Repository Deployment & Upstream Integration Workflow

To maintain a clean, lightweight footprint containing only custom bioacoustic pipeline scripts and the North American West bat model—while avoiding heavy upstream dependencies—follow this clean-room deployment workflow:

#### Step 1: Clone Your Clean Custom Repository

Clone your dedicated pipeline repository into your local environment:

```bash
git clone https://github.com/alarys/bat-detection-pipeline.git
cd bat-detection-pipeline
```

#### Step 2: Clone the Upstream Base Repository (Temporary)

To acquire the base framework, utilities, and models required to execute the pipeline, clone the upstream `BattyBirdNET-Analyzer` project into a temporary side directory:

```bash
git clone https://github.com/rdz-oss/BattyBirdNET-Analyzer.git ../BattyBirdNET-upstream
```

#### Step 3: Populate Required Paths and Structures

Copy the required foundational modules, directories, and the target North American West bat model checkpoint from the upstream directory into your active repository structure:

```bash
# Create necessary nested directory structures
mkdir -p checkpoints/bats/v1.0

# Copy core runtime modules and assets
cp ../BattyBirdNET-upstream/audio.py .
cp ../BattyBirdNET-upstream/model.py .
cp ../BattyBirdNET-upstream/utils.py .
cp ../BattyBirdNET-upstream/client.py .
cp ../BattyBirdNET-upstream/server.py .
cp ../BattyBirdNET-upstream/config.py .
cp ../BattyBirdNET-upstream/species.py .

# Copy necessary support folders
cp -r ../BattyBirdNET-upstream/labels/ .

# Copy the specific North American West bat model checkpoint
cp ../BattyBirdNET-upstream/checkpoints/bats/v1.0/BattyBirdNET-USA-WEST-256kHz.tflite checkpoints/bats/v1.0/
```

#### Step 4: Cleanup Temporary Upstream Files

Remove the temporary upstream clone to keep your workspace pristine:

```bash
rm -rf ../BattyBirdNET-upstream
```

---

### File Name Summary & Functions

| **File Name**                                           | **Primary Function**                                                                                                                                                                                                                                      | **Resume & Idempotency Mechanism**                                                                                         |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **`resample_multi_input.py`**<br><br><br><br><br><br>   | Ingests multi-rate raw ultrasonic files (192kHz or 250kHz), validates headers/durations, resamples them uniformly to 256kHz via `librosa`, and preserves 1-to-1 GUANO metadata. Includes a user-prompted chronological start filename filter.<br><br><br> | Checks if the target output `.wav` already exists in `/mnt/c/AudioMoth/Ready_256kHz`.<br><br><br>                          |
| **`inspect_wav.py`**<br><br><br><br><br><br>            | Legacy single-rate (192kHz) pre-flight header validation script used for auditing and filtering raw input directories before batch execution.<br><br><br>                                                                                                 | Destination file existence check combined with `resample_errors.log` tracking.<br><br><br>                                 |
| **`analyze.py`**<br><br><br><br><br><br>                | Core batch analyzer running the BattyBirdNET-256kHz TensorFlow Lite model across audio files, generating detailed detection CSV reports containing timestamps, confidence scores, and species names.<br><br><br>                                          | Checks for the existence of the expected target CSV report file in the output directory (`analyze_skips.log`).<br><br><br> |
| **`extract_snippets_final.py`**<br><br><br><br><br><br> | Parses analysis CSV reports, aggregates contiguous detection spans using sliding windows (+/- 5s padding), slices the raw audio directly into 24-bit PCM snippets, and inherits 1-to-1 GUANO hardware parameters and ID3 tags.<br><br><br>                | Maintains a persistent `.processed_manifest.log` file in the archive directory to track completed CSVs.<br><br><br>        |

 The scripts:

- **`resample_multi_input.py`** (your multi-rate ultrasonic resampler with the new date range prompt)

- **`extract_snippets_final.py`** (your snippet extractor with 1-to-1 GUANO inheritance)

- **`analyze.py`** (your core batch analyzer)

- **`inspect_wav.py`** (your single-rate validation audit tool)
