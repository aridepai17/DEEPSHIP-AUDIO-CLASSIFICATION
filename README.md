# DeepShip Underwater Acoustic Classification with DWSTr (Kaggle Edition)

### Hybrid Depthwise-Separable-Convolution + Transformer network for classifying ship-radiated noise, trained end-to-end in Google Colab from a Kaggle-hosted copy of the DeepShip dataset

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Architecture Overview](#architecture-overview)
- [Detailed Notebook Walkthrough](#detailed-notebook-walkthrough)
  - [Cell 1: Setup and Environment Configuration](#cell-1-setup-and-environment-configuration)
  - [Cell 2: DWSTr Preprocessor Class](#cell-2-dwstr-preprocessor-class)
  - [Cell 3: Directory-Based Data Loading for DeepShip](#cell-3-directory-based-data-loading-for-deepship)
  - [Cell 4: Kaggle API Download and Preprocessing (All-in-One)](#cell-4-kaggle-api-download-and-preprocessing-all-in-one)
  - [Cell 4.5: Mel-Spectrogram Matrix Visualization](#cell-45-mel-spectrogram-matrix-visualization)
  - [Cell 5: Data Splitting and Final Save](#cell-5-data-splitting-and-final-save)
  - [Cell 6: Load Final Data and Sanity Check](#cell-6-load-final-data-and-sanity-check)
  - [Cell 7: Import TensorFlow/Keras and Set Random Seeds](#cell-7-import-tensorflowkeras-and-set-random-seeds)
  - [Cell 8: Depthwise Separable Convolution (DWS) Block](#cell-8-depthwise-separable-convolution-dws-block)
  - [Cell 9: Transformer Encoder Block](#cell-9-transformer-encoder-block)
  - [Cell 10: Complete DWSTr Architecture](#cell-10-complete-dwstr-architecture)
  - [Cell 11: Build and Compile Model](#cell-11-build-and-compile-model)
  - [Cell 12: Setup Training Callbacks](#cell-12-setup-training-callbacks)
  - [Cell 13: Compute Smoothed Class Weights and Train](#cell-13-compute-smoothed-class-weights-and-train)
  - [Cell 14: Evaluate on Test Set](#cell-14-evaluate-on-test-set)
  - [Cell 15: Plot Training History](#cell-15-plot-training-history)
  - [Cell 16: Confusion Matrix and Per-Class Performance](#cell-16-confusion-matrix-and-per-class-performance)
- [Results and Performance](#results-and-performance)
- [Best Practices and Design Decisions](#best-practices-and-design-decisions)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

This project is a complete, end-to-end audio classification pipeline that identifies ship types from their underwater radiated-noise signatures. Everything - dataset download, signal preprocessing, model definition, training, and evaluation - lives in a single Colab notebook, **`DWSTr_Deepship_Kaggle.ipynb`**, and is designed to run top-to-bottom on a GPU-accelerated Colab runtime (the reference run used a T4).

Unlike a version of this pipeline that expects a dataset folder to already exist on Google Drive, this notebook is the **"Kaggle edition"**: it pulls the [DeepShip dataset](#dataset) directly from Kaggle using the Kaggle API, so the only manual step is supplying Kaggle API credentials. Everything downstream - segmenting audio, extracting Mel-spectrograms, splitting the data, training, and evaluating - is fully automated.

The model at the center of the notebook is **DWSTr**, a hybrid architecture that combines:

- **Depthwise Separable Convolution (DWS)** - a lightweight convolutional front end that extracts local time-frequency patterns from each Mel-spectrogram with far fewer parameters than a standard convolution.
- **A Vision-Transformer-style encoder** - the convolutional feature map is cut into patches, prepended with a learnable classification token (as in ViT/BERT), given learned positional embeddings, and passed through 6 stacked self-attention encoder blocks that model relationships between different time-frequency regions of the spectrogram.

Trained on the 4-class DeepShip dataset (**Cargo, Passengership, Tanker, Tug**), the notebook's own run reached **97.80% accuracy** on a held-out test set of over 15,000 spectrogram segments, with every class scoring above 96% per-class accuracy despite a roughly 4:1 imbalance between the largest and smallest class.

---

## Key Features

- ✅ **Zero-setup dataset acquisition** - downloads and unzips the DeepShip dataset straight from Kaggle inside the notebook; no manual upload of audio files required.
- ✅ **Automatic class discovery** - class names and labels are inferred directly from the dataset's folder structure, so the pipeline adapts to however many ship-type folders are present.
- ✅ **75ms segmentation strategy** - each raw recording is chopped into thousands of short, non-overlapping Mel-spectrogram "frames," turning a handful of audio files into a 76,000+ sample training set.
- ✅ **Hybrid CNN + Transformer architecture** - a depthwise separable convolution block for cheap local feature extraction, feeding a 6-block Transformer encoder for global context, with only ~1.33M parameters total.
- ✅ **Imbalance-aware training** - square-root-smoothed, balanced class weights keep the rarest class (Tug) from being drowned out by the most common one (Cargo) without over-correcting.
- ✅ **Production-style training loop** - model checkpointing, early stopping, and learning-rate-on-plateau reduction are all wired up before a single epoch runs.
- ✅ **Research-grade visualizations** - a per-class Mel-spectrogram gallery before training, and accuracy/loss curves plus a confusion matrix after it, all saved as high-resolution PNGs.
- ✅ **Reproducible evaluation** - a stratified 70/10/20 train/val/test split (with a fixed random seed) and a completely held-out test set that the model never sees until the final evaluation cell.

---

## Dependencies

Most of these ship pre-installed on a standard Google Colab runtime; the notebook's first cell explicitly installs the handful that don't.

| Library | Purpose | Installed by notebook? |
|---|---|---|
| `librosa` | Audio loading, resampling, pre-emphasis, and Mel-spectrogram extraction | ✅ `pip install` in Cell 1 |
| `soundfile` | Backend audio codec support used internally by `librosa` | ✅ `pip install` in Cell 1 |
| `pandas` | Lightweight data handling (imported for convenience; not heavily used in this notebook) | ✅ `pip install` in Cell 1 |
| `scikit-learn` | Stratified train/val/test splitting, balanced class weights, classification report, confusion matrix | ✅ `pip install` in Cell 1 |
| `tqdm` | Progress bars during audio preprocessing | ✅ `pip install` in Cell 1 |
| `numpy` | Array manipulation and storage for all feature/label tensors | ⚠️ Assumed pre-installed (Colab default) |
| `tensorflow` / `keras` | Model definition, training, evaluation | ⚠️ Assumed pre-installed (Colab default) |
| `matplotlib` | Spectrogram gallery, training curves | ⚠️ Assumed pre-installed (Colab default) |
| `seaborn` | Confusion matrix heatmap | ⚠️ Assumed pre-installed (Colab default) |
| `kaggle` | Command-line client used to download the dataset | ⚠️ Assumed pre-installed (Colab default) |

> **Note:** Cell 1's `pip install` line only lists `librosa soundfile pandas tqdm scikit-learn`. TensorFlow, Matplotlib, Seaborn, and the `kaggle` CLI are never explicitly installed anywhere in the notebook - it relies on them already being present in Colab's base image. If you run this notebook outside Colab (e.g., a local Jupyter server or a bare Docker container), you'll need to `pip install tensorflow matplotlib seaborn kaggle` yourself before Cells 4, 7, and 15/16 will work.

---

## Dataset

The notebook downloads the **DeepShip** dataset from Kaggle (`vasundharauppuluri/deep-ship`, a mirror of the original DeepShip release, listed under an MIT license). DeepShip is a benchmark of real-world, passive-sonar recordings of ship-radiated noise, originally introduced by Irfan et al. (see [Citation](#citation)). As downloaded and unzipped by this notebook, it is organized as:

```
DeepShip-main/
├── Cargo/
│   ├── <ship_1>/*.wav
│   ├── <ship_2>/*.wav
│   └── ...
├── Passengership/
│   └── ...
├── Tanker/
│   └── ...
└── Tug/
    └── ...
```

Each top-level folder name is treated as a class label, and every `.wav` file nested anywhere beneath it - regardless of how many ship-name subfolders deep - belongs to that class.

**In the notebook's own run:**

| Class | Test-set segments | Share of test set |
|---|---|---|
| Cargo | 6,199 | 40.62% |
| Tanker | 4,426 | 29.00% |
| Passengership | 3,056 | 20.02% |
| Tug | 1,581 | 10.36% |
| **Total** | **15,262** | **100%** |

Because the train/val/test split is *stratified* (Cell 5), these percentages are representative of the full, ~76,300-segment dataset, and scaling them up gives an approximate full-dataset picture:

| Class | Est. total segments | Est. train (70%) | Est. val (10%) | Est. test (20%) |
|---|---|---|---|---|
| Cargo | ≈30,995 | ≈21,696 | ≈3,100 | 6,199 |
| Tanker | ≈22,130 | ≈15,491 | ≈2,213 | 4,426 |
| Passengership | ≈15,280 | ≈10,696 | ≈1,528 | 3,056 |
| Tug | ≈7,905 | ≈5,534 | ≈790 | 1,581 |

*(The "Est." columns are back-calculated from the exact test-set support numbers in Cell 16's classification report, scaled by the known 70/10/20 split ratio - the notebook itself never prints full-dataset per-class counts directly.)*

The roughly **4:1 imbalance** between Cargo (the largest class) and Tug (the smallest) is exactly why Cell 13 computes balanced, square-root-smoothed class weights before training - see that section for the mechanics.

The dataset itself is derived from only **63 raw `.wav` recordings**, which the preprocessing pipeline (Cells 2–4) expands into **76,311 short spectrogram segments** - an average of about **1,211 segments (≈91 seconds of audio) per file**.

---

## Architecture Overview

The DWSTr model follows this data flow, from a raw 128×4 Mel-spectrogram "image" all the way to a 4-way class prediction:

```
Input Mel-Spectrogram (128 × 4 × 1)
              ↓
[Depthwise Separable Conv Block]   ← Cell 8: cheap local feature extraction (1 → 64 channels)
              ↓                       Output: (128 × 4 × 64)
[Reshape into 32 patches]          ← Cell 10: (128 × 4 × 64) → (32 patches × 1024 values)
              ↓
[Dense Patch Projection]           ← Cell 10: 1024 → 64-dim patch embeddings
              ↓
[Prepend Class Token]              ← Cell 10: 32 patches → 33 tokens (1 [CLS] + 32 patches)
              ↓
[+ Learned Positional Embedding]   ← Cell 10: injects position information (attention has none by default)
              ↓
[Dropout]
              ↓
[Transformer Encoder × 6]          ← Cell 9 (block) + Cell 10 (stacking): global self-attention over all 33 tokens
              ↓
[Extract Class-Token Output]       ← Cell 10: take token index 0 (the [CLS] token) as the pooled representation
              ↓
[LayerNorm → Dropout → Dense(1024, GELU) → Dropout]   ← Cell 10: classification head
              ↓
[Dense(4, Softmax)]                ← Cell 10: final class probabilities
```

**Key design principles:**

1. **Efficiency first.** The depthwise separable block keeps the convolutional front end to under 400 parameters, so almost the entire ~1.33M-parameter budget goes to the Transformer, where it's most useful (see the [full parameter breakdown](#cell-11-build-and-compile-model)).
2. **Convolution for local structure, attention for global structure.** The DWS block is good at picking up short-range time-frequency texture (e.g., harmonic bands); the Transformer's self-attention can then relate any patch to any other patch, regardless of distance, which plain convolution struggles to do without many stacked layers.
3. **ViT-style classification, not just pooling.** Rather than average-pooling the patch embeddings, the model uses a dedicated, learnable class token (exactly as in BERT/ViT) whose final-layer representation is trained to summarize the whole sequence for classification.
4. **Heavy regularization for a small model.** With `dropout_rate=0.3` applied in the attention, the MLP blocks, and the classification head, plus batch normalization in the DWS block, the architecture leans hard on regularization rather than sheer scale to avoid overfitting on a dataset built from only 63 source recordings.
5. **Imbalance handled at the loss level, not the architecture level.** There is no architectural trick for class imbalance here (e.g., no focal loss) - it's handled entirely through the sample weighting in Cell 13.

---

## Detailed Notebook Walkthrough

This section goes through **every cell in `DWSTr_Deepship_Kaggle.ipynb`**, in order, explaining what it does, why it's written the way it is, and what actually happened when it was run (using the real printed output and figures captured in the notebook). Cell numbers below match the `# CELL N:` banner comments the notebook itself uses, so you can jump straight to the matching cell when reading side-by-side.

---

### Cell 1: Setup and Environment Configuration

#### What It Does

This is the notebook's bootstrap cell. It mounts Google Drive, installs a handful of Python packages, imports everything the rest of the notebook needs, and defines the two paths the whole pipeline is built around.

```python
from google.colab import drive
drive.mount('/content/drive')

print("Installing required packages...")
!pip install librosa soundfile pandas tqdm scikit-learn -q

import numpy as np
import pandas as pd
import librosa
import os
import glob
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings('ignore')  # Ignore librosa warnings

DATASET_ROOT_PATH = '/content/drive/MyDrive/DeepShip-main'
SAVE_DIR = '/content/drive/MyDrive/processed_data_DeepShip_DWSTr'

os.makedirs(SAVE_DIR, exist_ok=True)
```

**Captured output:**
```
Mounted at /content/drive
Installing required packages...

Environment setup complete. Paths configured:
  Dataset Root: /content/drive/MyDrive/DeepShip-main
  Save Directory: /content/drive/MyDrive/processed_data_DeepShip_DWSTr
```

#### Why It's Written This Way

- **Drive mount = persistence.** Colab's local disk is wiped every time the runtime disconnects or recycles. Mounting Drive means the processed dataset, model checkpoints, and result plots all survive session restarts - you don't have to reprocess 63 audio files from scratch every time you reopen the notebook.
- **Two independent path variables.** `DATASET_ROOT_PATH` (where raw audio lives) and `SAVE_DIR` (where every processed artifact - `.npz` feature files, `class_names.pkl`, model checkpoints, and plot images - gets written) are kept separate on purpose, so the raw dataset folder is never accidentally overwritten by pipeline outputs.
- **`warnings.filterwarnings('ignore')`** silences the (harmless but noisy) deprecation and format warnings `librosa` tends to emit when loading many files in a loop, so the real progress output isn't buried.

#### Technical Considerations

- **This cell's `DATASET_ROOT_PATH` is provisional.** It assumes you already have an extracted `DeepShip-main` folder sitting on your Drive. In practice, this notebook doesn't use that assumption - **Cell 4 overwrites `DATASET_ROOT_PATH` entirely** once it downloads and extracts the dataset from Kaggle. If you skip Cell 4 and try to run Cell 3's loader against the Drive path directly, make sure a real dataset actually exists there first.
- **`tqdm.notebook`** is used instead of plain `tqdm` specifically because it renders an interactive HTML progress widget in Colab/Jupyter rather than a plain-text bar - this is what produces the widget output you see in Cell 4.
- **First-run friction:** the Drive mount triggers a Google authentication popup the first time it runs in a session, and can silently time out after long periods of inactivity - if a later cell throws a `FileNotFoundError` on a Drive path, re-running this cell is the first thing to try.

---

### Cell 2: DWSTr Preprocessor Class

#### What It Does

This cell defines `DWSTrPreprocessor`, the class responsible for turning one raw `.wav` file into a stack of fixed-size Mel-spectrogram tensors. Every later data-loading function (Cells 3 and 4) is built on top of it.

```python
class DWSTrPreprocessor:
    def __init__(self):
        self.sr = 22050
        self.segment_duration = 0.075
        self.segment_samples = int(self.sr * self.segment_duration)
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128
        self.fmax = self.sr // 2
```

**Configuration values and why they were chosen:**

| Parameter | Value | Rationale |
|---|---|---|
| `sr` (sample rate) | 22,050 Hz | Half of "CD quality" (44.1 kHz); halves the data volume and compute cost while still covering the frequency range where most ship-noise energy and harmonics live. |
| `segment_duration` | 0.075 s (75 ms) | Short enough to produce thousands of training examples from a handful of recordings; long enough to still contain a few periods of low-frequency engine/propeller harmonics. |
| `n_fft` | 2048 samples | FFT window size - at 22,050 Hz this is a ~93 ms analysis window, giving good frequency resolution. |
| `hop_length` | 512 samples | The stride between successive FFT windows (75% overlap between windows); controls the number of time frames in the output. |
| `n_mels` | 128 | Number of Mel-scale frequency bins - a common middle ground between coarse (e.g., 40) and very fine-grained (e.g., 256) spectral resolution. |
| `fmax` | `sr // 2` = 11,025 Hz | The Nyquist frequency for a 22,050 Hz signal - the theoretical maximum frequency that can be represented, so this covers the full audible spectrum captured by the sample rate. |

> **Precision note:** `segment_samples = int(22050 * 0.075)` evaluates to **1,653 samples**, not 1,654 - Python's `int()` truncates rather than rounds, and `22050 * 0.075 = 1653.75`. It's a one-sample difference (≈0.045 ms) that has no practical effect on the output, but worth knowing if you're trying to reconcile the numbers exactly.

**Method-by-method breakdown:**

**1. `preemphasis(audio, coef=0.97)`**

```python
def preemphasis(self, audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
    return np.append(audio[0], audio[1:] - coef * audio[:-1])
```

Implements the classic pre-emphasis filter `y[n] = x[n] - 0.97·x[n-1]`, a first-order high-pass filter. It's standard practice in speech/audio ML pipelines because raw audio tends to have more energy at low frequencies; boosting the high end beforehand makes higher-frequency features (which, for ships, can include propeller cavitation and higher-order engine harmonics) more prominent relative to the low-frequency rumble that would otherwise dominate the spectrogram.

**2. `segment_audio(audio)`**

```python
def segment_audio(self, audio: np.ndarray) -> list[np.ndarray]:
    segments = []
    for start in range(0, len(audio) - self.segment_samples + 1, self.segment_samples):
        segment = audio[start:start + self.segment_samples]
        if len(segment) == self.segment_samples:
            segments.append(segment)
    return segments
```

Chops the full waveform into **non-overlapping** 1,653-sample windows. Because the step size in the loop equals `segment_samples`, consecutive segments never overlap and any leftover audio shorter than one full segment at the end of the file is simply dropped. This is also what turns a small number of source recordings into a large training set: a single ~90-second recording yields roughly 1,200 independent training examples.

**3. `extract_mel_spectrogram(segment)`**

```python
def extract_mel_spectrogram(self, segment: np.ndarray) -> np.ndarray:
    emphasized = self.preemphasis(segment)
    mel_spec = librosa.feature.melspectrogram(
        y=emphasized, sr=self.sr, n_fft=self.n_fft,
        hop_length=self.hop_length, n_mels=self.n_mels,
        power=2.0, window='hann'
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    expected_frames = 4
    if mel_spec_db.shape[1] != expected_frames:
        if mel_spec_db.shape[1] < expected_frames:
            pad_width = expected_frames - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='edge')
        else:
            mel_spec_db = mel_spec_db[:, :expected_frames]
    return mel_spec_db
```

For each 1,653-sample segment: apply pre-emphasis, compute a power-spectrum Mel-spectrogram with a Hann window, then convert to a decibel (log) scale with `librosa.power_to_db`, which rescales so the loudest point in the segment sits at 0 dB and everything else is expressed relative to it (`ref=np.max`). This log compression matters because raw power spectrograms are extremely skewed - a handful of loud frequency bins would otherwise dominate any model trained on the raw values.

With these exact settings, librosa's frame-count formula (`1 + segment_samples // hop_length` under the library's default `center=True` padding) works out to `1 + 1653 // 512 = 4` - so the output is naturally `(128, 4)` and the pad/trim block is purely **defensive**: it guards against a segment ever coming out a frame short or long (for instance from a slightly unusual file), rather than something that fires in normal operation. `mode='edge'` padding (repeating the last column) is a sensible choice here since it avoids introducing artificial silence or discontinuities at a spectrogram's boundary the way zero-padding would.

**4. `process_file(audio_path)`**

```python
def process_file(self, audio_path: str) -> np.ndarray:
    audio, _ = librosa.load(audio_path, sr=self.sr, mono=True)
    segments = self.segment_audio(audio)
    mel_specs = [self.extract_mel_spectrogram(seg) for seg in segments]
    return np.array(mel_specs)
```

The orchestrator method: load the file (resampling to 22,050 Hz and downmixing to mono regardless of the source file's original format), segment it, and convert every segment to a Mel-spectrogram, returning an array of shape `(num_segments, 128, 4)` for that one file.

#### Technical Considerations

- **`list[np.ndarray]` type hints** (used without importing `List` from `typing`) require Python 3.9+, which Colab's default runtime satisfies - but this would break on an older interpreter.
- **A single `DWSTrPreprocessor` instance is meant to be reused** across an entire dataset loading pass (as Cells 3 and 4 do) rather than re-instantiated per file - construction is cheap either way, but reuse keeps the code simpler.
- **This class does no caching.** Every call to `process_file` re-reads and re-decodes audio from disk; if you need to reprocess the same files repeatedly during development, consider caching intermediate results yourself.

---

### Cell 3: Directory-Based Data Loading for DeepShip

#### What It Does

This cell defines two functions that turn a folder of `.wav` files into labeled training arrays, using the **folder structure itself** as the source of truth for class labels (unlike a metadata-CSV-driven approach, which some other DWSTr-style notebooks use). Nothing in this cell actually *runs* yet - it's pure function definition, executed later in Cell 4.

**1. `get_deepship_classes(dataset_path)`**

```python
def get_deepship_classes(dataset_path: str) -> tuple:
    class_names = sorted([d for d in os.listdir(dataset_path)
                          if os.path.isdir(os.path.join(dataset_path, d))])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    print(f"DeepShip classes mapped. Found {len(class_names)} unique classes: {class_names}")
    return class_names, class_to_idx
```

Lists every **immediate subdirectory** of `dataset_path` and treats each one as a class. Sorting alphabetically before assigning indices is what makes label assignment deterministic and reproducible - `Cargo` will always map to index 0, `Passengership` to 1, and so on, run after run, as long as the same four folders are present.

**2. `load_and_process_deepship_data(dataset_path)`**

```python
def load_and_process_deepship_data(dataset_path: str) -> tuple:
    class_names, class_to_idx = get_deepship_classes(dataset_path)
    preprocessor = DWSTrPreprocessor()
    X_list, y_list = [], []

    audio_files = glob.glob(os.path.join(dataset_path, '**', '*.wav'), recursive=True)

    for audio_file in tqdm(audio_files, desc="Preprocessing DeepShip Audio"):
        try:
            path_parts = os.path.normpath(audio_file).split(os.sep)
            class_name = None
            for cls in class_names:
                if cls in path_parts:
                    class_name = cls
                    break

            if class_name in class_to_idx:
                class_idx = class_to_idx[class_name]
                mel_specs = preprocessor.process_file(audio_file)
                if len(mel_specs) > 0:
                    X_list.extend(mel_specs)
                    y_list.extend([class_idx] * len(mel_specs))
        except Exception as e:
            continue  # Skip corrupted or unreadable files gracefully

    X = np.expand_dims(np.array(X_list), axis=-1)
    y = np.array(y_list)

    del X_list, y_list
    return X, y, class_names
```

Step by step:

1. **Recursive file discovery** - `glob.glob(..., '**', '*.wav', recursive=True)` finds every `.wav` file no matter how many ship-name subfolders deep it's nested under a class folder.
2. **Label inference from the path** - rather than assuming a fixed folder depth, it splits the full file path into its components and checks *which known class name appears as one of those components*. This is intentionally more robust than, say, always taking "two levels up" from the file - it works whether a file is nested one level or five levels below its class folder.
3. **One file → many labeled samples** - `preprocessor.process_file()` returns potentially hundreds of Mel-spectrograms per audio file; every one of them is appended to `X_list` with the *same* class label repeated via `[class_idx] * len(mel_specs)`.
4. **Silent per-file error handling** - a bare `try/except: continue` means a single corrupted, empty, or unreadable `.wav` file won't crash a multi-hour preprocessing run; it's just skipped.
5. **Channel dimension added at the very end** - `np.expand_dims(..., axis=-1)` turns each `(128, 4)` spectrogram into `(128, 4, 1)`, matching the shape Keras convolutional layers expect (height, width, channels).
6. **Explicit `del X_list, y_list`** - once the Python lists have been converted to compact NumPy arrays, the original lists (which store many separate small objects and are far less memory-efficient) are deleted to free RAM immediately, rather than waiting for them to fall out of scope.

#### Technical Considerations

- **The silent `except: continue`** is a double-edged sword: it makes the pipeline resilient to a handful of bad files, but if *most* of your files are being skipped for some reason (wrong extension, wrong permissions, a bug elsewhere), you'll only find out indirectly - from a much smaller-than-expected sample count - rather than from an error message. If a preprocessing run finishes suspiciously fast or with a suspiciously small segment count, it's worth temporarily removing this `try/except` to see the real traceback.
- **Class detection relies on exact path-component matches**, not substring matches (`cls in path_parts` checks list membership, not `cls in audio_file`), so a ship name that happens to contain a class name as a substring (e.g., a boat named `"CargoKing"`) won't be mis-detected - only an actual folder named exactly `Cargo`, `Tanker`, etc. counts.

---

### Cell 4: Kaggle API Download and Preprocessing (All-in-One)

#### What It Does

This is where the pipeline actually *runs* for the first time - it downloads the real dataset from Kaggle, extracts it, and executes the loading/preprocessing functions defined in Cells 2 and 3 against it.

```python
# 1. Setup Kaggle API securely
!mkdir -p ~/.kaggle
!cp /content/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# 2. Download the DeepShip dataset directly from Kaggle
!kaggle datasets download -d vasundharauppuluri/deep-ship -p /content

# 3. Unzip the dataset
ZIP_PATH = '/content/deep-ship.zip'
EXTRACT_DIR = '/content/DeepShip_Dataset'
if os.path.exists(ZIP_PATH):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
else:
    raise FileNotFoundError("Dataset zip file not found! Check if kaggle.json is uploaded.")

# 4. Determine correct path (handling the Kaggle 'DeepShip-main' nested folder)
if os.path.exists(os.path.join(EXTRACT_DIR, 'DeepShip-main')):
    DATASET_ROOT_PATH = os.path.join(EXTRACT_DIR, 'DeepShip-main')
elif os.path.exists(os.path.join(EXTRACT_DIR, 'DeepShip')):
    DATASET_ROOT_PATH = os.path.join(EXTRACT_DIR, 'DeepShip')
else:
    DATASET_ROOT_PATH = EXTRACT_DIR

# 5. Run the pipeline
X_all, y_all, class_names = load_and_process_deepship_data(DATASET_ROOT_PATH)

# 6. Save immediately
np.savez_compressed(os.path.join(SAVE_DIR, 'full_data_preprocessed.npz'), X=X_all, y=y_all)

# 7. Free RAM
del X_all, y_all

# 8. Persist class names for later cells/sessions
with open(os.path.join(SAVE_DIR, 'class_names.pkl'), 'wb') as f:
    pickle.dump(class_names, f)
```

**Captured output (abridged):**
```
Setting up Kaggle API...
Downloading DeepShip dataset...
Dataset URL: https://www.kaggle.com/datasets/vasundharauppuluri/deep-ship
License(s): MIT
Downloading deep-ship.zip to /content
100% 503M/503M [00:28<00:00, 18.3MB/s]

Extracting /content/deep-ship.zip to /content/DeepShip_Dataset...
Extraction complete!

Targeting dataset root: /content/DeepShip_Dataset/DeepShip-main
DeepShip classes mapped. Found 4 unique classes: ['Cargo', 'Passengership', 'Tanker', 'Tug']
DWSTr Preprocessor initialized. Output shape: (128x4)

Found 63 total audio files to process.
Preprocessing DeepShip Audio: 100%|██████████| 63/63 [04:37<00:00, 4.22s/it]

============================================================
Preprocessing Complete: 76311 total segments generated.
Initial Feature Shape: (76311, 128, 4, 1)
============================================================
Saving full dataset to .../full_data_preprocessed.npz...
Save complete. Data is secure on Drive.
Memory cleared for X_all and y_all.
Class names saved to .../class_names.pkl
```

So, in the actual run: a 503 MB zip download (~28 seconds), extracting into `DeepShip-main`, 4 classes auto-detected, and 63 raw recordings turned into **76,311 preprocessed 128×4 spectrogram segments** in about 4 minutes 37 seconds (≈4.2 seconds/file - this preprocessing step is CPU-bound audio decoding + FFT work, so it doesn't benefit from the GPU that later training will use).

#### Why It's Written This Way

- **All-in-one, single-cell execution.** Download, extraction, path resolution, preprocessing, and saving all happen in one cell so that re-running the notebook from a clean runtime is a single click rather than a multi-step manual process.
- **Immediate save-and-delete pattern.** As soon as `X_all`/`y_all` are computed, they're written to compressed `.npz` on Drive and then `del`eted from memory. This means a crash or disconnect *after* this cell finishes doesn't cost you the ~5 minutes of preprocessing - Cell 5 can just reload from disk.
- **Nested-folder path handling.** Checking for both `DeepShip-main` and `DeepShip` subfolders (rather than assuming one or the other) is defensive coding around how the archive happens to be packaged - the `-main` suffix is the same convention GitHub uses when it zips a repository, which is a strong hint this Kaggle dataset was mirrored from a GitHub-hosted release of DeepShip.

#### Technical Considerations

- **The `kaggle.json` step actually failed in this exact run** - the captured output includes `cp: cannot stat '/content/kaggle.json': No such file or directory`, meaning no fresh credentials file was uploaded to `/content` before running this cell. The download still succeeded anyway, which means valid Kaggle credentials were already cached in `~/.kaggle/kaggle.json` from earlier in the session. **This is not something to rely on for a fresh runtime** - for a clean run, you need to:
  1. Go to your Kaggle account → **Settings → API → Create New Token**, which downloads a `kaggle.json` file.
  2. Upload that file to `/content/kaggle.json` in the Colab file browser *before* running this cell.
  
  If you skip this and no cached credentials exist, `kaggle datasets download` will fail with a 401 error, `ZIP_PATH` won't exist, and the cell will stop with the explicit `FileNotFoundError` raised in step 3.
- **Shell commands via `!` don't raise Python exceptions on failure.** The `!mkdir`, `!cp`, and `!chmod` lines will print error text to the output (as they did here) but won't halt execution - only the explicit `raise FileNotFoundError(...)` after the zip-extraction check acts as a real guard rail.
- **This cell silently overrides `DATASET_ROOT_PATH` from Cell 1.** Whatever Drive-based path Cell 1 configured is discarded the moment step 4 above runs - the value used for the rest of the notebook is always the freshly downloaded Kaggle copy, at `/content/DeepShip_Dataset/DeepShip-main`. If you intended to preprocess a *different*, custom dataset already sitting on your Drive, don't run this cell - go back to using Cell 1's `DATASET_ROOT_PATH` with Cell 3's functions directly instead.
- **`/content` (not Drive) is where the raw zip and extracted `.wav` files live.** Only the *processed* `.npz`/`.pkl` outputs are written to Drive (`SAVE_DIR`). This is deliberate: keeping ~500 MB of raw audio off Drive saves storage quota, since it can always be re-downloaded from Kaggle, while the much smaller processed features are worth persisting.

---

### Cell 4.5: Mel-Spectrogram Matrix Visualization

#### What It Does

Labeled in the notebook as *"Professional Mel-Spectrogram Matrix Visualization,"* this cell builds a grid of example spectrograms - one row per class, up to 4 columns of randomly sampled files per class - as a sanity check that the data actually looks the way you'd expect before spending time training a model on it.

```python
SAMPLES_PER_CLASS = 4

class_names = sorted([d for d in os.listdir(DATASET_ROOT_PATH)
                      if os.path.isdir(os.path.join(DATASET_ROOT_PATH, d))])
num_classes = len(class_names)

fig, axes = plt.subplots(
    nrows=num_classes, ncols=SAMPLES_PER_CLASS,
    figsize=(4 * SAMPLES_PER_CLASS + 2, 2.5 * num_classes),
    dpi=120, squeeze=False
)

for row_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(DATASET_ROOT_PATH, class_name)
    audio_files = glob.glob(os.path.join(class_dir, '**', '*.wav'), recursive=True)
    sampled_files = random.sample(audio_files, min(SAMPLES_PER_CLASS, len(audio_files)))

    for col_idx in range(SAMPLES_PER_CLASS):
        ax = axes[row_idx, col_idx]
        if col_idx < len(sampled_files):
            y, sr = librosa.load(sampled_files[col_idx], sr=22050, mono=True)
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128,
                power=2.0, window='hann', fmax=sr // 2
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            img = librosa.display.specshow(
                mel_spec_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel',
                fmax=sr // 2, cmap='magma', ax=ax
            )
            fig.colorbar(img, ax=ax, format='%+2.0f', pad=0.02).set_label('dB', size=8)
            ax.set_title(f"{class_name} | {display_name}", fontsize=10, fontweight='bold')
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'Insufficient Class Samples', ha='center', va='center',
                    fontsize=8, color='gray', transform=ax.transAxes)

plt.suptitle("DeepShip Mel-Spectrogram Distribution Matrix", fontsize=16, fontweight='bold')
plt.show()
```

The result is a `num_classes × 4` grid figure (here, 4×4 = 16 panels), each showing a **full-length** Mel-spectrogram (not the 75ms training segments) for one randomly chosen recording, using the `magma` colormap with a dB colorbar, mel-frequency on the y-axis, and time on the x-axis.

#### Why It's Written This Way

- **Full-length spectrograms, not training segments.** This cell recomputes a Mel-spectrogram over the *entire* audio file, with no segmenting, for visualization purposes - it's meant to give you a human-readable overview of what each class's recordings broadly look like, which the 75ms training segments (a fraction of a second wide) wouldn't show.
- **Graceful degradation for classes with fewer than 4 files.** The `else` branch - with the code comment *"Handle missing data elegantly (e.g., Tug class missing its 4th file)"* - turns off the axis and displays a placeholder label instead of crashing when `random.sample` can't find 4 files for a class. This comment is a strong hint that in the real dataset, at least one class (very plausibly **Tug**, the smallest class per the [dataset breakdown](#dataset)) doesn't have a full 4 source recordings to sample from.
- **Dynamic figure sizing.** Both `figsize` and the subplot grid dimensions scale with `num_classes` and `SAMPLES_PER_CLASS`, so this cell would still produce a sensible layout even if you pointed it at a dataset with a different number of classes.

#### Technical Considerations

- **Not reproducible run-to-run as written.** `random.sample` draws from Python's global random module, which hasn't been seeded anywhere yet at this point in the notebook (the `RANDOM_SEED` used later, in Cell 7, only seeds NumPy and TensorFlow). Re-running this cell will show a different random sample of files each time. If you need a fixed, reproducible gallery, add `random.seed(...)` before the sampling loop.
- **This figure is never saved to disk.** Unlike Cells 15 and 16, there's no `plt.savefig(...)` call here - only `plt.show()`. If you want to keep this visualization, add a `plt.savefig(os.path.join(SAVE_DIR, 'spectrogram_matrix.png'), dpi=300, bbox_inches='tight')` line before `plt.show()`.
- **This cell re-derives `class_names` and `DATASET_ROOT_PATH`-dependent logic independently of Cell 3.** It duplicates the "list subdirectories" logic from `get_deepship_classes` inline rather than calling that function - harmless here since both use the same sorting logic, but worth knowing if you ever change how classes are discovered in one place and forget to update the other.

---

### Cell 5: Data Splitting and Final Save

#### What It Does

Reloads the full preprocessed dataset, splits it into training, validation, and test sets using **stratified sampling**, and saves each split to its own compressed file.

```python
data = np.load(os.path.join(SAVE_DIR, 'full_data_preprocessed.npz'))
X, y = data['X'], data['y']

TRAIN_RATIO, TEST_RATIO, VAL_RATIO = 0.7, 0.2, 0.1
RANDOM_STATE = 42

# First split: 70% train vs 30% (test+val) combined
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=(TEST_RATIO + VAL_RATIO), stratify=y, random_state=RANDOM_STATE
)

# Second split: carve the 30% into 20% test / 10% val
val_ratio_adjusted = VAL_RATIO / (TEST_RATIO + VAL_RATIO)  # = 1/3
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=val_ratio_adjusted, stratify=y_temp, random_state=RANDOM_STATE
)

np.savez_compressed(os.path.join(SAVE_DIR, 'train_data.npz'), X=X_train, y=y_train)
np.savez_compressed(os.path.join(SAVE_DIR, 'test_data.npz'), X=X_test, y=y_test)
np.savez_compressed(os.path.join(SAVE_DIR, 'val_data.npz'), X=X_val, y=y_val)

del X, y, X_temp, y_temp
```

**Captured output:**
```
Data reloaded. Total samples: 76311
Data shape: (76311, 128, 4, 1)

Performing stratified splitting (70% Train, 20% Test, 10% Val)...
Splitting complete. Saving final array files to Drive...

============================================================
DATASET SPLIT SUMMARY
============================================================
Training samples:   53,417 (70.0%)
Testing samples:    15,262 (20.0%)
Validation samples: 7,632 (10.0%)

Final files successfully saved to: .../processed_data_DeepShip_DWSTr
Memory cleared.
```

#### Why It's Written This Way

- **Two-stage splitting to hit an exact 70/20/10 ratio.** `train_test_split` only splits into two parts at a time, so getting three parts requires calling it twice: first peeling off 70% for training, then splitting the *remaining* 30% again in the correct proportion (`val_ratio_adjusted = 0.1 / 0.3 = 1/3`) so that the final test and validation sets come out to exactly 20% and 10% of the *original* total, not 20%/10% of the remaining 30%.
- **`stratify=y` on both splits.** Without stratification, a random split could easily under- or over-represent the rarer classes (like Tug) in the validation or test sets purely by chance. Stratifying on `y` both times keeps the ~40/29/20/10% class balance consistent across all three splits - which is exactly what makes the [dataset breakdown table](#dataset) a valid way to estimate full-dataset counts from the test set alone.
- **Fixed `RANDOM_STATE = 42`** for both split calls means re-running this cell (on the same input data) always produces byte-identical train/val/test partitions - essential for comparing training runs fairly.
- **Reload-then-split, rather than splitting directly after Cell 4's preprocessing**, decouples preprocessing from splitting: you can experiment with different split ratios or strategies without re-running the ~5-minute audio preprocessing step each time, as long as `full_data_preprocessed.npz` already exists on Drive.

#### Technical Considerations

- **"Low memory cost" reload is accurate here, but only because the feature shape is tiny.** The cell's own comment describes this reload as low-cost; concretely, the full `(76311, 128, 4, 1)` float32 array is about **149 MB**, and the resulting train/test/val arrays are about 104 MB / 30 MB / 15 MB respectively - all comfortably within Colab's default RAM even without a high-RAM runtime. This works out favorably specifically because each sample is only a 128×4 patch; a design using longer spectrogram windows would make this reload step considerably heavier.
- **The `del X, y, X_temp, y_temp` line only removes the intermediate arrays**, not `X_train`/`X_test`/`X_val`, which remain in memory *and* on disk simultaneously at the end of this cell - by design, since Cell 6 re-loads them from disk again rather than assuming they're still in memory (making it safe to run Cell 6 in a fresh session).

---

### Cell 6: Load Final Data and Sanity Check

#### What It Does

Loads the three split `.npz` files (and the pickled class names) back from disk, and runs a couple of quick integrity checks before letting the model-building cells touch the data.

```python
def load_final_data(data_dir: str):
    with np.load(os.path.join(data_dir, 'train_data.npz')) as data:
        X_train, y_train = data['X'], data['y']
    with np.load(os.path.join(data_dir, 'test_data.npz')) as data:
        X_test, y_test = data['X'], data['y']
    with np.load(os.path.join(data_dir, 'val_data.npz')) as data:
        X_val, y_val = data['X'], data['y']
    with open(os.path.join(data_dir, 'class_names.pkl'), 'rb') as f:
        class_names = pickle.load(f)
    return X_train, X_test, X_val, y_train, y_test, y_val, class_names

X_train, X_test, X_val, y_train, y_test, y_val, class_names = load_final_data(SAVE_DIR)

print(f"Expected feature shape (128, 4, 1): {X_train.shape[1:] == (128, 4, 1)}")
```

**Captured output:**
```
============================================================
Final Data Loaded for Model Training
============================================================
X_train shape: (53417, 128, 4, 1)
X_test shape: (15262, 128, 4, 1)
X_val shape: (7632, 128, 4, 1)
Number of Classes: 4 (['Cargo', 'Passengership', 'Tanker', 'Tug'])

Running quick integrity check:
  Expected feature shape (128, 4, 1): True
  Train label count: 53417 | Test label count: 15262
```

#### Why It's Written This Way

- **The `with np.load(...) as data:` pattern** opens each `.npz` archive, immediately pulls out just the two arrays it needs (`X` and `y`), and lets the file handle close automatically - more efficient and less error-prone than manually managing file handles for three separate archives.
- **This cell is a clean re-entry point.** Because it reloads everything from disk rather than assuming any variables survived from earlier cells, you can restart the Colab runtime, skip straight to Cell 6 (assuming Cells 1–5 have been run at least once before), and pick up exactly where you left off - without re-downloading or re-splitting anything.
- **The shape assertion is informational, not a hard stop.** `X_train.shape[1:] == (128, 4, 1)` just gets printed as `True`/`False` - if it printed `False`, the notebook would keep running with mismatched data rather than halting. Turning this into a real `assert` statement (so a shape mismatch raises an error immediately) would catch a corrupted or mismatched save file much earlier and more loudly.

#### Technical Considerations

- **This function reads `data_dir` files positionally by name** (`train_data.npz`, `test_data.npz`, `val_data.npz`, `class_names.pkl`) - if you rename any of Cell 5's output files, this cell needs the same rename.
- **The train/test count check (`53417`/`15262`) is a light form of cross-validation against Cell 5's own printed summary** - worth glancing at both outputs together the first time you run the notebook, since a mismatch here would indicate the save/reload round-trip lost or duplicated data somewhere.

---

### Cell 7: Import TensorFlow/Keras and Set Random Seeds

#### What It Does

Brings in the deep learning stack, fixes random seeds for reproducibility, and configures GPU memory behavior.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

**Captured output:**
```
TensorFlow version: 2.20.0
GPU Available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
✅ GPU memory growth enabled
```

#### Why It's Written This Way

- **Seeding both NumPy and TensorFlow** (rather than just one) matters because this notebook uses both libraries for randomness that affects the final model: NumPy for the earlier stratified split (Cell 5) and any NumPy-side shuffling, TensorFlow for weight initialization and any TF-native random ops during training.
- **`set_memory_growth(gpu, True)`** tells TensorFlow to allocate GPU memory incrementally, as needed, instead of its default behavior of grabbing (nearly) the entire GPU's memory the moment the first op runs. This matters most when sharing a GPU with other processes, or simply to get a more informative out-of-memory error later (pointing at the operation that actually needed more memory) instead of an opaque allocation failure at import time.
- **Importing `matplotlib.pyplot` here, not in Cell 1**, keeps all the deep-learning-stack imports together in one place, right before they're first needed by Cell 8 onward.

#### Technical Considerations

- **Seeding does not guarantee bit-for-bit determinism on GPU.** Many cuDNN convolution and attention kernels use non-deterministic reduction orders for performance reasons; `tf.random.set_seed` controls the *sequence* of random numbers TensorFlow draws, but not necessarily floating-point summation order on GPU hardware. Expect results to be very close, but not necessarily identical, across repeated runs on GPU - for the strictest reproducibility, `tf.config.experimental.enable_op_determinism()` would need to be added (at some training-speed cost).
- **Running this cell more than once is harmless** - reseeding and re-checking GPU status doesn't disturb any already-loaded data or already-built model, so it's safe to re-run if, e.g., a runtime is reconnected mid-session.

---

### Cell 8: Depthwise Separable Convolution (DWS) Block

#### What It Does

Defines the convolutional front end of DWSTr as a small, reusable Keras sub-model: one depthwise convolution followed by one pointwise convolution, each with batch normalization and a ReLU activation.

```python
def create_dws_block(input_shape=(128, 4, 1), num_filters=64):
    inputs = layers.Input(shape=input_shape, name='mel_spectrogram_input')

    x = layers.DepthwiseConv2D(
        kernel_size=(3, 3), strides=(1, 1), padding='same',
        dilation_rate=(1, 1), depthwise_initializer='glorot_uniform', name='depthwise_conv'
    )(inputs)
    x = layers.BatchNormalization(name='dw_batch_norm')(x)
    x = layers.ReLU(name='dw_relu')(x)

    x = layers.Conv2D(
        filters=num_filters, kernel_size=(1, 1), strides=(1, 1),
        padding='valid', kernel_initializer='glorot_uniform', name='pointwise_conv'
    )(x)
    x = layers.BatchNormalization(name='pw_batch_norm')(x)
    x = layers.ReLU(name='pw_relu')(x)

    return models.Model(inputs=inputs, outputs=x, name='DWS_Block')
```

This is the standard **depthwise separable convolution** pattern popularized by MobileNet: instead of one standard `Conv2D` that mixes spatial filtering and channel mixing together, it's factored into two cheaper steps run back-to-back:

1. **Depthwise convolution** - a 3×3 spatial filter applied independently to each input channel (no cross-channel mixing at this stage), `padding='same'` so the spatial dimensions (128×4) are preserved.
2. **Pointwise convolution** - a 1×1 convolution that *does* mix channels, expanding from however many channels came out of the depthwise step up to `num_filters=64`.

Each convolution is followed by batch normalization (stabilizes and speeds up training by normalizing layer activations) and a ReLU.

#### Why It's Written This Way

- **Parameter efficiency.** Splitting a standard convolution into depthwise + pointwise steps is the entire point of "separable" convolution: for a standard `k×k` convolution mapping `C_in` channels to `C_out` channels, the cost is roughly `k²·C_in·C_out` parameters; splitting it into depthwise (`k²·C_in`) + pointwise (`C_in·C_out`) is cheaper whenever `C_out` is reasonably large, which is exactly the 1→64 channel expansion happening here.
- **A named, self-contained sub-model** (`name='DWS_Block'`), rather than a bare function that returns a tensor, means this block shows up as its own labeled row in `model.summary()` (see [Cell 11](#cell-11-build-and-compile-model)) and can be inspected, saved, or reused independently of the larger DWSTr model.

#### Technical Considerations

- **With only 1 input channel, the "separable" savings mostly show up at the pointwise step, not the depthwise step.** The depthwise convolution here is a single 3×3 filter over a single-channel input (`kernel_size=(3,3)`, `in_channels=1`) - barely different in cost from an ordinary `Conv2D(1, (3,3))`. The real efficiency win is in the pointwise step: expanding 1 channel to 64 with a 1×1 convolution costs only `1×64 + 64 (bias) = 128` parameters, versus `3×3×1×64 + 64 = 640` parameters a single standard `Conv2D(64, (3,3))` would cost to go straight from 1 to 64 channels. The efficiency argument for depthwise separable convolutions becomes much stronger in deeper networks where the *input* channel count is also large (e.g., 64 → 128), which isn't quite the situation on this very first layer.
- **Every non-trainable parameter in the entire DWSTr model comes from this block.** The two `BatchNormalization` layers here are the *only* batch-norm layers anywhere in the architecture (the Transformer stack uses `LayerNormalization` instead, which has no non-trainable statistics) - worth remembering when you see "130 non-trainable params" in the [full model summary](#cell-11-build-and-compile-model); all 130 of them live in these two layers (2 from the 1-channel batch norm after the depthwise step, 128 from the 64-channel batch norm after the pointwise step).
- **`padding='same'` on the depthwise convolution, `padding='valid'` on the pointwise convolution** - this pairing is standard for this block type: `'same'` keeps spatial dimensions unchanged where padding actually matters (a 3×3 spatial filter), while `'valid'` is a no-op difference for a 1×1 filter (there's no padding to add either way), so it's really just an explicit default in the pointwise case.

---

### Cell 9: Transformer Encoder Block

#### What It Does

Defines `TransformerBlock`, a custom Keras layer implementing one **pre-norm Transformer encoder block** - the same fundamental building block used in Vision Transformers (ViT), BERT, and GPT-style architectures, here repurposed to process sequences of spectrogram patches instead of image patches or text tokens.

```python
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate=0.3, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate

        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate,
            name="multi_head_attention"
        )
        self.mlp = keras.Sequential([
            layers.Dense(mlp_dim, activation='gelu'),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim),
            layers.Dropout(dropout_rate),
        ], name="mlp_block")

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6, name="layernorm_1")
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6, name="layernorm_2")

    def call(self, inputs, training=False):
        inputs_norm = self.layernorm1(inputs)
        attn_output = self.att(query=inputs_norm, value=inputs_norm, key=inputs_norm, training=training)
        x = inputs + attn_output                      # residual connection #1

        x_norm = self.layernorm2(x)
        mlp_output = self.mlp(x_norm, training=training)
        return x + mlp_output                          # residual connection #2

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim, "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim, "dropout_rate": self.dropout_rate,
        })
        return config
```

Each block has exactly two sub-layers, each wrapped in a **pre-norm residual connection** (`x + sublayer(norm(x))`, rather than the older `norm(x + sublayer(x))` post-norm style):

1. **Self-attention sub-layer** - `LayerNormalization` → `MultiHeadAttention` (the same tensor used as query, key, *and* value, hence *self*-attention) → added back to the un-normalized input.
2. **MLP sub-layer** - `LayerNormalization` → a small feed-forward network (`Dense(mlp_dim, gelu)` → `Dropout` → `Dense(embed_dim)` → `Dropout`) → added back to the residual stream.

#### Why It's Written This Way

- **Pre-norm over post-norm.** Normalizing *before* each sub-layer (rather than after adding the residual) is the modern standard because it keeps gradients flowing more stably through very deep stacks - with 6 blocks chained together (see [Cell 10](#cell-10-complete-dwstr-architecture)), this matters for keeping training numerically well-behaved.
- **Residual connections around both sub-layers** let the network default to an identity mapping if a given block isn't useful for a particular input, rather than forcing every input through the full transformation - this is what makes stacking many Transformer blocks trainable at all, rather than suffering vanishing/exploding gradients.
- **GELU activation in the MLP** (rather than ReLU) is the standard choice in Transformer architectures - it's smoother than ReLU (no hard zero cutoff) and empirically trains slightly better in attention-based models, which is why it appears here and in the classification head in Cell 10, even though the *convolutional* DWS block in Cell 8 uses plain ReLU instead.
- **Dropout in three separate places** - inside the attention operation itself (`dropout=dropout_rate` on `MultiHeadAttention`, which drops attention weights) and twice inside the MLP - stacks up meaningful regularization within a single block, appropriate for a dataset built from only 63 source recordings where overfitting is a real risk.
- **`get_config()` is not optional boilerplate here - it's load-bearing.** Because `TransformerBlock` is a *custom* Keras layer (not a built-in one), Keras has no built-in way to know how to reconstruct it when loading a saved model back from disk. `get_config()` tells Keras exactly which constructor arguments to pass back in; without it, [Cell 14](#cell-14-evaluate-on-test-set)'s `keras.models.load_model(..., custom_objects=...)` call would fail.

#### Technical Considerations

- **`key_dim=embed_dim` is a deliberate, non-default sizing choice.** In Keras's `MultiHeadAttention`, `key_dim` sets the size of *each individual attention head's* query/key vectors - it is not automatically derived from `embed_dim` and `num_heads`. Here, `key_dim` is explicitly set equal to the *full* `embed_dim` (64), not `embed_dim // num_heads` (which would be 16 for 4 heads). This means each of the 4 attention heads operates in a full 64-dimensional space rather than a 16-dimensional slice, giving the attention mechanism more representational capacity per head - at the cost of more parameters and computation than the "standard" head-splitting convention would use. This single choice is responsible for the bulk of each Transformer block's parameter count (see the [parameter breakdown](#cell-11-build-and-compile-model)).
- **The `training` argument is threaded through explicitly** to both the attention call and the MLP's `Sequential` call - necessary because `Dropout` and (if used) `BatchNormalization` layers behave differently at train vs. inference time, and a custom layer's nested sub-layers don't automatically inherit the outer call's training mode unless you pass it down yourself.
- **This block has zero non-trainable parameters** - everything here (attention projections, MLP weights, LayerNorm's gain/bias) is trainable; the model's only non-trainable weights come from the `BatchNormalization` layers back in the DWS block.

---

### Cell 10: Complete DWSTr Architecture

#### What It Does

This cell defines the two remaining custom layers the architecture needs - `ClassTokenLayer` and `PositionalEmbedding` - and then `create_dwstr_model()`, the function that wires the DWS block (Cell 8) and the Transformer blocks (Cell 9) together into the full DWSTr model.

**1. `ClassTokenLayer`**

```python
class ClassTokenLayer(layers.Layer):
    def __init__(self, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.class_token = self.add_weight(
            name='class_token', shape=(1, 1, self.projection_dim),
            initializer='random_normal', trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        class_tokens = tf.broadcast_to(self.class_token, [batch_size, 1, self.projection_dim])
        return tf.concat([class_tokens, inputs], axis=1)
```

A single learnable vector of shape `(1, 1, projection_dim)` - the classification token, directly analogous to BERT's `[CLS]` token or ViT's class token - is broadcast to match the batch size and **prepended** (via `tf.concat(..., axis=1)`) to the front of the patch-embedding sequence. This token carries no information from the input at first; its value is learned purely through gradient descent, driven by how useful its *final*, post-attention representation turns out to be for classification.

**2. `PositionalEmbedding`**

```python
class PositionalEmbedding(layers.Layer):
    def __init__(self, num_positions, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_positions = num_positions
        self.projection_dim = projection_dim
        self.position_embedding = layers.Embedding(input_dim=num_positions, output_dim=projection_dim)

    def call(self, inputs):
        positions = tf.range(start=0, limit=self.num_positions, delta=1)
        position_embeddings = self.position_embedding(positions)
        return inputs + position_embeddings
```

A standard `Embedding` lookup table with one learned `projection_dim`-length vector per sequence position (`0, 1, ..., num_positions - 1`), added element-wise to the token embeddings. This is necessary because self-attention, by itself, is *permutation-invariant* - without positional information, the model would have no way to distinguish "patch at frequency band 3" from "patch at frequency band 30." Note this is **learned**, not the fixed sinusoidal encoding from the original "Attention Is All You Need" paper - a common simplification in ViT-style architectures when the sequence length is small and fixed, as it is here (33 positions, always).

**3. `create_dwstr_model(...)`**

```python
def create_dwstr_model(
    input_shape=(128, 4, 1), num_classes=4, dws_filters=64, patch_size=4,
    projection_dim=64, num_transformer_blocks=6, num_heads=4, mlp_dim=1024, dropout_rate=0.3
):
    inputs = layers.Input(shape=input_shape, name='input')

    # 1. Convolutional feature extraction
    dws_block = create_dws_block(input_shape=input_shape, num_filters=dws_filters)
    feature_map = dws_block(inputs)                          # (128, 4, 64)

    # 2. Patch embedding: turn the 2D feature map into a 1D sequence of patch vectors
    num_patches_h = input_shape[0] // patch_size              # 128 // 4 = 32
    num_patches_w = input_shape[1] // patch_size              # 4   // 4 = 1
    num_patches = num_patches_h * num_patches_w                # 32 patches total

    patches = layers.Reshape((num_patches, (patch_size * patch_size * dws_filters)))(feature_map)
    patch_embeddings = layers.Dense(projection_dim, name='patch_projection')(patches)

    # 3. Prepend class token, add positional embeddings
    patch_embeddings = ClassTokenLayer(projection_dim)(patch_embeddings)   # 32 -> 33 tokens
    num_positions = num_patches + 1
    encoded_patches = PositionalEmbedding(num_positions, projection_dim)(patch_embeddings)
    encoded_patches = layers.Dropout(dropout_rate)(encoded_patches)

    # 4. Transformer encoder stack
    for i in range(num_transformer_blocks):
        encoded_patches = TransformerBlock(
            embed_dim=projection_dim, num_heads=num_heads, mlp_dim=mlp_dim,
            dropout_rate=dropout_rate, name=f'transformer_block_{i}'
        )(encoded_patches)

    # 5. Classification head: pull out the class token and project to logits
    representation = layers.Lambda(lambda x: x[:, 0], name='extract_class_token')(encoded_patches)
    representation = layers.LayerNormalization(epsilon=1e-6)(representation)
    representation = layers.Dropout(dropout_rate)(representation)
    representation = layers.Dense(mlp_dim, activation='gelu')(representation)
    representation = layers.Dropout(dropout_rate)(representation)

    outputs = layers.Dense(num_classes, activation='softmax', name='classification_output')(representation)
    return models.Model(inputs=inputs, outputs=outputs, name='DWSTr')
```

#### Why It's Written This Way - Tracing the Shapes

This is easiest to follow by tracking the tensor shape at every stage, with the notebook's actual default hyperparameters (`input_shape=(128, 4, 1)`, `patch_size=4`, `projection_dim=64`):

| Stage | Shape | What happened |
|---|---|---|
| Input | `(128, 4, 1)` | One 128-mel-bin × 4-time-frame spectrogram segment |
| After DWS block | `(128, 4, 64)` | Depthwise+pointwise conv expands 1 → 64 channels; spatial size unchanged |
| After `Reshape` | `(32, 1024)` | Cut into 32 patches of `4×4×64 = 1024` values each (see note below) |
| After `patch_projection` (`Dense`) | `(32, 64)` | Each 1024-dim patch linearly projected down to a 64-dim embedding |
| After `ClassTokenLayer` | `(33, 64)` | Learnable class token prepended at position 0 |
| After `PositionalEmbedding` | `(33, 64)` | Positional vectors 0–32 added elementwise (shape unchanged) |
| After 6× `TransformerBlock` | `(33, 64)` | Self-attention mixes information across all 33 tokens; shape never changes through the stack |
| After `extract_class_token` (`x[:, 0]`) | `(64,)` | Only the class token's final representation is kept - the 32 patch tokens' final states are discarded |
| After classification head | `(4,)` | Softmax probabilities over Cargo / Passengership / Tanker / Tug |

**The "patch" geometry has an important quirk worth calling out explicitly:** with `patch_size=4` applied to an input that's only `4` wide on the time axis, `num_patches_w = 4 // 4 = 1`. In other words, **patching only subdivides the frequency (mel) axis** - the model carves the 128 mel bins into 32 groups of 4, but each patch spans the *entire* 4-frame time axis in one go. There's no further subdivision along time at all. This is a direct, sensible consequence of the 128×4 input shape chosen back in the preprocessing step ([Cell 2](#cell-2-dwstr-preprocessor-class)): with such a short time axis to begin with, there isn't much time resolution left to subdivide further.

#### Technical Considerations

- **Only the class token's output is used for classification - the 32 patch tokens' final representations are computed and then thrown away.** This is standard ViT/BERT-style pooling, but it does mean all of the "useful" information the 32 patch tokens accumulated through 6 layers of self-attention only matters insofar as it got *written into* the class token via attention - nothing about the patch tokens' own final states is used directly. An alternative design (not used here) would be to global-average-pool over all 33 tokens instead.
- **`layers.Lambda(lambda x: x[:, 0], ...)`** is a simple slicing operation wrapped as a Keras layer so it can sit inside the functional-API graph; it has no learnable parameters.
- **The classification head reuses GELU** (`Dense(mlp_dim, activation='gelu')`) rather than ReLU, keeping activation choice consistent with the Transformer blocks it immediately follows.
- **`num_classes` defaults to 4 in the function signature but is always passed explicitly** from [Cell 11](#cell-11-build-and-compile-model) as `len(class_names)`, so the model automatically adapts if DeepShip's folder structure ever gains or loses a class - you would not need to edit this cell at all for that to work correctly.

---

### Cell 11: Build and Compile Model

#### What It Does

Instantiates the model with a concrete set of hyperparameters, compiles it with an optimizer/loss/metric, and prints a full architecture summary.

```python
MODEL_CONFIG = {
    'input_shape': (128, 4, 1),
    'num_classes': len(class_names),      # 4, dynamically detected
    'dws_filters': 64,
    'patch_size': 4,
    'projection_dim': 64,
    'num_transformer_blocks': 6,
    'num_heads': 4,
    'mlp_dim': 1024,
    'dropout_rate': 0.3
}

model = create_dwstr_model(**MODEL_CONFIG)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
print(f"Total Parameters: {model.count_params():,}")
```

**Captured output (abridged):**
```
Building DWSTr model with configuration:
  input_shape: (128, 4, 1)
  num_classes: 4
  dws_filters: 64
  patch_size: 4
  projection_dim: 64
  num_transformer_blocks: 6
  num_heads: 4
  mlp_dim: 1024
  dropout_rate: 0.3
...
Total params: 1,331,666 (5.08 MB)
Trainable params: 1,331,536 (5.08 MB)
Non-trainable params: 130 (520.00 B)

📊 Total Parameters: 1,331,666
```

#### Why It's Written This Way

- **`sparse_categorical_crossentropy`, not `categorical_crossentropy`.** The labels produced back in Cells 3/4 (`y_list`) are plain integers (`0, 1, 2, 3`), not one-hot vectors. The "sparse" variant of the loss accepts integer labels directly, avoiding an unnecessary `to_categorical()` conversion step and the extra memory a one-hot label array would use.
- **`Adam` with decoupled `weight_decay=0.0001`.** Keras's `Adam` optimizer supports a `weight_decay` argument that implements *decoupled* weight decay (the AdamW formulation) directly, rather than the older trick of adding an L2 penalty to the loss (which interacts with Adam's adaptive learning rates in a way that doesn't correspond to true weight decay). This gives an extra layer of regularization on top of the dropout already built into the model.
- **`num_classes=len(class_names)`** - rather than hardcoding `4` - is what makes the model definition dataset-agnostic: point this pipeline at a version of DeepShip (or any similarly-organized dataset) with a different number of class folders, and the classification head resizes automatically.

#### Full Parameter Breakdown

Working through the architecture component by component (and cross-checked against the exact total Keras reports) shows exactly where all ~1.33M parameters live:

| Component | Parameters | Share |
|---|---:|---:|
| DWS Block (Cell 8) | 398 | 0.03% |
| Patch projection (`Dense`, 1024→64) | 65,600 | 4.93% |
| Class token | 64 | 0.005% |
| Positional embedding (33×64) | 2,112 | 0.16% |
| 6 × Transformer block (198,784 each) | 1,192,704 | 89.56% |
| Classification head (LayerNorm + Dense(1024) + Dense(4)) | 70,788 | 5.32% |
| **Total** | **1,331,666** | **100%** |

**A single Transformer block's 198,784 parameters break down further into:**

| Sub-component | Parameters |
|---|---:|
| Multi-head attention (Q/K/V/output projections, `key_dim=64`, 4 heads) | 66,368 |
| MLP block (`Dense(1024, gelu)` + `Dense(64)`) | 132,160 |
| 2 × `LayerNormalization` | 256 |
| **Per block total** | **198,784** |

The takeaway: nearly **90% of the model's capacity lives in the 6-block Transformer stack**, not the convolutional front end - consistent with the architecture's design philosophy of using DWS convolution as a cheap feature extractor and spending the parameter budget on the attention mechanism instead.

Also worth noting directly: of the **130 non-trainable parameters** in the whole model, **all 130** come from the two `BatchNormalization` layers inside the DWS block (2 from the post-depthwise batch norm, 128 from the post-pointwise batch norm) - there is no other source of non-trainable weights anywhere in the Transformer stack or classification head, since those use `LayerNormalization`, which has no running statistics to track.

#### Technical Considerations

- **`safe_mode=False` will be required to reload this model later** ([Cell 14](#cell-14-evaluate-on-test-set)) because it contains a `Lambda` layer (`extract_class_token`) - Keras 3's default "safe mode" for model loading blocks arbitrary Python callables like `Lambda` layers as a security precaution, since a saved model could otherwise smuggle in arbitrary code.
- **At ~5 MB, this model is small enough to comfortably run inference on CPU** even though training benefits from GPU - worth keeping in mind if you ever want to deploy the trained classifier somewhere without GPU access.
- **The parameter count is entirely independent of dataset size** (train/val/test sample counts don't appear anywhere in this calculation) - only `MODEL_CONFIG`'s architectural hyperparameters determine it, so this exact 1,331,666 figure will reproduce on any DeepShip-shaped (4-class) run of this notebook, regardless of how much data is used to train it.

---

### Cell 12: Setup Training Callbacks

#### What It Does

Configures three Keras callbacks that will run automatically at the end of every training epoch, before any actual training happens.

```python
checkpoint_dir = os.path.join(SAVE_DIR, 'model_checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'dwstr_best_model.keras'),
        monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
    ),
    EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1
    )
]
```

| Callback | Watches | Behavior |
|---|---|---|
| `ModelCheckpoint` | `val_accuracy` (maximize) | Overwrites `dwstr_best_model.keras` every time validation accuracy hits a new high; the file on disk always reflects the single best epoch seen so far. |
| `EarlyStopping` | `val_loss` (minimize) | Stops training if validation loss hasn't improved for 5 consecutive epochs, and rolls the model's weights back to whichever epoch had the lowest validation loss (`restore_best_weights=True`). |
| `ReduceLROnPlateau` | `val_loss` (minimize) | Halves the learning rate (`factor=0.5`) if validation loss hasn't improved for 3 consecutive epochs, down to a floor of `1e-7`. |

#### Why It's Written This Way

- **`EarlyStopping` and `ModelCheckpoint` intentionally watch *different* metrics.** `ModelCheckpoint` optimizes for the best validation *accuracy* (the metric you ultimately care about for a classifier), while `EarlyStopping` and `ReduceLROnPlateau` both watch validation *loss* (a smoother, more sensitive training signal that tends to reveal overfitting earlier than accuracy, which can plateau in coarse steps). This is a deliberate, common pattern - accuracy is a noisier, less-informative training signal than the underlying loss, so loss is a better trigger for "should we stop / reduce the learning rate," even though accuracy is the number that ultimately matters for the checkpoint you keep. In the notebook's own run, these two criteria happened to agree - see [Cell 13](#cell-13-compute-smoothed-class-weights-and-train) - but they aren't guaranteed to.
- **`patience=3` for the learning-rate reduction is shorter than `patience=5` for early stopping**, so the schedule reduces the learning rate *before* giving up entirely - giving the model a chance to keep improving at a lower learning rate rather than stopping the moment progress stalls.
- **`min_lr=1e-7`** puts a floor under the learning-rate decay so it can't shrink indefinitely toward zero and effectively freeze training in a very long run.

#### Technical Considerations

- **`ModelCheckpoint`'s file is overwritten in place, not versioned per epoch** - only the single best-so-far model ever exists at `dwstr_best_model.keras`. If you want every epoch's checkpoint kept separately, the `filepath` would need to include a formatting placeholder like `{epoch:02d}`.
- **These callbacks are configured once, in a Python list, and reused as-is by `model.fit(...)` in the next cell** - since callback objects carry internal state (like "best score seen so far"), re-running Cell 12 to reset that state before a fresh `model.fit()` call is good practice if you intend to retrain from scratch in the same session.

---

### Cell 13: Compute Smoothed Class Weights and Train

#### What It Does

This is the main training cell: it first computes per-class sample weights to counteract the dataset's class imbalance, then calls `model.fit(...)`.

```python
BATCH_SIZE = 64
EPOCHS = 100  # EarlyStopping will catch this automatically

unique_classes = np.unique(y_train)
standard_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)

# Square-root smoothing
smoothed_weights = [math.sqrt(w) for w in standard_weights]
class_weight_dict = dict(enumerate(smoothed_weights))

history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE, epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)
```

**Class weighting, in two steps:**

1. **`compute_class_weight('balanced', ...)`** implements the standard inverse-frequency formula:

   $$w_c = \frac{N}{K \cdot n_c}$$

   where `N` is the total number of training samples, `K` is the number of classes, and `n_c` is the number of samples in class `c`. A rarer class gets a proportionally larger weight, so misclassifying one of its (few) examples costs the loss function just as much, in aggregate, as misclassifying one of the common class's (many) examples.

2. **Square-root smoothing** (`math.sqrt(w)` on every weight) pulls all the weights **toward 1.0**, compressing the gap between the most- and least-weighted classes. This is a deliberate *damping* of the full "balanced" correction - fully balancing an imbalance this large can overcorrect, causing the model to over-prioritize the rare class at the expense of overall accuracy, or destabilize training with unusually large gradient contributions from a handful of examples.

   Using the [approximate class distribution](#dataset) for the training set (Cargo ≈21,696, Passengership ≈10,696, Tanker ≈15,491, Tug ≈5,534 out of 53,417 total), the balanced weights and their square-root-smoothed versions work out to approximately:

   | Class | Balanced weight ($N / (4 n_c)$) | Square-root-smoothed weight |
   |---|---:|---:|
   | Cargo (largest) | ≈0.616 | ≈0.785 |
   | Passengership | ≈1.249 | ≈1.117 |
   | Tanker | ≈0.862 | ≈0.929 |
   | Tug (smallest) | ≈2.413 | ≈1.554 |

   *(These are illustrative, back-calculated from the estimated class distribution - the notebook doesn't print the exact weight values, only a confirmation that they were computed - but the shape of the effect is accurate: without smoothing, Tug's weight would be roughly 2.4×; with smoothing, it's pulled down to roughly 1.55×.)*

#### What Actually Happened During Training

Training ran for **58 of the maximum 100 epochs** before `EarlyStopping` intervened. The learning rate was cut in half five times along the way:

| Epoch | Event |
|---|---|
| 22 | LR reduced: `0.001` → `0.0005` |
| 38 | LR reduced: `0.0005` → `0.00025` |
| 42 | LR reduced: `0.00025` → `0.000125` |
| 46 | LR reduced: `0.000125` → `0.0000625` |
| 56 | LR reduced: `0.0000625` → `0.00003125` |
| 58 | `EarlyStopping` triggers; weights restored from **epoch 53** |

And the training trajectory itself (train / validation accuracy and loss at representative epochs):

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|---|---:|---:|---:|---:|
| 1 | 43.18% | 1.1211 | 49.17% | 0.9855 |
| 9 | 77.78% | 0.5170 | 81.53% | 0.4499 |
| 17 | 87.86% | 0.3058 | 89.92% | 0.2706 |
| 25 | 93.80% | 0.1604 | 95.37% | 0.1351 |
| 41 | 96.48% | 0.0894 | 96.28% | 0.1194 |
| **53 (best)** | **97.48%** | **0.0638** | **97.75%** | **0.0701** |
| 58 (last) | 97.67% | 0.0589 | 97.64% | 0.0743 |

Notably, in this run, **epoch 53 was simultaneously the best validation-accuracy epoch *and* the best validation-loss epoch** - so `ModelCheckpoint` (watching accuracy) and `EarlyStopping`'s weight restoration (watching loss) ended up agreeing on the exact same checkpoint, even though they were configured to watch different metrics. The final saved model (`dwstr_best_model.keras`) and the in-memory model after `model.fit()` returns are therefore identical for this particular run.

#### Why It's Written This Way

- **`class_weight`, not resampling.** Rather than physically duplicating minority-class samples (oversampling) or discarding majority-class samples (undersampling), passing a `class_weight` dictionary to `model.fit()` reweights each sample's contribution to the loss function directly - no data duplication, no information thrown away, and every real segment is seen exactly once per epoch.
- **`EPOCHS = 100` is a ceiling, not a target.** The comment in the code (*"EarlyStopping will catch this automatically"*) makes the intent explicit: 100 is deliberately set high so that `EarlyStopping` - not an arbitrary fixed epoch count - is what actually decides when training stops. In this run, that happened at epoch 58, well short of the 100-epoch ceiling.

#### Technical Considerations

- **`class_weight` in Keras only reweights the loss, not the validation metrics.** Validation accuracy/loss (what `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` all watch) are computed on the *unweighted* validation set, which is the correct behavior for getting an honest read on real-world performance - only the training loss that drives gradient updates is affected by the class weights.
- **Total training time isn't printed explicitly**, but individual epoch timings in the log (visible after the first, slower epoch, which includes some one-time warm-up/compilation overhead) run in the 20–25 second range on the T4 GPU used for this run, putting total training time for all 58 epochs at roughly **20–25 minutes**.

---

### Cell 14: Evaluate on Test Set

#### What It Does

Reloads the best saved checkpoint from disk (rather than trusting the in-memory `model` object) and evaluates it once against the held-out test set - data the model has never seen in any form, not even for validation-based early stopping decisions.

```python
custom_objects = {
    'TransformerBlock': TransformerBlock,
    'ClassTokenLayer': ClassTokenLayer,
    'PositionalEmbedding': PositionalEmbedding
}

checkpoint_path = os.path.join(SAVE_DIR, 'model_checkpoints', 'dwstr_best_model.keras')
best_model = keras.models.load_model(checkpoint_path, custom_objects=custom_objects, safe_mode=False)

test_loss, test_accuracy = best_model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=1)
```

**Captured output:**
```
239/239 ━━━━━━━━━━━━━━━━━━━━ 8s 14ms/step - accuracy: 0.9780 - loss: 0.0624

📊 Final Test Results:
  Loss: 0.0624
  Accuracy: 97.80%
```

#### Why It's Written This Way

- **Reloading from disk, rather than reusing the in-memory `model` variable, is a deliberate correctness check.** Even though `EarlyStopping(restore_best_weights=True)` should already leave `model` holding the best-epoch weights in memory, explicitly reloading `dwstr_best_model.keras` from disk verifies that what actually got *saved* matches what you intend to evaluate and ultimately deploy - catching, for instance, a save/serialization bug that wouldn't be visible if you only ever evaluated the in-memory object.
- **`custom_objects` is required precisely because of the three custom layer classes** defined in Cells 9 and 10. Keras's model-loading mechanism needs an explicit mapping from each class *name* (as a string, stored in the `.keras` file) back to the actual Python class, since it has no way to import your notebook's custom code automatically - this is exactly what each class's `get_config()` method (see [Cell 9](#cell-9-transformer-encoder-block)) was preparing for.
- **`batch_size=64`** reuses the same batch size training used - not strictly required for evaluation, but keeps memory usage and throughput consistent with the rest of the notebook.

#### Technical Considerations

- **`safe_mode=False` is required here too**, for the same `Lambda`-layer reason noted in [Cell 11](#cell-11-build-and-compile-model). Only pass `safe_mode=False` for models you built yourself (or otherwise trust) - Keras's safe mode exists specifically to prevent loading a `.keras` file from executing arbitrary attacker-controlled code via a malicious `Lambda` layer.
- **97.80% test accuracy essentially matches the 97.75% validation accuracy** from the best training epoch - a good sign that the model generalizes consistently to genuinely unseen data, rather than having overfit to quirks of the validation set specifically.
- **This cell must be run *after* Cells 9 and 10** (or at least after those classes have been defined) in a given session - even in a fresh runtime, `custom_objects` needs live references to the actual `TransformerBlock`, `ClassTokenLayer`, and `PositionalEmbedding` classes, not just their names.

---

### Cell 15: Plot Training History

#### What It Does

Defines and immediately calls `plot_training_history()`, which renders the full epoch-by-epoch accuracy and loss curves (training vs. validation) side by side, and saves the figure to disk.

```python
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.show()

plot_training_history(history)
```

Both curves are pulled directly from the `history` object `model.fit()` returned in Cell 13 - `history.history` is a plain dictionary with one list per tracked metric (`'accuracy'`, `'val_accuracy'`, `'loss'`, `'val_loss'`), one value per epoch actually run (58 entries each, in this run, since `EarlyStopping` cut training short of the 100-epoch ceiling).

Based on the [training trajectory captured in Cell 13](#cell-13-compute-smoothed-class-weights-and-train), the resulting plot shows both accuracy curves climbing steeply for the first ~20 epochs (from ~43%/49% up past 90%), then flattening into a long, gradual climb through the mid-90s as the learning-rate reductions kick in, with training and validation curves tracking each other closely throughout - visually confirming the absence of any significant overfitting gap.

#### Why It's Written This Way

- **Side-by-side subplots, not overlaid on one axis.** Accuracy (bounded 0–1) and loss (unbounded, and on a very different numeric scale, especially early in training when loss starts above 1.0) don't share a sensible y-axis, so plotting them separately avoids the visual distortion of squeezing both onto one scale.
- **`dpi=300, bbox_inches='tight'`** produces a publication-quality, print-resolution PNG with minimal whitespace padding around the figure - appropriate for pasting directly into a report or paper rather than just eyeballing on screen.
- **Saving *before* `plt.show()`** ensures the file is written to disk regardless of whether the interactive display step renders correctly in whatever environment the notebook happens to be running in.

#### Technical Considerations

- **This cell will raise a `NameError` if run out of order** - it depends entirely on the `history` variable produced by Cell 13's `model.fit()` call, so re-running just this cell in isolation (e.g., after a runtime restart) without first re-running training will fail. Unlike Cells 6 and 14, there's no disk-based reload path for training history - if you need to revisit these curves later without retraining, you'd need to have separately saved `history.history` (e.g., via `pickle`) during Cell 13.
- **The output file, `training_history.png`, lands in `SAVE_DIR`** alongside the other persisted artifacts (`.npz` files, `class_names.pkl`, and the `model_checkpoints/` folder) - everything the pipeline produces ends up in one place on Drive.

---

### Cell 16: Confusion Matrix and Per-Class Performance

#### What It Does

The final cell: generates predictions for the entire test set, prints a full per-class precision/recall/F1 report, and renders a confusion matrix heatmap.

```python
y_pred = best_model.predict(X_test, batch_size=BATCH_SIZE)
y_pred_classes = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred_classes, target_names=class_names, digits=4))

cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Segment Count'})
plt.title('Confusion Matrix - DeepShip DWSTr', fontsize=14, fontweight='bold')
plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.show()

per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
for i, class_name in enumerate(class_names):
    print(f"  {class_name:15s}: {per_class_accuracy[i]*100:.2f}%")
```

**Captured output - full classification report:**
```
               precision    recall  f1-score   support

        Cargo     0.9834    0.9777    0.9806      6199
Passengership     0.9697    0.9836    0.9766      3056
       Tanker     0.9797    0.9690    0.9743      4426
          Tug     0.9685    0.9930    0.9806      1581

     accuracy                         0.9780     15262
    macro avg     0.9753    0.9809    0.9780     15262
 weighted avg     0.9781    0.9780    0.9780     15262
```

**Captured output - per-class accuracy:**
```
  Cargo          : 97.77%
  Passengership  : 98.36%
  Tanker         : 96.90%
  Tug            : 99.30%
```

#### Why It's Written This Way

- **`argmax(y_pred, axis=1)`** converts the model's softmax output - a `(15262, 4)` array of class probabilities - into a single predicted class index per sample, by picking whichever of the 4 probabilities is highest. This is the standard way to turn a probabilistic classifier's output into a hard label for reporting metrics like precision and recall.
- **Precision, recall, and F1 give a fuller picture than accuracy alone**, especially on an imbalanced dataset like this one: a model could reach high *overall* accuracy just by being good at the majority class (Cargo) while quietly failing on the minority class (Tug). Reporting all three per class - and per-class accuracy on top - is exactly how you'd catch that failure mode if it existed.
- **`cm.diagonal() / cm.sum(axis=1)`** computes *recall* by another name: for each true class (each row of the confusion matrix), the diagonal entry is how many of that class's samples were correctly predicted, and dividing by the row sum (all samples that were truly that class) gives the fraction correctly identified - which is why the "per-class accuracy" numbers printed here exactly match the "recall" column in the classification report above.
- **`seaborn.heatmap` with `annot=True, fmt='d'`** overlays the exact integer segment count in every cell, so the matrix is both a visual pattern (via the `Blues` color intensity) and an exact lookup table at the same time.

#### Reading the Results

- **No class fell below 96.9% accuracy**, despite the ~4:1 imbalance between Cargo and Tug - direct evidence that the square-root-smoothed class weighting in Cell 13 worked as intended, rather than the model just defaulting to always predicting the majority class.
- **Tug - the smallest class by far (only ~10% of the data) - actually has the *highest* recall/accuracy of all four classes (99.30%)**, and its precision (96.85%) is the lowest of the four. Put differently: the model catches almost every real Tug segment, at the cost of occasionally mislabeling a small number of non-Tug segments *as* Tug. This is a plausible, even desirable, side effect of up-weighting a rare class in the loss function - it pushes the model to err on the side of predicting the rare class when uncertain, trading a bit of precision for a meaningful gain in recall on the class that matters most to get right precisely because it has so little data to begin with.
- **Cargo and Tanker - the two largest classes - show the most confusion with each other** relative to their own high individual scores (both still above 96.9%), which would show up in the confusion matrix as the largest off-diagonal values outside the diagonal itself; this is a reasonable pattern if large cargo vessels and tankers share broadly similar hull sizes and engine profiles compared to the more visually and acoustically distinct Passengership and Tug classes.

#### Technical Considerations

- **This cell depends on `best_model` from Cell 14**, not the `model` object from Cell 11/13 - so, as with Cell 14, this only reflects the *best checkpoint*, not necessarily whatever the in-memory `model` variable holds if you've since done anything else with it in the same session.
- **`predict()` runs a second full forward pass over the test set**, separate from the one `evaluate()` already did in Cell 14 - slightly redundant computationally, but necessary because `evaluate()` only returns aggregate loss/accuracy, not the raw per-sample predictions needed to build a confusion matrix.

---

## Results and Performance

### Final Model Performance

| Metric | Value |
|---|---|
| **Test Accuracy** | **97.80%** |
| **Test Loss** | 0.0624 |
| **Total Parameters** | 1,331,666 (~5.08 MB) |
| **Non-Trainable Parameters** | 130 |
| **Epochs Trained** | 58 (of a 100-epoch ceiling; stopped by `EarlyStopping`) |
| **Best Epoch** | 53 (best validation accuracy *and* best validation loss) |
| **Best Validation Accuracy** | 97.75% |
| **Best Validation Loss** | 0.0701 |
| **Batch Size** | 64 |
| **Final Learning Rate** | 3.125 × 10⁻⁵ (reduced 5× from an initial 1 × 10⁻³) |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support | Accuracy |
|---|---:|---:|---:|---:|---:|
| Cargo | 98.34% | 97.77% | 98.06% | 6,199 | 97.77% |
| Passengership | 96.97% | 98.36% | 97.66% | 3,056 | 98.36% |
| Tanker | 97.97% | 96.90% | 97.43% | 4,426 | 96.90% |
| Tug | 96.85% | 99.30% | 98.06% | 1,581 | 99.30% |
| **Macro avg** | **97.53%** | **98.09%** | **97.80%** | 15,262 | - |
| **Weighted avg** | **97.81%** | **97.80%** | **97.80%** | 15,262 | - |

**Key observations:**

- **Every class scores above 96.9% on every metric** - there is no class the model is meaningfully failing on, which is the real bar for success on an imbalanced 4-class problem (a naive "always predict Cargo" baseline would only reach ~40.6% accuracy, the [Cargo class's share](#dataset) of the test set).
- **Tug (the rarest class, ~10% of the data) has the single highest recall/accuracy (99.30%)** of the four classes - strong evidence that the square-root-smoothed class weighting in [Cell 13](#cell-13-compute-smoothed-class-weights-and-train) successfully protected the minority class from being under-learned.
- **The tightest margins are between Cargo and Tanker** - both large-vessel classes with plausibly similar acoustic signatures - while Passengership and Tug, likely more acoustically distinctive, show slightly different precision/recall trade-off patterns.
- **Macro-average F1 (97.80%) and weighted-average F1 (97.80%) are nearly identical**, which is itself a good imbalance-robustness signal: when a model is quietly failing on minority classes, macro metrics (which weight every class equally) tend to fall noticeably *below* weighted metrics (which are dominated by the majority class) - here, they essentially agree.

---

## Best Practices and Design Decisions

### 1. Data Preprocessing
- **75ms non-overlapping segmentation** turns 63 source recordings into 76,311 independent training examples - the single biggest lever this pipeline pulls to get enough data for a Transformer-containing model to train well.
- **Mel-scale + dB compression** matches how the pipeline represents frequency and loudness to how the underlying acoustic energy is actually distributed and perceived, rather than working with raw linear-frequency, linear-power spectra.
- **Pre-emphasis filtering** boosts high-frequency content before spectral analysis, making harmonic detail more prominent in the resulting Mel-spectrogram.

### 2. Model Architecture
- **Hybrid DWS-CNN + Transformer** - convolution handles cheap local pattern extraction, attention handles global relationships between time-frequency regions, and only ~0.03% of the model's parameters are "spent" on the convolutional stage.
- **Learnable class token + learned positional embeddings**, ViT/BERT-style, rather than global average pooling or fixed sinusoidal position encodings - a natural fit given the very short (33-token), fixed-length sequences this architecture always processes.
- **Heavy, multi-point regularization** (`dropout_rate=0.3` in attention, MLP blocks, and the classification head, plus decoupled Adam weight decay) - appropriate given the small number of *source* recordings (63) behind an otherwise large segment count.

### 3. Training Strategy
- **Square-root-smoothed balanced class weighting** handles the dataset's ~4:1 class imbalance without the instability that can come from full inverse-frequency weighting.
- **A three-callback safety net** - checkpointing the best model, stopping early on stalled validation loss, and reducing the learning rate on plateaus - means the notebook can safely be pointed at `EPOCHS = 100` and trusted to stop at the right time on its own (it stopped at epoch 58 in the reference run).
- **Explicit, fixed random seeds** (`RANDOM_STATE=42` for the data split, `RANDOM_SEED=42` for NumPy/TensorFlow) make the overall pipeline close to reproducible end-to-end, GPU non-determinism notwithstanding.

### 4. Evaluation Methodology
- **Stratified splitting** at both split stages keeps class proportions consistent across train, validation, and test sets.
- **A genuinely held-out test set** - never touched by training, validation-based early stopping, or learning-rate scheduling - is only opened in Cells 14–16, giving an honest final performance read.
- **Reload-from-checkpoint before evaluating** ([Cell 14](#cell-14-evaluate-on-test-set)) verifies the saved model artifact matches what's being reported, not just whatever happens to be sitting in memory.

### 5. Memory and Storage Management
- **Delete-after-save discipline** - `X_all`/`y_all` (Cell 4) and the intermediate split arrays (Cell 5) are explicitly `del`eted from memory the moment they're no longer needed, rather than left to accumulate across a long-running session.
- **Raw audio stays off Drive; only processed features and results persist** - the ~500 MB Kaggle download lands in Colab's ephemeral `/content` space, while the much smaller `.npz`/`.pkl`/`.keras`/`.png` artifacts (well under 200 MB combined) are the only things written to the Drive-backed `SAVE_DIR`.

---

## Troubleshooting

### Common Issues and Solutions

**1. `cp: cannot stat '/content/kaggle.json'` / Kaggle download fails with a 401 error**
```
cp: cannot stat '/content/kaggle.json': No such file or directory
```
**Cause:** No Kaggle API token has been uploaded to `/content/kaggle.json` for this session, and none is cached from a previous run.
**Solution:**
- Go to your Kaggle account → **Settings → API → Create New Token** to download a `kaggle.json` file.
- In the Colab file browser (left sidebar), upload that file to `/content/kaggle.json` *before* re-running Cell 4.
- If the download step reports a `403`/`401` even with a fresh token, double-check you've accepted the dataset's terms on its Kaggle page at least once.

**2. `FileNotFoundError: Dataset zip file not found!`**
```
FileNotFoundError: Dataset zip file not found! Check if kaggle.json is uploaded.
```
**Cause:** This is Cell 4's own explicit guard rail - it means the `kaggle datasets download` command above it failed silently (shell commands via `!` don't stop notebook execution on their own).
**Solution:** Scroll up in that cell's output to see the actual Kaggle CLI error, which is almost always the missing-credentials issue in #1 above.

**3. Out-of-memory errors during preprocessing or training**
```
RuntimeError: CUDA out of memory (or a Colab runtime crash during Cell 4)
```
**Solutions:**
- Use a high-RAM Colab runtime for Cell 4's preprocessing pass (**Runtime → Change runtime type → High-RAM**), especially if you point this pipeline at a larger dataset than the 63-file DeepShip copy used here.
- Reduce `BATCH_SIZE` in Cell 13 (e.g., from 64 down to 32).
- Confirm GPU memory growth is actually enabled - check for the `✅ GPU memory growth enabled` line in Cell 7's output.

**4. `NameError` when running Cell 15 or Cell 16 in isolation**
```
NameError: name 'history' is not defined
NameError: name 'best_model' is not defined
```
**Cause:** These cells depend on variables produced by Cells 13 and 14 respectively, which don't persist across a runtime restart.
**Solution:** Re-run Cells 1, 6, 7, 8, 9, 10, 11, 12, 13, 14 in order (or simply "Run all") before jumping to Cell 15/16 in a fresh session.

**5. `Unknown layer: 'TransformerBlock'` (or `ClassTokenLayer` / `PositionalEmbedding`) when loading the saved model**
```
ValueError: Unknown layer: 'TransformerBlock'. Please ensure this object is passed to the `custom_objects` argument.
```
**Cause:** Attempting to load `dwstr_best_model.keras` without first defining the custom layer classes in the current session, or omitting one of them from the `custom_objects` dictionary.
**Solution:** Make sure Cells 9 and 10 (which define `TransformerBlock`, `ClassTokenLayer`, and `PositionalEmbedding`) have been executed in the current session before calling `keras.models.load_model(...)`, and that all three classes are included in `custom_objects`, exactly as Cell 14 does it.

**6. Drive mount timeout / stale paths**
```
FileNotFoundError: /content/drive/MyDrive/processed_data_DeepShip_DWSTr/...
```
**Solution:**
- Re-run Cell 1 to remount Google Drive.
- Confirm `SAVE_DIR` actually contains the expected files (`full_data_preprocessed.npz`, `train_data.npz`, etc.) via the Colab file browser.

**7. Validation accuracy stuck near 25% (random-guess level for 4 classes)**
**Check:**
- That `class_weight_dict` was actually built and passed into `model.fit()` (Cell 13's printed confirmation message).
- That `X_train`/`y_train` shapes look correct (`(N, 128, 4, 1)` and `(N,)` respectively) via Cell 6's sanity check.
- That the correct, freshly-downloaded `DATASET_ROOT_PATH` (from Cell 4) - not the placeholder one from Cell 1 - was actually used when Cell 4's preprocessing ran.

---

## Future Improvements

### Potential Enhancements

1. **Data Augmentation**
   - A SpecAugment-style layer (random frequency/time masking, applied only during training) is notably absent from this notebook - adding one between the DWS block and the patch embedding step would be a natural way to further reduce overfitting risk, especially given how few source recordings (63) back the dataset.
   - Mixup or waveform-level augmentation (time-stretching, pitch-shifting, background noise injection) before the Mel-spectrogram extraction step.

2. **Architecture Experiments**
   - Setting `key_dim` to the more conventional `projection_dim // num_heads` (16, here) in the `MultiHeadAttention` layer, to compare against the current, more expensive `key_dim=embed_dim` choice.
   - Increasing `num_transformer_blocks` beyond 6, now that there's headroom - this run stopped well short of overfitting (train/val metrics stayed close together through epoch 58).
   - Trying additional patch subdivision strategies for the time axis (e.g., a longer input window than 4 frames) to give the "patch" concept more to work with along time, not just frequency.

3. **Training Optimizations**
   - A learning-rate warmup phase before the main schedule, common practice for Transformer-containing models.
   - Mixed-precision training (`tf.keras.mixed_precision`) to speed up epochs on GPU with minimal accuracy impact.
   - Gradient clipping, as extra insurance against any attention-related instability if the model or dataset scale grows.

4. **Evaluation Enhancements**
   - Attention-weight visualization - since the class token attends over all 32 patches, plotting those attention weights back onto the Mel-spectrogram's frequency axis could show *which* frequency bands the model actually relies on per class.
   - t-SNE or UMAP visualization of the pre-classification-head embeddings, to visually inspect how well-separated the four classes are in learned feature space.
   - Cross-validation across multiple random seeds, to report performance as a mean ± standard deviation rather than a single run's numbers.

---

## Citation

If you use or build on this implementation, consider citing the original dataset paper this pipeline is built around:

```bibtex
@article{irfan2021deepship,
  title   = {DeepShip: An underwater acoustic benchmark dataset and a separable convolution based autoencoder for classification},
  author  = {Irfan, Muhammad and Jiangbin, Zheng and Ali, Shahid and Iqbal, Muhammad and Masood, Zafar and Hamid, Umar},
  journal = {Expert Systems with Applications},
  volume  = {183},
  pages   = {115270},
  year    = {2021},
  doi     = {10.1016/j.eswa.2021.115270}
}
```

And, for this specific notebook implementation:

```bibtex
@misc{deepship_dwstr_kaggle,
  title  = {DWSTr: Depthwise Separable Convolution and Transformer for DeepShip Audio Classification (Kaggle Edition)},
  author = {Your Name},
  year   = {2026},
  note   = {Notebook implementation combining a depthwise separable convolution block with a Vision-Transformer-style encoder for 4-class underwater ship-noise classification}
}
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details. The DeepShip dataset itself, as distributed via the Kaggle mirror this notebook downloads from, is also listed under an MIT license - always double-check the dataset's Kaggle page for the current license terms before redistributing it.

---

## Acknowledgments

- **Muhammad Irfan et al.**, for creating and publishing the original DeepShip dataset and benchmark ([Irfan et al., 2021](#citation)).
- The Kaggle dataset maintainer (`vasundharauppuluri`), for hosting an accessible, API-downloadable mirror of DeepShip.
- The broader **Vision Transformer / BERT** line of research, whose class-token-plus-positional-embedding pattern this architecture's Transformer half is directly built on.
- The **TensorFlow/Keras** team, for the deep learning framework this entire pipeline is implemented in.
- **Google Colab**, for the free GPU-backed notebook environment this was designed and run on.

---

**Last Updated:** September 2026
**Maintainer:** [Your Name]
**Notebook:** `DWSTr_Deepship_Kaggle.ipynb`
**Status:** Production Ready ✅ (97.80% test accuracy)
