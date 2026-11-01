# SignDetection

A webcam app for recognizing 100 isolated ASL signs with `ISLR_Transformer_VCN`.
After each sign, the app displays five candidate words. Hold your index fingertip
inside a candidate box for one second to add that word to the transcript.

This repository contains the app and its VCN training code. Model weights,
MediaPipe assets, datasets, and reference videos are shared separately.

## Requirements

- **Python 3.11**, preferably installed through Conda.
- A working webcam. The app uses camera index `0` by default.
- The three model files listed in step 2.

A GPU is optional: the app uses CUDA when available and otherwise runs on CPU.
The examples below use PowerShell. Run commands from the project root, where
`environment.yml` and `requirements.txt` are located.

## 1. Install the environment

Clone this repository or download and extract its ZIP. Open a terminal in the
project folder, then run:

```powershell
conda env create -f environment.yml
conda activate signdetection-share
```

Alternatively, create the environment manually:

```powershell
conda create -n signdetection-share python=3.11 pip -y
conda activate signdetection-share
python -m pip install -r requirements.txt
```

Check the installation:

```powershell
python --version
python -m pip check
```

Python should report `3.11.x`. Install only `opencv-contrib-python`, which
provides `cv2`; do not install a second OpenCV package alongside it.

## 2. Add the model files

Ask the project maintainer for these three files and place them at the following
paths relative to the project root:

```text
SignDetection-share/
|-- app/
|   `-- labels/
|       `-- gloss2idx.json                # Included in the repository
|-- outputs/
|   `-- models/
|       `-- landmarks_100_v1.pt          # Obtain separately
|-- pretrained/
|   |-- hand_landmarker.task            # Obtain separately
|   `-- pose_landmarker_heavy.task      # Obtain separately
|-- config.py
|-- environment.yml
|-- requirements.txt
`-- train.py
```

Create the `pretrained` and `outputs/models` folders if necessary. The checkpoint
must match the included `app/labels/gloss2idx.json` mapping.

The binary files are excluded by `.gitignore`: cloning the repository does not
download them. Training data and reference videos are not required for webcam
recognition.

## 3. Start the app

From the project root:

```powershell
conda activate signdetection-share
python -m app.run
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. Keep the
terminal running; press **Ctrl+C** there to stop the app.

The Python process captures the webcam. Allow desktop apps to access your camera
if your operating system requests permission.

## 4. Recognize and select a word

1. Face the camera with your shoulders and hands visible.
2. Bring at least one wrist into the central start zone, with both shoulders
   inside the zone. Stay in the zone for at least **0.5 seconds** while signing.
   Short accidental entries are discarded without a prediction.
3. Move your hands out of the zone and wait briefly for the five candidate words
   and their model probabilities to appear.
4. Move your **index fingertip** into a candidate box. Hold for **one second**,
   until the progress bar fills. The chosen word is added to the transcript.
5. Select **CANCEL / SIGN AGAIN** to discard the candidates and try again.
6. Move your hands out of the start zone before beginning the next sign.

Use one hand for selection. If both hands are visible, the app follows the right
hand. Leaving a box, losing hand detection, or switching hands resets the timer.
Recording and the camera's delete controls are paused during candidate selection.

Outside candidate selection, hold a wrist in **BACK SPACE** or **CLEAR** to delete
the last word or clear the transcript. The transcript panel also has copy and
clear buttons.

## Troubleshooting

### NumPy reports `metadata-generation-failed`

Check `python --version`. NumPy `1.26.4` does not support Python 3.13. If you
created the environment with that version, switch to Python 3.11:

```powershell
conda install -n signdetection-share python=3.11 pip -y
conda deactivate
conda activate signdetection-share
python -m pip install -r requirements.txt
```

### A `.task` file or checkpoint cannot be opened

Check the paths in step 2, especially `pretrained/hand_landmarker.task` and
`pretrained/pose_landmarker_heavy.task`. Obtain missing files from the maintainer;
reinstalling Python packages does not install these assets.

### The camera feed is empty

Close other programs using the webcam, allow desktop camera access, and restart
the app. If the webcam is not camera `0`, change the `cv.VideoCapture` index in
`app/run.py` to the appropriate device index.

### The Gloss Library is empty

Reference videos are optional. To enable them, provide the following separately:

```text
datasets/annotations/WLASL100/train.json
datasets/annotations/WLASL100/val.json
datasets/annotations/WLASL100/test.json
datasets/raw/WLASL/videos/<video_id>.mp4
```

Only signs with an existing reference video are listed. Webcam recognition works
without this library.

## Optional: use model files stored elsewhere

Set environment variables before starting the app. For example:

```powershell
$env:SLR_MODEL_PATH = "C:/private/models/landmarks_100_v1.pt"
$env:SLR_PRETRAINED_DIR = "C:/private/mediapipe"
python -m app.run
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `SLR_MODEL_PATH` | `outputs/models/landmarks_100_v1.pt` | Recognition checkpoint file |
| `SLR_PRETRAINED_DIR` | `pretrained/` | Directory containing both MediaPipe `.task` files |
| `SLR_LABEL_DIR` | `app/labels/` | Directory containing the checkpoint's `gloss2idx.json` |
| `SLR_DATA_ROOT` | `datasets/` | Root for training data and optional reference videos |

Default paths are resolved relative to the project root.

## Optional: train the model

`train.py` contains `ISLR_Transformer_VCN`, the required graph blocks, the landmark
dataset loader, and the training/validation loop. Prepare the data separately:

```text
datasets/annotations/WLASL100/gloss2idx.json
datasets/annotations/WLASL100/train.json
datasets/annotations/WLASL100/val.json
datasets/annotations/WLASL100/test.json
datasets/processed/wlasl_features_0_6/<video_id>/pose.npy
datasets/processed/wlasl_features_0_6/<video_id>/left_hand.npy
datasets/processed/wlasl_features_0_6/<video_id>/right_hand.npy
```

Each split JSON is a list of objects containing `video_id` and `gloss`. Pose arrays
have shape `(T, 33, 3)`; each hand array has shape `(T, 21, 3)`. The loader uses XY
coordinates and concatenates `[shape, average, position]` features into `(T, 306)`.

```powershell
python train.py --epochs 100 --batch_size 8 --val_mode val
```

Use `val` for model selection and reserve `test` for final evaluation. The CLI
default is `--val_mode test`, preserved from the original training configuration,
so pass `--val_mode val` explicitly as shown above.

See `python train.py --help` for `--data_path`, `--label_path`, `--output`, and other
options. When training a different label set, use its matching label mapping in
the app as well.

## Repository contents

- `app/`: webcam app, detectors, feature preparation, templates, and styles.
- `train.py`: VCN architecture and training code.
- `config.py`: shared model, path, and interaction settings.
- `environment.yml` and `requirements.txt`: environment setup.

Model weights, datasets, MediaPipe assets, videos, notebooks, logs, and Python
caches are excluded by `.gitignore`. Share the required model files separately
with team members.
