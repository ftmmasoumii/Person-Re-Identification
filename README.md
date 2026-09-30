# Person Re-Identification (Face-Primary)

A real-time webcam system that detects people, tracks them, and recognizes **who** each person is using their **face** — so changing clothes never changes a registered person's identity.

**Pipeline:** YOLOv8-Seg (detection + masks) → ArcFace via insightface (identity) → OSNet + HSV/LAB colour (auxiliary, anonymous bookkeeping only)

---

## How It Works

Earlier versions identified people by body appearance and clothing colour, which broke whenever someone changed clothes. This version changes the architecture:

- **Face (ArcFace)** is the *only* signal allowed to decide which registered `Person_XXX` a track is.
- **OSNet + colour descriptor** are only used for the anonymous `Unknown_XXX` path, for people whose face has never been visible. They are never compared against a registered person to decide identity.
- If a known person's face is temporarily hidden (turned away, occluded), nothing changes: **identity stickiness** keeps the existing identity. There is no body/colour "double check" fallback.

### Main components

| Component | Description |
|---|---|
| Detection | YOLOv8-Seg finds people and produces segmentation masks |
| Tracking | Purely geometric: IoU, normalized centre distance, motion prediction, and a stable torso anchor from the mask. Appearance is never used. |
| Face detection | insightface `buffalo_l`, run every 0.15 s and cached between frames for speed |
| Identification | Every 1.0 s, a track's face is compared with registered `Person_` profiles |
| Stickiness | Identity switch needs 3 consistent cycles; 5 weak matches in a row clears the identity |
| Unknown handling | Face-based `Unknown_XXX` when a face is visible, body/colour-based when it never is (confirmed after 4 consistent cycles) |
| Database | One `.npy` file per identity in `database/` |

---

## Requirements

- Python 3.8+
- A webcam
- **Windows** (the code uses `cv2.CAP_DSHOW`; on Linux/macOS change it to `cv2.VideoCapture(CAMERA_INDEX)`)
- GPU optional but recommended

### Install

```bash
pip install ultralytics torch opencv-python numpy insightface onnxruntime
pip install torchreid
```

For GPU, install `onnxruntime-gpu` instead of `onnxruntime`.

On first run, these models download automatically (internet needed once):

- `yolov8n-seg.pt` (YOLO)
- `buffalo_l` (~300 MB, insightface)
- OSNet pretrained weights (torchreid)

---

## Usage

```bash
python main.py
```

(Replace `main.py` with your script's file name.)

### Controls

| Key | Action |
|---|---|
| `S` | Register a new person. Locks onto the largest person in view and collects 30 face samples. Face must be visible. |
| `A` | Add more samples to the currently recognized `Person_` (locked to that track, face required) |
| `C` | Cancel registration / add mode |
| `Q` | Quit |

**Tip for registration:** look at the camera and slowly turn your head left, right, up and down for a more robust profile.

### On-screen display

Each person gets a green box with a label like:

```
Person_001 | T:3 [F]
```

- `Person_001` / `Unknown_002`: identity (or `Unknown?` if not yet confirmed)
- `T:3`: track ID
- `[F]`: a face was seen very recently
- `TOO SMALL FOR REID`: person is too small in the frame to process

The bottom of the window shows FPS, track count, database size, and whether the face model is active.

---

## Database

Stored in `database/` as `Person_XXX.npy` and `Unknown_XXX.npy`. Each file is a pickled dict containing:

| Key | Content |
|---|---|
| `embedding` | OSNet body embedding (mean) |
| `color_profile` | HSV/LAB colour descriptor (mean) |
| `face_profile` | ArcFace embedding (mean) |
| `samples`, `color_samples`, `face_samples` | Individual samples (capped at 50–100) |

Known people automatically gain fresh samples over time while recognized.

> **Security:** files are loaded with `allow_pickle=True`. Only place `.npy` files you trust in `database/`.

---

## Configuration

Key parameters at the top of the script:

| Parameter | Default | Meaning |
|---|---|---|
| `YOLO_CONF` | 0.50 | Person detection confidence |
| `FACE_MIN_DET_SCORE` | 0.55 | Minimum face detection score |
| `FACE_CONFIDENT_THRESHOLD` | 0.50 | Min ArcFace similarity to accept an identity |
| `FACE_MIN_DIFFERENCE` | 0.05 | Min gap between best and second-best match |
| `UNKNOWN_MATCH_THRESHOLD_FACE` | 0.42 | Grouping threshold for unregistered faces |
| `REGISTER_CONSISTENCY_THRESHOLD` | 0.55 | Rejects inconsistent samples during registration |
| `FACE_STALENESS_SECONDS` | 1.5 | How long a last-seen face stays usable |
| `IDENTIFICATION_INTERVAL` | 1.0 | Seconds between identification runs |
| `NUM_SAMPLES` | 30 | Face samples collected on registration |
| `TRACK_TIMEOUT` | 3.5 | Seconds before a lost track is removed |

Thresholds are starting points. Tune them for your camera and lighting using the similarity scores.

---

## Known Limitations

- If insightface fails to load, **no `Person_` can be registered or recognized** (by design).
- Persons registered without a face profile (e.g. from an older version) are never identified until re-registered with `S`.
- Faceless `Unknown_` tracking still relies on clothing, so it can split one person into several `Unknown_` entries after a clothing change.
- Small/distant people (height below 70 px) are not processed for ReID.

---

## Privacy Notice

This system stores **biometric face data**. Using it on other people (employees, customers, visitors) generally requires their informed consent and compliance with local privacy laws (e.g. GDPR). Store and protect the `database/` folder accordingly.

---

## Tech Stack

[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) · [insightface](https://github.com/deepinsight/insightface) · [torchreid](https://github.com/KaiyangZhou/deep-person-reid) · OpenCV · PyTorch
