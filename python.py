# ============================================================
# Person Re-Identification
# YOLOv8-Seg (detection/segmentation) + ArcFace (identity) + OSNet (aux)
#
# ------------------------------------------------------------
# ARCHITECTURE CHANGE (per explicit request): clothing must NEVER be
# able to change who the system thinks a registered person is.
#
#   - FACE (ArcFace via insightface) is now the ONLY signal allowed to
#     decide "which registered Person_ is this track?". It is
#     clothes-invariant by nature.
#   - OSNet (body appearance) + the HSV/LAB colour descriptor are kept,
#     but are now ONLY used for the anonymous "Unknown_" bookkeeping
#     path - i.e. when a track has NEVER shown a face at all, we still
#     don't want to lose track of the fact that *someone* is there.
#     They are never compared against a registered Person_'s profile
#     to decide identity, so a colour/clothing change can no longer
#     cause a mis-identification of a known person.
#   - When a track currently HAS an identity (because its face matched
#     someone earlier) but the face is temporarily not visible (person
#     turned around, walked behind something, etc.), the code simply
#     does nothing for that cycle: identity stickiness keeps the
#     existing identity. It does NOT fall back to body/colour to
#     "double check" - that fallback was exactly the source of the
#     clothes-swap bug this rewrite fixes.
#
# Features:
#   - Person detection with YOLOv8 Segmentation
#   - Real person mask used as OSNet input (background removed)
#   - Face detection + ArcFace embedding (insightface, buffalo_l)
#   - IoU + normalized center-distance + stable body-anchor tracking
#   - Identity Stickiness (face-driven)
#   - Stable Unknown confirmation (face-driven when a face is seen,
#     body/colour-driven as a fallback for faceless tracks)
#   - Manual Person registration with S (locked to one track, requires
#     a real face - clothing samples are just extra bookkeeping)
#   - Add samples with A (locked to one track, requires a real face)
#   - No Figure / no Matplotlib window
#   - Everything shown in ONE camera window
# ============================================================

import os
import cv2
import time
import numpy as np
import torch

from ultralytics import YOLO
import torchreid


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_PATH = "database"

YOLO_MODEL = "yolov8n-seg.pt"
YOLO_CONF = 0.50

OSNET_MODEL = "osnet_x1_0"

CAMERA_INDEX = 0

# ------------------------------------------------------------
# ReID input (OSNet - now only used for the anonymous/Unknown path)
# ------------------------------------------------------------

REID_WIDTH = 128
REID_HEIGHT = 256

# ImageNet normalization
MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)

# ------------------------------------------------------------
# Face recognition (ArcFace via insightface) - THE identity signal
#
# pip install insightface onnxruntime   (or onnxruntime-gpu)
# The "buffalo_l" model auto-downloads (~300MB) the first time this
# runs, so an internet connection is needed once.
# ------------------------------------------------------------

USE_FACE_RECOGNITION = True

FACE_MODEL_NAME = "buffalo_l"
FACE_DET_SIZE = (640, 640)
FACE_MIN_DET_SCORE = 0.55

# Face detection (RetinaFace inside insightface) is the heaviest part
# of the pipeline. Faces barely move within ~150ms, so we reuse the
# last result for frames in between instead of re-running it every
# single frame.
FACE_DETECT_INTERVAL = 0.15

# A track may keep using its last-seen face embedding for identity
# decisions for up to this many seconds after the face actually
# disappeared from view (person turned briefly, walked behind an
# object, etc.) before we stop trusting it as "fresh".
FACE_STALENESS_SECONDS = 1.5

# ArcFace cosine similarities are numerically smaller than the OSNet
# scores this project used before, so these thresholds are on a
# different scale. They are reasonable starting points - the on-screen
# score next to each box will tell you whether they need tuning for
# your camera/lighting.
FACE_CONFIDENT_THRESHOLD = 0.50
FACE_MIN_DIFFERENCE = 0.05

# Threshold used only to group together repeated sightings of the same
# UNIDENTIFIED face into one Unknown_XXX entry (not for matching a
# registered Person_).
UNKNOWN_MATCH_THRESHOLD_FACE = 0.42

# ------------------------------------------------------------
# Anonymous/"Unknown" bookkeeping via body+colour
#
# This is ONLY a fallback for tracks that never show a face at all
# (e.g. someone who keeps their back to the camera the whole time).
# It never touches a registered Person_'s identity.
# ------------------------------------------------------------

UNKNOWN_MATCH_THRESHOLD = 0.68
UNKNOWN_PROFILE_UPDATE_THRESHOLD = 0.72

# Same idea, but for the OSNet/colour samples kept alongside a
# registered Person_ profile (bookkeeping only - these are never used
# to decide who someone is, but a low-quality frame - motion blur,
# bad angle - should still not be allowed to pollute them).
PERSON_PROFILE_UPDATE_THRESHOLD = 0.72

# Second defense layer for manual registration / add-samples, on top
# of track_id locking: reject a face sample if it doesn't look
# consistent with what has been collected so far in this session
# (registration) or with the existing stored face profile
# (add-samples).
REGISTER_CONSISTENCY_THRESHOLD = 0.55

# ------------------------------------------------------------
# Tracking
# ------------------------------------------------------------

TRACK_IOU_THRESHOLD = 0.20

TRACK_CENTER_DIST_MULTIPLIER = 2.5
TRACK_PREDICTION_ERROR_MULTIPLIER = 2.0
TRACK_MASK_ANCHOR_WEIGHT = 0.35

TRACK_ANCHOR_REJECT_THRESHOLD = 1.20

TRACK_TIMEOUT = 3.5

MIN_PERSON_HEIGHT = 70

# ------------------------------------------------------------
# Identity stability
# ------------------------------------------------------------

CONFIRM_UNKNOWN_FRAMES = 4

MAX_WEAK_ID_FRAMES = 5

IDENTITY_SWITCH_CONFIRM_FRAMES = 3

# ------------------------------------------------------------
# History
# ------------------------------------------------------------

HISTORY_SIZE = 8

# ------------------------------------------------------------
# Database samples
# ------------------------------------------------------------

NUM_SAMPLES = 30

SAVE_INTERVAL = 0.70

DUPLICATE_SIM_THRESHOLD = 0.995
REGISTER_DUPLICATE_SIM_THRESHOLD = 0.985
REGISTER_MIN_INTERVAL = 0.12

# ------------------------------------------------------------
# Identification
# ------------------------------------------------------------

IDENTIFICATION_INTERVAL = 1.0


# ============================================================
# GLOBAL VARIABLES
# ============================================================

tracks = {}

next_track_id = 1

register_mode = False
register_samples = []
register_color_samples = []
register_face_samples = []

register_target_name = None

add_mode = False
add_samples = []
add_color_samples = []
add_face_samples = []
add_target_name = None

# Track IDs used to lock manual registration/add-sample collection
register_target_track_id = None
register_last_sample_time = 0.0
add_target_track_id = None

status_message = ""
status_message_until = 0

last_frame_time = time.time()

last_faces = []
last_face_detect_time = 0.0


# ============================================================
# DATABASE
# ============================================================

os.makedirs(DATABASE_PATH, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("PERSON RE-ID SYSTEM (face-primary)")
print("=" * 60)

print("Device:", device)


# ============================================================
# LOAD YOLO
# ============================================================

print("\nLoading YOLO...")

yolo = YOLO(YOLO_MODEL)

print("YOLO loaded.")


# ============================================================
# LOAD OSNET
# ============================================================

print("\nLoading OSNet (auxiliary/body only)...")

reid_model = torchreid.models.build_model(
    name=OSNET_MODEL,
    num_classes=1000,
    pretrained=True
)

reid_model = reid_model.to(device)
reid_model.eval()

print("OSNet loaded.")


# ============================================================
# LOAD FACE RECOGNITION MODEL (ArcFace via insightface)
#
# This is what lets the system recognise someone by their face /
# facial structure instead of clothing, so the same person stays
# recognised for as long as they work here, no matter what they wear.
# ============================================================

face_app = None

if USE_FACE_RECOGNITION:

    try:

        from insightface.app import FaceAnalysis

        print("\nLoading face recognition model (buffalo_l)...")

        if device.type == "cuda":
            face_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            face_ctx_id = 0
        else:
            face_providers = ["CPUExecutionProvider"]
            face_ctx_id = -1

        face_app = FaceAnalysis(
            name=FACE_MODEL_NAME,
            providers=face_providers
        )

        face_app.prepare(
            ctx_id=face_ctx_id,
            det_size=FACE_DET_SIZE
        )

        print("Face recognition model loaded.")

    except Exception as e:

        print(
            "\n[WARNING] Could not load face recognition "
            "(insightface). Person_ identification will NOT work "
            "without it (by design - body/clothing alone is no "
            "longer allowed to decide a registered identity).\n"
            "Install with: pip install insightface onnxruntime\n"
            "Error:", repr(e)
        )

        face_app = None


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def get_person_files():

    files = []

    if not os.path.exists(DATABASE_PATH):
        return files

    for filename in os.listdir(DATABASE_PATH):

        if filename.lower().endswith(".npy"):

            if (
                filename.startswith("Person_")
                or filename.startswith("Unknown_")
            ):
                files.append(
                    os.path.join(DATABASE_PATH, filename)
                )

    return sorted(files)


def get_next_person_id():

    max_id = 0

    for path in get_person_files():

        name = os.path.basename(path)

        if name.startswith("Person_"):

            try:
                number = int(
                    name.replace("Person_", "")
                        .replace(".npy", "")
                )

                max_id = max(max_id, number)

            except:
                pass

    return max_id + 1


def get_next_unknown_id():

    max_id = 0

    for path in get_person_files():

        name = os.path.basename(path)

        if name.startswith("Unknown_"):

            try:
                number = int(
                    name.replace("Unknown_", "")
                        .replace(".npy", "")
                )

                max_id = max(max_id, number)

            except:
                pass

    return max_id + 1


def save_profile(
    name,
    embedding,
    color_descriptor,
    samples=None,
    color_samples=None,
    face_descriptor=None,
    face_samples=None
):
    path = os.path.join(DATABASE_PATH, name + ".npy")

    if embedding is None:
        return False, "embedding is None"

    embedding = normalize_vector(embedding)
    if embedding is None:
        return False, "invalid embedding (NaN/Inf/empty)"

    data = {
        "name": name,
        "embedding": embedding
    }

    # None means no colour feature. Never serialize None with np.asarray().
    if color_descriptor is not None:
        color_descriptor = normalize_vector(color_descriptor)
        if color_descriptor is not None:
            data["color_profile"] = color_descriptor

    # None means no face feature captured (should be rare for Person_,
    # but Unknown_ entries created purely from a faceless body sighting
    # legitimately have none).
    if face_descriptor is not None:
        face_descriptor = normalize_vector(face_descriptor)
        if face_descriptor is not None:
            data["face_profile"] = face_descriptor

    if samples is not None:
        valid_samples = []
        for sample in samples:
            sample = normalize_vector(sample)
            if sample is not None:
                valid_samples.append(sample)
        if valid_samples:
            data["samples"] = np.asarray(valid_samples, dtype=np.float32)

    if color_samples is not None:
        valid_color_samples = []
        for sample in color_samples:
            sample = normalize_vector(sample)
            if sample is not None:
                valid_color_samples.append(sample)
        if valid_color_samples:
            data["color_samples"] = np.asarray(valid_color_samples, dtype=np.float32)

    if face_samples is not None:
        valid_face_samples = []
        for sample in face_samples:
            sample = normalize_vector(sample)
            if sample is not None:
                valid_face_samples.append(sample)
        if valid_face_samples:
            data["face_samples"] = np.asarray(valid_face_samples, dtype=np.float32)

    try:
        np.save(path, data, allow_pickle=True)
        return True, path
    except Exception as e:
        print("SAVE ERROR:", repr(e))
        return False, str(e)


def load_database():

    database = {}

    for path in get_person_files():
        try:
            data = np.load(path, allow_pickle=True).item()

            name = data.get(
                "name",
                os.path.basename(path).replace(".npy", "")
            )

            embedding = np.asarray(
                data["embedding"], dtype=np.float32
            )
            if embedding.size == 0 or not np.all(np.isfinite(embedding)):
                print("INVALID EMBEDDING SKIPPED:", path)
                continue

            color_profile = data.get("color_profile", None)
            if color_profile is not None:
                color_profile = np.asarray(
                    color_profile, dtype=np.float32
                )
                if (
                    color_profile.size == 0
                    or not np.all(np.isfinite(color_profile))
                ):
                    print("INVALID COLOR PROFILE IGNORED:", path)
                    color_profile = None

            face_profile = data.get("face_profile", None)
            if face_profile is not None:
                face_profile = np.asarray(
                    face_profile, dtype=np.float32
                )
                if (
                    face_profile.size == 0
                    or not np.all(np.isfinite(face_profile))
                ):
                    print("INVALID FACE PROFILE IGNORED:", path)
                    face_profile = None

            samples = data.get("samples", None)
            if samples is not None:
                samples = np.asarray(samples, dtype=np.float32)
                if samples.ndim == 1:
                    samples = samples.reshape(1, -1)
                if samples.ndim != 2:
                    samples = None
                else:
                    valid = [
                        x for x in samples
                        if x.size > 0 and np.all(np.isfinite(x))
                    ]
                    samples = (
                        np.asarray(valid, dtype=np.float32)
                        if valid else None
                    )

            color_samples = data.get("color_samples", None)
            if color_samples is not None:
                color_samples = np.asarray(
                    color_samples, dtype=np.float32
                )
                if color_samples.ndim == 1:
                    color_samples = color_samples.reshape(1, -1)
                if color_samples.ndim != 2:
                    color_samples = None
                else:
                    valid_color = [
                        x for x in color_samples
                        if x.size > 0 and np.all(np.isfinite(x))
                    ]
                    color_samples = (
                        np.asarray(valid_color, dtype=np.float32)
                        if valid_color else None
                    )

            face_samples = data.get("face_samples", None)
            if face_samples is not None:
                face_samples = np.asarray(
                    face_samples, dtype=np.float32
                )
                if face_samples.ndim == 1:
                    face_samples = face_samples.reshape(1, -1)
                if face_samples.ndim != 2:
                    face_samples = None
                else:
                    valid_face = [
                        x for x in face_samples
                        if x.size > 0 and np.all(np.isfinite(x))
                    ]
                    face_samples = (
                        np.asarray(valid_face, dtype=np.float32)
                        if valid_face else None
                    )

            if name.startswith("Person_") and face_profile is None:
                print(
                    f"[WARNING] {name} has no face profile - it will "
                    f"NEVER be identified until you re-register it "
                    f"with a visible face (S)."
                )

            database[name] = {
                "embedding": embedding,
                "color_profile": color_profile,
                "samples": samples,
                "color_samples": color_samples,
                "face_profile": face_profile,
                "face_samples": face_samples
            }

        except Exception as e:
            print("Could not load:", path, repr(e))

    return database


database = load_database()

print(
    "\nLoaded identities:",
    len(database)
)

for name in sorted(database.keys()):
    print("  -", name)


# ============================================================
# VECTOR NORMALIZATION
# ============================================================

def normalize_vector(vector):
    if vector is None:
        return None
    vector = np.asarray(vector, dtype=np.float32)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        return None
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1e-12:
        return None
    return vector / norm


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):
    a = normalize_vector(a)
    b = normalize_vector(b)
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    value = float(np.dot(a, b))
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, -1.0, 1.0))


# ============================================================
# LETTERBOX
# ============================================================

def letterbox_resize(
    image,
    target_width,
    target_height
):

    h, w = image.shape[:2]

    scale = min(
        target_width / w,
        target_height / h
    )

    new_w = max(
        1,
        int(w * scale)
    )

    new_h = max(
        1,
        int(h * scale)
    )

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    canvas = np.zeros(
        (
            target_height,
            target_width,
            3
        ),
        dtype=np.uint8
    )

    x = (
        target_width - new_w
    ) // 2

    y = (
        target_height - new_h
    ) // 2

    canvas[
        y:y + new_h,
        x:x + new_w
    ] = resized

    return canvas


# ============================================================
# CREATE OSNET INPUT (auxiliary/body only - not used for identity)
#
# Background pixels are filled with ImageNet mean colour.
# ============================================================

def create_reid_input(
    frame,
    bbox,
    mask
):

    x1, y1, x2, y2 = bbox

    h, w = frame.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))

    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[
        y1:y2,
        x1:x2
    ].copy()

    if crop.size == 0:
        return None

    crop_h, crop_w = crop.shape[:2]

    if mask is None:

        person_mask = np.ones(
            (crop_h, crop_w),
            dtype=np.uint8
        ) * 255

    else:

        person_mask = mask[
            y1:y2,
            x1:x2
        ]

        if person_mask.shape != (
            crop_h,
            crop_w
        ):

            person_mask = cv2.resize(
                person_mask,
                (crop_w, crop_h),
                interpolation=cv2.INTER_NEAREST
            )

    person_mask = (
        person_mask > 127
    ).astype(np.uint8)

    mean_bgr = np.array(
        [
            MEAN[2] * 255,
            MEAN[1] * 255,
            MEAN[0] * 255
        ],
        dtype=np.uint8
    )

    background = np.zeros_like(crop)

    background[:] = mean_bgr

    masked = np.where(
        person_mask[:, :, None] > 0,
        crop,
        background
    )

    prepared = letterbox_resize(
        masked,
        REID_WIDTH,
        REID_HEIGHT
    )

    return prepared


# ============================================================
# COLOR DESCRIPTOR (auxiliary/body only - not used for identity)
# ============================================================

def extract_color_descriptor(
    frame,
    bbox,
    mask
):

    x1, y1, x2, y2 = bbox

    crop = frame[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        return None

    crop_h, crop_w = crop.shape[:2]

    if mask is not None:

        m = mask[
            y1:y2,
            x1:x2
        ]

        if m.shape != (
            crop_h,
            crop_w
        ):

            m = cv2.resize(
                m,
                (crop_w, crop_h),
                interpolation=cv2.INTER_NEAREST
            )

        m = (
            m > 127
        ).astype(np.uint8)

    else:

        m = np.ones(
            (crop_h, crop_w),
            dtype=np.uint8
        )

    yy = np.linspace(
        0,
        1,
        crop_h
    )

    vertical_weight = (
        0.60 +
        0.80 * np.exp(
            -((yy - 0.58) ** 2) / 0.10
        )
    )

    weights = (
        m.astype(np.float32) *
        vertical_weight[:, None]
    )

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV
    )

    lab = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2LAB
    )

    valid = weights > 0

    if not np.any(valid):
        return None

    pixels_hsv = hsv[valid]
    pixels_lab = lab[valid]
    pixel_weights = weights[valid]

    h_hist = np.histogram(
        pixels_hsv[:, 0],
        bins=18,
        range=(0, 180),
        weights=pixel_weights
    )[0]

    s_hist = np.histogram(
        pixels_hsv[:, 1],
        bins=12,
        range=(0, 256),
        weights=pixel_weights
    )[0]

    v_hist = np.histogram(
        pixels_hsv[:, 2],
        bins=12,
        range=(0, 256),
        weights=pixel_weights
    )[0]

    if np.sum(h_hist) > 0:
        h_hist /= np.sum(h_hist)

    if np.sum(s_hist) > 0:
        s_hist /= np.sum(s_hist)

    if np.sum(v_hist) > 0:
        v_hist /= np.sum(v_hist)

    wsum = np.sum(pixel_weights)

    lab_mean = np.sum(
        pixels_lab *
        pixel_weights[:, None],
        axis=0
    ) / max(wsum, 1e-6)

    lab_std = np.sqrt(
        np.sum(
            (
                pixels_lab -
                lab_mean
            ) ** 2 *
            pixel_weights[:, None],
            axis=0
        ) /
        max(wsum, 1e-6)
    )

    descriptor = np.concatenate(
        [
            h_hist,
            s_hist,
            v_hist,
            lab_mean / 255.0,
            lab_std / 255.0
        ]
    )

    return normalize_vector(
        descriptor
    )


# ============================================================
# COLOR SIMILARITY
# ============================================================

def color_similarity(a, b):

    if a is None or b is None:
        return 0.0

    return cosine_similarity(a, b)


# ============================================================
# OSNET EMBEDDING (auxiliary/body only)
# ============================================================

@torch.no_grad()
def extract_embedding(
    prepared_image
):

    if prepared_image is None:
        return None

    image_rgb = cv2.cvtColor(
        prepared_image,
        cv2.COLOR_BGR2RGB
    )

    image_float = (
        image_rgb.astype(
            np.float32
        ) / 255.0
    )

    image_float = (
        image_float - MEAN
    ) / STD

    tensor = torch.from_numpy(
        image_float
    ).permute(
        2,
        0,
        1
    ).unsqueeze(0)

    tensor = tensor.to(device)

    features = reid_model(
        tensor
    )

    if isinstance(features, tuple):
        features = features[0]

    features = features.detach().cpu().numpy()

    features = features.reshape(
        -1
    )

    return normalize_vector(
        features
    )


# ============================================================
# FACE DETECTION (ArcFace via insightface) - primary identity signal
# ============================================================

def detect_faces(frame, now):

    global last_faces, last_face_detect_time

    if face_app is None:
        return []

    if now - last_face_detect_time < FACE_DETECT_INTERVAL:
        return last_faces

    try:
        raw_faces = face_app.get(frame)
    except Exception as e:
        print("Face detection error:", repr(e))
        last_face_detect_time = now
        last_faces = []
        return last_faces

    results = []

    h, w = frame.shape[:2]

    for f in raw_faces:

        if f.det_score < FACE_MIN_DET_SCORE:
            continue

        embedding = normalize_vector(f.embedding)

        if embedding is None:
            continue

        x1, y1, x2, y2 = f.bbox

        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w))
        y2 = max(0, min(int(y2), h))

        results.append({
            "bbox": (x1, y1, x2, y2),
            "embedding": embedding,
            "score": float(f.det_score)
        })

    last_faces = results
    last_face_detect_time = now

    return results


def match_face_to_person(person_bbox, faces):

    px1, py1, px2, py2 = person_bbox

    # A face belongs to the head, i.e. the upper portion of the
    # person's bounding box.
    head_limit = py1 + 0.65 * (py2 - py1)

    best = None
    best_score = -1.0

    for f in faces:

        fx1, fy1, fx2, fy2 = f["bbox"]

        fcx = (fx1 + fx2) / 2.0
        fcy = (fy1 + fy2) / 2.0

        if not (px1 <= fcx <= px2 and py1 <= fcy <= py2):
            continue

        if fcy > head_limit:
            continue

        if f["score"] > best_score:
            best_score = f["score"]
            best = f

    return best


# ============================================================
# IOU
# ============================================================

def calculate_iou(
    box_a,
    box_b
):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(
        ax1,
        bx1
    )

    inter_y1 = max(
        ay1,
        by1
    )

    inter_x2 = min(
        ax2,
        bx2
    )

    inter_y2 = min(
        ay2,
        by2
    )

    inter_w = max(
        0,
        inter_x2 - inter_x1
    )

    inter_h = max(
        0,
        inter_y2 - inter_y1
    )

    intersection = (
        inter_w *
        inter_h
    )

    area_a = max(
        0,
        ax2 - ax1
    ) * max(
        0,
        ay2 - ay1
    )

    area_b = max(
        0,
        bx2 - bx1
    ) * max(
        0,
        by2 - by1
    )

    union = (
        area_a +
        area_b -
        intersection
    )

    if union <= 0:
        return 0.0

    return (
        intersection /
        union
    )


# ============================================================
# CENTER DISTANCE
# ============================================================

def center_of_box(box):

    x1, y1, x2, y2 = box

    return np.array(
        [
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0
        ],
        dtype=np.float32
    )


def normalized_center_distance(
    box_a,
    box_b
):

    ca = center_of_box(box_a)
    cb = center_of_box(box_b)

    distance = np.linalg.norm(
        ca - cb
    )

    aw = max(
        1,
        box_a[2] - box_a[0]
    )

    ah = max(
        1,
        box_a[3] - box_a[1]
    )

    bw = max(
        1,
        box_b[2] - box_b[0]
    )

    bh = max(
        1,
        box_b[3] - box_b[1]
    )

    diagonal = (
        np.sqrt(
            aw ** 2 +
            ah ** 2
        ) +
        np.sqrt(
            bw ** 2 +
            bh ** 2
        )
    ) / 2.0

    return float(
        distance /
        max(diagonal, 1.0)
    )


# ============================================================
# STABLE BODY ANCHOR FROM SEGMENTATION MASK
#
# Tracking must NOT depend on the exact silhouette of hands/arms.
# We therefore derive an anchor from the central torso region.
# ============================================================

def get_stable_body_anchor(bbox, mask):
    x1, y1, x2, y2 = bbox
    default = center_of_box(bbox)

    if mask is None:
        return default

    m = mask[y1:y2, x1:x2]
    if m.size == 0:
        return default

    m = (m > 127).astype(np.uint8)

    yy1 = int(0.12 * m.shape[0])
    yy2 = int(0.88 * m.shape[0])
    xx1 = int(0.22 * m.shape[1])
    xx2 = int(0.78 * m.shape[1])
    roi = np.zeros_like(m)
    roi[yy1:yy2, xx1:xx2] = m[yy1:yy2, xx1:xx2]

    kernel = np.ones((5, 5), np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

    ys, xs = np.where(roi > 0)
    if len(xs) < 20:
        return default

    return np.array([
        x1 + float(np.mean(xs)),
        y1 + float(np.mean(ys))
    ], dtype=np.float32)


# ============================================================
# TRACK MATCH (purely geometric - never uses appearance/identity)
# ============================================================

def track_match_cost(
    track,
    bbox,
    mask=None
):

    current_box = track["bbox"]

    iou = calculate_iou(
        current_box,
        bbox
    )

    center_dist = normalized_center_distance(
        current_box,
        bbox
    )

    current_anchor = track.get(
        "body_anchor",
        center_of_box(current_box)
    )
    new_anchor = get_stable_body_anchor(bbox, mask)
    anchor_distance = np.linalg.norm(
        new_anchor - current_anchor
    )
    anchor_norm = anchor_distance / max(
        1.0, current_box[3] - current_box[1]
    )

    old_center = center_of_box(
        current_box
    )

    new_center = center_of_box(
        bbox
    )

    velocity = track.get(
        "velocity",
        np.zeros(2)
    )

    predicted_center = (
        old_center +
        velocity
    )

    prediction_error = np.linalg.norm(
        new_center -
        predicted_center
    )

    box_h = max(
        1,
        current_box[3] -
        current_box[1]
    )

    prediction_error_norm = (
        prediction_error /
        box_h
    )

    allowed_center_distance = TRACK_CENTER_DIST_MULTIPLIER
    allowed_prediction_error = TRACK_PREDICTION_ERROR_MULTIPLIER

    if (
        iou < TRACK_IOU_THRESHOLD
        and
        center_dist > allowed_center_distance
        and
        prediction_error_norm > allowed_prediction_error
        and
        anchor_norm > TRACK_ANCHOR_REJECT_THRESHOLD
    ):
        return None

    cost = (
        0.50 * (1.0 - iou)
        + 0.20 * center_dist
        + 0.10 * prediction_error_norm
        + TRACK_MASK_ANCHOR_WEIGHT * anchor_norm
    )

    return cost


# ============================================================
# UPDATE TRACK MOTION
# ============================================================

def update_track_motion(
    track,
    new_bbox,
    mask=None
):

    old_center = center_of_box(
        track["bbox"]
    )

    new_center = center_of_box(
        new_bbox
    )

    instant_velocity = (
        new_center -
        old_center
    )

    old_velocity = track.get(
        "velocity",
        np.zeros(2)
    )

    velocity = (
        0.70 *
        old_velocity +
        0.30 *
        instant_velocity
    )

    track["velocity"] = velocity

    new_anchor = get_stable_body_anchor(new_bbox, mask)
    old_anchor = track.get("body_anchor", old_center)
    track["body_anchor"] = (
        0.70 * old_anchor +
        0.30 * new_anchor
    ).astype(np.float32)

    track["bbox"] = new_bbox

    track["missed_frames"] = 0


# ============================================================
# CREATE TRACK
# ============================================================

def create_track(
    bbox,
    mask,
    now
):

    global next_track_id

    track_id = next_track_id

    next_track_id += 1

    track = {

        "track_id": track_id,

        "bbox": bbox,

        "mask": mask,

        "body_anchor": get_stable_body_anchor(
            bbox, mask
        ),

        "created": now,

        "last_seen": now,

        "history": [],

        "identity": None,

        "identity_score": 0.0,

        "weak_id_frames": 0,

        "unknown_streak": 0,

        "unknown_candidate_embedding": None,

        "unknown_candidate_color": None,

        "unknown_face_streak": 0,

        "unknown_face_candidate_embedding": None,

        "embedding": None,

        "color": None,

        "face_embedding": None,

        "last_face_seen": 0.0,

        "last_identification": 0,

        "last_saved": 0,

        "samples": [],

        "color_samples": [],

        "velocity": np.zeros(
            2,
            dtype=np.float32
        ),

        "missed_frames": 0,

        "candidate_identity": None,

        "candidate_identity_streak": 0,

        "reid_valid": False
    }

    tracks[track_id] = track

    return track


# ============================================================
# REMOVE OLD TRACKS
# ============================================================

def cleanup_tracks(now):

    remove_ids = []

    for track_id, track in tracks.items():

        if (
            now -
            track["last_seen"]
            >
            TRACK_TIMEOUT
        ):

            remove_ids.append(
                track_id
            )

    for track_id in remove_ids:

        del tracks[track_id]


# ============================================================
# FACE MATCHING AGAINST REGISTERED PEOPLE
#
# This is the ONLY function allowed to say "this track is Person_XXX".
# ============================================================

def compare_faces_with_database(face_embedding):

    if face_embedding is None:
        return None

    results = []

    for name, data in database.items():

        if not name.startswith("Person_"):
            continue

        stored_face = data.get("face_profile")

        if stored_face is None:
            continue

        profile_score = cosine_similarity(face_embedding, stored_face)

        best_sample_score = profile_score

        face_samples = data.get("face_samples")

        if face_samples is not None:
            try:
                if len(face_samples) > 0:
                    sample_scores = [
                        cosine_similarity(face_embedding, s)
                        for s in face_samples
                    ]
                    best_sample_score = max(sample_scores)
            except:
                pass

        final_score = 0.5 * profile_score + 0.5 * best_sample_score

        results.append({
            "name": name,
            "score": float(final_score)
        })

    if not results:
        return None

    results.sort(key=lambda x: x["score"], reverse=True)

    best = results[0]

    second_score = results[1]["score"] if len(results) > 1 else 0.0

    best["difference"] = best["score"] - second_score

    return best


# ============================================================
# UNKNOWN MATCH (BODY+COLOUR) - faceless fallback only
# ============================================================

def compare_unknown_candidate(
    embedding,
    color_descriptor
):

    best = None

    for name, data in database.items():

        if not name.startswith(
            "Unknown_"
        ):
            continue

        stored_embedding = data.get(
            "embedding"
        )

        if stored_embedding is None:
            continue

        reid_score = cosine_similarity(
            embedding,
            stored_embedding
        )

        stored_color = data.get(
            "color_profile"
        )

        if (
            color_descriptor is not None
            and
            stored_color is not None
        ):

            color_score = color_similarity(
                color_descriptor,
                stored_color
            )

        else:

            color_score = 0.0

        score = (
            0.75 *
            reid_score
            +
            0.25 *
            color_score
        )

        if (
            best is None
            or
            score >
            best["score"]
        ):

            best = {
                "name": name,
                "score": float(score)
            }

    return best


def compare_unknown_face_candidate(face_embedding):

    best = None

    for name, data in database.items():

        if not name.startswith("Unknown_"):
            continue

        stored_face = data.get("face_profile")

        if stored_face is None:
            continue

        score = cosine_similarity(face_embedding, stored_face)

        if best is None or score > best["score"]:
            best = {"name": name, "score": float(score)}

    return best


# ============================================================
# CREATE UNKNOWN (BODY+COLOUR) - faceless fallback only
# ============================================================

def create_unknown(
    embedding,
    color_descriptor
):

    global database

    existing = compare_unknown_candidate(
        embedding,
        color_descriptor
    )

    if (
        existing is not None
        and
        existing["score"] >=
        UNKNOWN_MATCH_THRESHOLD
    ):

        return existing["name"], False

    number = get_next_unknown_id()

    name = (
        f"Unknown_{number:03d}"
    )

    success, result = save_profile(
        name,
        embedding,
        color_descriptor,
        samples=[
            embedding
        ],
        color_samples=[
            color_descriptor
        ]
        if color_descriptor is not None
        else None
    )

    if not success:

        print(
            "Could not create Unknown:",
            result
        )

        return None, False

    database[name] = {

        "embedding": embedding.copy(),

        "color_profile":
            None
            if color_descriptor is None
            else color_descriptor.copy(),

        "samples":
            np.asarray(
                [embedding],
                dtype=np.float32
            ),

        "color_samples":
            None
            if color_descriptor is None
            else np.asarray(
                [color_descriptor],
                dtype=np.float32
            ),

        "face_profile": None,

        "face_samples": None
    }

    print(
        "NEW UNKNOWN (body-based) SAVED:",
        name
    )

    return name, True


# ============================================================
# CREATE UNKNOWN (FACE-BASED) - preferred whenever a stranger's
# face is visible, so the same unregistered person doesn't get a
# new Unknown_ entry every time their clothes change.
# ============================================================

def create_unknown_face(face_embedding, body_embedding, color_descriptor):

    global database

    existing = compare_unknown_face_candidate(face_embedding)

    if existing is not None and existing["score"] >= UNKNOWN_MATCH_THRESHOLD_FACE:
        return existing["name"], False

    # save_profile requires a real body embedding for schema
    # consistency, even though it is not used to decide this entry's
    # identity. In practice OSNet always runs alongside the face
    # detector, so this should basically never be missing.
    if body_embedding is None:
        print("Could not create face-based Unknown: no body embedding.")
        return None, False

    number = get_next_unknown_id()
    name = f"Unknown_{number:03d}"

    success, result = save_profile(
        name,
        body_embedding,
        color_descriptor,
        samples=[body_embedding],
        color_samples=(
            [color_descriptor] if color_descriptor is not None else None
        ),
        face_descriptor=face_embedding,
        face_samples=[face_embedding]
    )

    if not success:
        print("Could not create Unknown:", result)
        return None, False

    database[name] = {
        "embedding": body_embedding.copy(),
        "color_profile": (
            None if color_descriptor is None else color_descriptor.copy()
        ),
        "samples": np.asarray([body_embedding], dtype=np.float32),
        "color_samples": (
            None if color_descriptor is None
            else np.asarray([color_descriptor], dtype=np.float32)
        ),
        "face_profile": face_embedding.copy(),
        "face_samples": np.asarray([face_embedding], dtype=np.float32)
    }

    print("NEW UNKNOWN (face-based) SAVED:", name)

    return name, True


# ============================================================
# UPDATE UNKNOWN PROFILE (BODY+COLOUR bookkeeping)
# ============================================================

def update_unknown_profile(
    name,
    embedding,
    color_descriptor
):
    if name not in database or not name.startswith("Unknown_"):
        return False

    embedding = normalize_vector(embedding)
    if embedding is None:
        return False

    data = database[name]
    stored_embedding = data.get("embedding")

    if (
        stored_embedding is not None
        and cosine_similarity(embedding, stored_embedding)
        < UNKNOWN_PROFILE_UPDATE_THRESHOLD
    ):
        return False

    samples = data.get("samples")
    valid_samples = []
    if samples is not None:
        for sample in np.asarray(samples):
            sample_norm = normalize_vector(sample)
            if sample_norm is not None:
                valid_samples.append(sample_norm)

    if not is_duplicate_sample(embedding, valid_samples):
        valid_samples.append(embedding)

    valid_samples = valid_samples[-30:]
    data["samples"] = np.asarray(valid_samples, dtype=np.float32)
    data["embedding"] = normalize_vector(
        np.mean(data["samples"], axis=0)
    )

    if color_descriptor is not None:
        color_descriptor = normalize_vector(color_descriptor)
        if color_descriptor is not None:
            color_samples = data.get("color_samples")
            valid_colors = []
            if color_samples is not None:
                for sample in np.asarray(color_samples):
                    sample_norm = normalize_vector(sample)
                    if sample_norm is not None:
                        valid_colors.append(sample_norm)

            if not is_duplicate_sample(color_descriptor, valid_colors):
                valid_colors.append(color_descriptor)

            valid_colors = valid_colors[-30:]
            data["color_samples"] = np.asarray(
                valid_colors, dtype=np.float32
            )
            data["color_profile"] = normalize_vector(
                np.mean(data["color_samples"], axis=0)
            )

    save_profile(
        name,
        data["embedding"],
        data.get("color_profile"),
        data.get("samples"),
        data.get("color_samples"),
        face_descriptor=data.get("face_profile"),
        face_samples=data.get("face_samples")
    )
    return True


# ============================================================
# IDENTITY STICKINESS
#
# The ONLY function that assigns/changes a track's identity, and it
# is ONLY ever called with a FACE-based prediction. Body/colour are
# never allowed anywhere near this decision - see main() for why.
# ============================================================

def update_track_identity(
    track,
    prediction
):

    if prediction is None:
        return

    predicted_name = prediction.get("name")
    predicted_score = prediction.get("score", 0.0)
    difference = prediction.get("difference", 0.0)

    current_identity = track.get("identity")

    # ========================================================
    # CURRENT IDENTITY ALREADY STABLE
    # ========================================================

    if current_identity is not None:

        if (
            predicted_score < FACE_CONFIDENT_THRESHOLD
            or difference < FACE_MIN_DIFFERENCE
        ):

            track["weak_id_frames"] += 1

            if track["weak_id_frames"] >= MAX_WEAK_ID_FRAMES:
                track["identity"] = None
                track["identity_score"] = 0.0
                track["candidate_identity"] = None
                track["candidate_identity_streak"] = 0
                track["unknown_streak"] = 0
                track["unknown_candidate_embedding"] = None
                track["unknown_candidate_color"] = None
                track["unknown_face_streak"] = 0
                track["unknown_face_candidate_embedding"] = None
                track["weak_id_frames"] = 0

            return

        # Same identity
        if predicted_name == current_identity:

            track["weak_id_frames"] = 0

            track["identity_score"] = (
                0.80 * track["identity_score"]
                + 0.20 * predicted_score
            )

            track["candidate_identity"] = None
            track["candidate_identity_streak"] = 0

            return

        # Different identity - require multiple consistent frames
        if predicted_name == track.get("candidate_identity"):
            track["candidate_identity_streak"] += 1
        else:
            track["candidate_identity"] = predicted_name
            track["candidate_identity_streak"] = 1

        if (
            track["candidate_identity_streak"]
            >= IDENTITY_SWITCH_CONFIRM_FRAMES
        ):

            track["identity"] = predicted_name
            track["identity_score"] = predicted_score
            track["weak_id_frames"] = 0
            track["candidate_identity"] = None
            track["candidate_identity_streak"] = 0

        return

    # ========================================================
    # NO CURRENT IDENTITY
    # ========================================================

    if (
        predicted_name is not None
        and predicted_score >= FACE_CONFIDENT_THRESHOLD
        and difference >= FACE_MIN_DIFFERENCE
    ):

        track["identity"] = predicted_name
        track["identity_score"] = predicted_score
        track["weak_id_frames"] = 0
        track["unknown_streak"] = 0
        track["unknown_face_streak"] = 0

        return


# ============================================================
# PROCESS UNKNOWN FOR TRACK (BODY+COLOUR) - faceless fallback
# ============================================================

def process_unknown_track(
    track,
    embedding,
    color_descriptor
):

    current_identity = track.get("identity")

    if current_identity is not None:
        return

    candidate_embedding = track.get("unknown_candidate_embedding")

    if candidate_embedding is None:

        track["unknown_candidate_embedding"] = embedding.copy()

        track["unknown_candidate_color"] = (
            None if color_descriptor is None else color_descriptor.copy()
        )

        track["unknown_streak"] = 1

        return

    reid_similarity = cosine_similarity(embedding, candidate_embedding)

    candidate_color = track.get("unknown_candidate_color")

    if color_descriptor is not None and candidate_color is not None:
        color_similarity_value = color_similarity(
            color_descriptor, candidate_color
        )
    else:
        color_similarity_value = 0.0

    consistency_score = (
        0.75 * reid_similarity + 0.25 * color_similarity_value
    )

    if consistency_score >= UNKNOWN_MATCH_THRESHOLD:

        track["unknown_streak"] += 1

        candidate_embedding = normalize_vector(
            0.75 * candidate_embedding + 0.25 * embedding
        )

        track["unknown_candidate_embedding"] = candidate_embedding

        if color_descriptor is not None:

            if candidate_color is None:
                candidate_color = color_descriptor.copy()
            else:
                candidate_color = normalize_vector(
                    0.75 * candidate_color + 0.25 * color_descriptor
                )

            track["unknown_candidate_color"] = candidate_color

    else:

        track["unknown_streak"] = 1
        track["unknown_candidate_embedding"] = embedding.copy()
        track["unknown_candidate_color"] = (
            None if color_descriptor is None else color_descriptor.copy()
        )

    if track["unknown_streak"] >= CONFIRM_UNKNOWN_FRAMES:

        name, created = create_unknown(
            track["unknown_candidate_embedding"],
            track.get("unknown_candidate_color")
        )

        if name is not None:

            track["identity"] = name
            track["identity_score"] = UNKNOWN_MATCH_THRESHOLD
            track["unknown_streak"] = 0
            track["unknown_candidate_embedding"] = None
            track["unknown_candidate_color"] = None


# ============================================================
# PROCESS UNKNOWN FACE FOR TRACK - preferred whenever a stranger's
# face is visible.
# ============================================================

def process_unknown_face_track(track, face_embedding, color_descriptor):

    current_identity = track.get("identity")

    if current_identity is not None:
        return

    candidate = track.get("unknown_face_candidate_embedding")

    if candidate is None:
        track["unknown_face_candidate_embedding"] = face_embedding.copy()
        track["unknown_face_streak"] = 1
        return

    similarity = cosine_similarity(face_embedding, candidate)

    if similarity >= UNKNOWN_MATCH_THRESHOLD_FACE:
        track["unknown_face_streak"] += 1
        candidate = normalize_vector(
            0.75 * candidate + 0.25 * face_embedding
        )
        track["unknown_face_candidate_embedding"] = candidate
    else:
        track["unknown_face_streak"] = 1
        track["unknown_face_candidate_embedding"] = face_embedding.copy()

    if track["unknown_face_streak"] >= CONFIRM_UNKNOWN_FRAMES:

        name, created = create_unknown_face(
            track["unknown_face_candidate_embedding"],
            track.get("embedding"),
            color_descriptor
        )

        if name is not None:
            track["identity"] = name
            track["identity_score"] = UNKNOWN_MATCH_THRESHOLD_FACE
            track["unknown_face_streak"] = 0
            track["unknown_face_candidate_embedding"] = None


# ============================================================
# DUPLICATE CHECK
# ============================================================

def is_duplicate_sample_with_threshold(
    embedding,
    samples,
    threshold
):
    if samples is None or len(samples) == 0:
        return False

    for sample in samples:
        similarity = cosine_similarity(embedding, sample)
        if similarity >= threshold:
            return True
    return False


def is_duplicate_sample(
    embedding,
    samples
):

    if samples is None:
        return False

    if len(samples) == 0:
        return False

    for sample in samples:

        similarity = cosine_similarity(
            embedding,
            sample
        )

        if (
            similarity >=
            DUPLICATE_SIM_THRESHOLD
        ):

            return True

    return False


# ============================================================
# ADD SAMPLE TO TRACK (body+colour bookkeeping only)
# ============================================================

def add_sample_to_track(
    track,
    embedding,
    color_descriptor,
    now
):

    if embedding is None:
        return False

    if is_duplicate_sample(
        embedding,
        track["samples"]
    ):

        return False

    track["samples"].append(
        embedding.copy()
    )

    if color_descriptor is not None:

        track["color_samples"].append(
            color_descriptor.copy()
        )

    if len(track["samples"]) > 30:

        track["samples"] = (
            track["samples"][-30:]
        )

    if len(track["color_samples"]) > 30:

        track["color_samples"] = (
            track["color_samples"][-30:]
        )

    track["last_saved"] = now

    return True


# ============================================================
# SAVE CURRENT KNOWN TRACK SAMPLE
#
# Keeps the Person_'s body/colour AND face samples fresh over time.
# Body/colour are archival bookkeeping only (never used to decide
# identity); the face samples are what actually keep recognition
# working long-term.
# ============================================================

def save_track_sample(
    track,
    now
):

    identity = track.get("identity")

    if identity is None:
        return

    if not identity.startswith("Person_"):
        return

    embedding = track.get("embedding")
    color_descriptor = track.get("color")

    if embedding is None:
        return

    if now - track.get("last_saved", 0) < SAVE_INTERVAL:
        return

    if identity not in database:
        return

    data = database[identity]

    stored_embedding = data.get("embedding")

    if (
        stored_embedding is not None
        and cosine_similarity(embedding, stored_embedding)
        < PERSON_PROFILE_UPDATE_THRESHOLD
    ):
        return

    added = add_sample_to_track(
        track,
        embedding,
        color_descriptor,
        now
    )

    if not added:
        return

    old_samples = data.get("samples")

    if old_samples is None:
        old_samples = np.empty((0, len(embedding)), dtype=np.float32)

    new_samples = np.vstack([old_samples, embedding.reshape(1, -1)])

    if len(new_samples) > 50:
        new_samples = new_samples[-50:]

    new_embedding = normalize_vector(np.mean(new_samples, axis=0))

    data["samples"] = new_samples
    data["embedding"] = new_embedding

    if color_descriptor is not None:

        old_color_samples = data.get("color_samples")

        if old_color_samples is None:
            old_color_samples = np.empty(
                (0, len(color_descriptor)), dtype=np.float32
            )

        new_color_samples = np.vstack(
            [old_color_samples, color_descriptor.reshape(1, -1)]
        )

        if len(new_color_samples) > 50:
            new_color_samples = new_color_samples[-50:]

        new_color_profile = normalize_vector(
            np.mean(new_color_samples, axis=0)
        )

        data["color_samples"] = new_color_samples
        data["color_profile"] = new_color_profile

    # --------------------------------------------------------
    # Face bookkeeping: only add a face sample when we actually saw a
    # FRESH face very recently (avoid repeatedly re-adding the same
    # stale cached vector every SAVE_INTERVAL).
    # --------------------------------------------------------

    face_embedding = track.get("face_embedding")

    if (
        face_embedding is not None
        and now - track.get("last_face_seen", -999) < 1.0
    ):

        old_face_samples = data.get("face_samples")

        if old_face_samples is None:
            old_face_samples = np.empty(
                (0, len(face_embedding)), dtype=np.float32
            )

        new_face_samples = np.vstack(
            [old_face_samples, face_embedding.reshape(1, -1)]
        )

        if len(new_face_samples) > 50:
            new_face_samples = new_face_samples[-50:]

        new_face_profile = normalize_vector(
            np.mean(new_face_samples, axis=0)
        )

        data["face_samples"] = new_face_samples
        data["face_profile"] = new_face_profile

    save_profile(
        identity,
        data["embedding"],
        data.get("color_profile"),
        data.get("samples"),
        data.get("color_samples"),
        face_descriptor=data.get("face_profile"),
        face_samples=data.get("face_samples")
    )


# ============================================================
# GET VALID TRACKS
# ============================================================

def get_visible_tracks():

    return list(
        tracks.values()
    )


# ============================================================
# DETECTION
# ============================================================

def detect_persons(
    frame,
    now
):

    detections = []

    results = yolo.predict(
        frame,
        conf=YOLO_CONF,
        classes=[0],
        verbose=False
    )

    faces = detect_faces(frame, now)

    if not results:
        return detections

    result = results[0]

    boxes = result.boxes

    masks = result.masks

    if boxes is None:
        return detections

    for i in range(
        len(boxes)
    ):

        cls = int(
            boxes.cls[i].item()
        )

        if cls != 0:
            continue

        confidence = float(
            boxes.conf[i].item()
        )

        xyxy = boxes.xyxy[
            i
        ].cpu().numpy()

        x1, y1, x2, y2 = map(
            int,
            xyxy
        )

        height = (
            y2 - y1
        )

        width = (
            x2 - x1
        )

        if height <= 0 or width <= 0:
            continue

        reid_valid = (
            height >=
            MIN_PERSON_HEIGHT
        )

        mask = None

        if masks is not None:

            try:

                mask_tensor = (
                    masks.data[i]
                    .cpu()
                    .numpy()
                )

                mask = (
                    mask_tensor *
                    255
                ).astype(
                    np.uint8
                )

                mask = cv2.resize(
                    mask,
                    (
                        frame.shape[1],
                        frame.shape[0]
                    ),
                    interpolation=cv2.INTER_NEAREST
                )

            except Exception as e:

                mask = None

        matched_face = match_face_to_person((x1, y1, x2, y2), faces)

        detections.append(
            {
                "bbox": (
                    x1,
                    y1,
                    x2,
                    y2
                ),
                "confidence": confidence,
                "mask": mask,
                "reid_valid": reid_valid,
                "face_embedding": (
                    matched_face["embedding"] if matched_face else None
                ),
                "face_score": (
                    matched_face["score"] if matched_face else 0.0
                )
            }
        )

    return detections


# ============================================================
# REGISTRATION FUNCTIONS
# ============================================================

def start_registration():

    global register_mode
    global register_samples
    global register_color_samples
    global register_face_samples
    global register_target_name
    global register_target_track_id
    global register_last_sample_time

    if register_mode:
        return

    number = get_next_person_id()

    register_target_name = (
        f"Person_{number:03d}"
    )

    register_samples = []
    register_color_samples = []
    register_face_samples = []

    register_target_track_id = None
    register_last_sample_time = 0.0

    register_mode = True

    set_status(
        f"REGISTRATION STARTED: "
        f"{register_target_name}",
        4
    )

    print(
        "\nRegistration started:",
        register_target_name
    )

    print(
        "Keep your face clearly visible to the camera. Turn your "
        "head slightly left/right/up/down during collection for a "
        "more robust profile."
    )


def cancel_registration():

    global register_mode
    global register_samples
    global register_color_samples
    global register_face_samples
    global register_target_name
    global register_target_track_id
    global register_last_sample_time

    register_mode = False

    register_samples = []
    register_color_samples = []
    register_face_samples = []

    register_target_name = None
    register_target_track_id = None
    register_last_sample_time = 0.0

    set_status(
        "REGISTRATION CANCELLED",
        3
    )


def process_registration(
    track,
    embedding,
    color_descriptor,
    face_embedding
):

    global register_mode
    global register_samples
    global register_color_samples
    global register_face_samples
    global register_target_name
    global register_target_track_id
    global register_last_sample_time

    if not register_mode:
        return

    if register_target_track_id is None:
        return

    if track.get("track_id") != register_target_track_id:
        return

    # A real face is required to build a clothes-independent profile.
    # Body/colour samples are still collected opportunistically below,
    # but they are never the thing that decides when registration is
    # "done".
    if face_embedding is None:
        return

    now_sample = time.time()

    if now_sample - register_last_sample_time < REGISTER_MIN_INTERVAL:
        return

    if is_duplicate_sample_with_threshold(
        face_embedding,
        register_face_samples,
        REGISTER_DUPLICATE_SIM_THRESHOLD
    ):
        return

    if register_face_samples:

        running_mean = normalize_vector(
            np.mean(register_face_samples, axis=0)
        )

        consistency = (
            cosine_similarity(face_embedding, running_mean)
            if running_mean is not None
            else 1.0
        )

        if consistency < REGISTER_CONSISTENCY_THRESHOLD:

            print(
                f"\n[REJECTED] Face inconsistent with previous "
                f"captures (similarity={consistency:.2f}). Possibly "
                f"a different person / track jump - skipped."
            )

            return

    register_face_samples.append(face_embedding.copy())
    register_last_sample_time = now_sample

    if embedding is not None:
        register_samples.append(embedding.copy())

    if color_descriptor is not None:
        register_color_samples.append(color_descriptor.copy())

    print(
        f"\rRegistering "
        f"{register_target_name}: "
        f"{len(register_face_samples)}/"
        f"{NUM_SAMPLES} (face samples)",
        end="",
        flush=True
    )

    if len(register_face_samples) >= NUM_SAMPLES:

        face_profile = normalize_vector(
            np.mean(register_face_samples, axis=0)
        )

        embedding_profile = (
            normalize_vector(np.mean(register_samples, axis=0))
            if register_samples else None
        )

        color_profile = (
            normalize_vector(np.mean(register_color_samples, axis=0))
            if register_color_samples else None
        )

        if embedding_profile is None:

            print(
                "\n[ERROR] No usable body embedding collected during "
                "registration - aborting save."
            )

            set_status("REGISTRATION FAILED (no body embedding)", 5)

            register_mode = False
            register_samples = []
            register_color_samples = []
            register_face_samples = []
            register_target_name = None
            register_target_track_id = None

            return

        success, result = save_profile(
            register_target_name,
            embedding_profile,
            color_profile,
            register_samples,
            register_color_samples,
            face_descriptor=face_profile,
            face_samples=register_face_samples
        )

        if success:

            database[register_target_name] = {

                "embedding": embedding_profile,

                "color_profile": color_profile,

                "samples": np.asarray(
                    register_samples, dtype=np.float32
                ),

                "color_samples": (
                    np.asarray(register_color_samples, dtype=np.float32)
                    if register_color_samples else None
                ),

                "face_profile": face_profile,

                "face_samples": np.asarray(
                    register_face_samples, dtype=np.float32
                )
            }

            print(
                "\n\n======================================"
            )

            print(
                register_target_name,
                "SAVED SUCCESSFULLY"
            )

            print("File:", result)

            print("Face samples:", len(register_face_samples))

            print(
                "======================================\n"
            )

            set_status(
                f"{register_target_name} SAVED SUCCESSFULLY",
                5
            )

        else:

            print("\nSAVE FAILED:", result)

            set_status(f"SAVE FAILED: {result}", 5)

        register_mode = False
        register_samples = []
        register_color_samples = []
        register_face_samples = []
        register_target_name = None
        register_target_track_id = None


# ============================================================
# ADD SAMPLES TO EXISTING PERSON
# ============================================================

def start_add_samples():

    global add_mode
    global add_samples
    global add_color_samples
    global add_face_samples
    global add_target_name
    global add_target_track_id

    stable_tracks = [
        t
        for t in tracks.values()
        if (
            t.get("identity") is not None
            and t["identity"].startswith("Person_")
        )
    ]

    if not stable_tracks:

        set_status(
            "A: NO KNOWN PERSON TRACK",
            4
        )

        return

    stable_tracks.sort(
        key=lambda t: t["last_seen"],
        reverse=True
    )

    target = stable_tracks[0]

    add_target_name = target["identity"]

    add_samples = []
    add_color_samples = []
    add_face_samples = []

    add_target_track_id = target.get("track_id")

    add_mode = True

    set_status(
        f"ADDING SAMPLES: {add_target_name}",
        4
    )

    print(
        "\nAdding samples to:",
        add_target_name
    )


def process_add_samples(
    track,
    embedding,
    color_descriptor,
    face_embedding
):

    global add_mode
    global add_samples
    global add_color_samples
    global add_face_samples
    global add_target_name
    global add_target_track_id

    if not add_mode:
        return

    if add_target_track_id is None:
        return

    if track.get("track_id") != add_target_track_id:
        return

    if track.get("identity") != add_target_name:
        return

    if face_embedding is None:
        return

    if is_duplicate_sample(face_embedding, add_face_samples):
        return

    if add_target_name in database:

        stored_face = database[add_target_name].get("face_profile")

        profile_score = (
            cosine_similarity(face_embedding, stored_face)
            if stored_face is not None
            else 1.0
        )

        if profile_score < REGISTER_CONSISTENCY_THRESHOLD:

            print(
                f"\n[REJECTED] Face does not match existing profile "
                f"of {add_target_name} (similarity="
                f"{profile_score:.2f}). Skipped."
            )

            return

    add_face_samples.append(face_embedding.copy())

    if embedding is not None:
        add_samples.append(embedding.copy())

    if color_descriptor is not None:
        add_color_samples.append(color_descriptor.copy())

    print(
        f"\rAdding "
        f"{add_target_name}: "
        f"{len(add_face_samples)}/"
        f"{NUM_SAMPLES} (face samples)",
        end="",
        flush=True
    )

    if len(add_face_samples) >= NUM_SAMPLES:

        if add_target_name not in database:

            set_status("TARGET NOT IN DATABASE", 4)

            add_mode = False
            add_target_track_id = None
            return

        data = database[add_target_name]

        if add_samples:

            old_samples = data.get("samples")

            if old_samples is None:
                old_samples = np.empty(
                    (0, len(add_samples[0])), dtype=np.float32
                )

            all_samples = np.vstack(
                [old_samples, np.asarray(add_samples, dtype=np.float32)]
            )

            if len(all_samples) > 100:
                all_samples = all_samples[-100:]

            new_embedding = normalize_vector(
                np.mean(all_samples, axis=0)
            )

            data["samples"] = all_samples
            data["embedding"] = new_embedding

        if add_color_samples:

            old_color_samples = data.get("color_samples")

            if old_color_samples is None:
                old_color_samples = np.empty(
                    (0, len(add_color_samples[0])), dtype=np.float32
                )

            all_color_samples = np.vstack(
                [
                    old_color_samples,
                    np.asarray(add_color_samples, dtype=np.float32)
                ]
            )

            if len(all_color_samples) > 100:
                all_color_samples = all_color_samples[-100:]

            new_color_profile = normalize_vector(
                np.mean(all_color_samples, axis=0)
            )

            data["color_samples"] = all_color_samples
            data["color_profile"] = new_color_profile

        old_face_samples = data.get("face_samples")

        if old_face_samples is None:
            old_face_samples = np.empty(
                (0, len(add_face_samples[0])), dtype=np.float32
            )

        all_face_samples = np.vstack(
            [
                old_face_samples,
                np.asarray(add_face_samples, dtype=np.float32)
            ]
        )

        if len(all_face_samples) > 100:
            all_face_samples = all_face_samples[-100:]

        new_face_profile = normalize_vector(
            np.mean(all_face_samples, axis=0)
        )

        data["face_samples"] = all_face_samples
        data["face_profile"] = new_face_profile

        success, result = save_profile(
            add_target_name,
            data["embedding"],
            data.get("color_profile"),
            data.get("samples"),
            data.get("color_samples"),
            face_descriptor=data.get("face_profile"),
            face_samples=data.get("face_samples")
        )

        if success:

            print("\n\n", add_target_name, "UPDATED SUCCESSFULLY\n")

            set_status(f"{add_target_name} UPDATED SUCCESSFULLY", 5)

        else:

            print("\nUPDATE FAILED:", result)

            set_status(f"UPDATE FAILED: {result}", 5)

        add_mode = False
        add_samples = []
        add_color_samples = []
        add_face_samples = []
        add_target_name = None
        add_target_track_id = None


# ============================================================
# STATUS MESSAGE
# ============================================================

def set_status(
    message,
    duration=3
):

    global status_message
    global status_message_until

    status_message = message

    status_message_until = (
        time.time() +
        duration
    )


# ============================================================
# DRAW TEXT
# ============================================================

def draw_text(
    frame,
    text,
    position,
    scale=0.65,
    thickness=2
):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# MAIN
# ============================================================

def main():

    global register_mode
    global register_samples
    global register_color_samples
    global register_face_samples
    global register_target_name
    global register_target_track_id

    global add_mode
    global add_samples
    global add_color_samples
    global add_face_samples
    global add_target_name
    global add_target_track_id

    global last_frame_time

    print("\nOpening camera...")

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():

        print(
            "ERROR: Could not open camera."
        )

        return

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    print("\nCamera started.")

    print("Controls:")
    print("  S = Register new person (needs a visible face)")
    print("  A = Add samples to current known person (needs a face)")
    print("  C = Cancel registration/add mode")
    print("  Q = Quit")

    if face_app is None:
        print(
            "\n[WARNING] Face recognition is NOT active - install "
            "insightface + onnxruntime and restart. Until then, no "
            "Person_ can be registered or recognised."
        )

    print("\n--------------------------------------")

    while True:

        ret, frame = cap.read()

        if not ret:

            print("Could not read frame.")

            break

        now = time.time()

        # ----------------------------------------------------
        # Detection (persons + faces)
        # ----------------------------------------------------

        detections = detect_persons(frame, now)

        # ----------------------------------------------------
        # Match detections to tracks
        # ----------------------------------------------------

        used_track_ids = set()

        frame_tracks = []

        for detection in detections:

            bbox = detection["bbox"]
            mask = detection["mask"]

            candidates = []

            for track_id, track in tracks.items():

                if track_id in used_track_ids:
                    continue

                if now - track["last_seen"] > TRACK_TIMEOUT:
                    continue

                cost = track_match_cost(track, bbox, mask)

                if cost is not None:
                    candidates.append((cost, track))

            if candidates:

                candidates.sort(key=lambda x: x[0])

                track = candidates[0][1]

                update_track_motion(track, bbox, mask)

                track["mask"] = mask
                track["last_seen"] = now

            else:

                track = create_track(bbox, mask, now)

            track["reid_valid"] = detection["reid_valid"]

            used_track_ids.add(track["track_id"])

            frame_tracks.append((track, detection))

        cleanup_tracks(now)

        # ----------------------------------------------------
        # Process each track
        # ----------------------------------------------------

        for track, detection in frame_tracks:

            bbox = detection["bbox"]
            mask = detection["mask"]
            reid_valid = detection["reid_valid"]

            embedding = None
            color_descriptor = None
            face_embedding_this_frame = None

            if reid_valid:

                prepared = create_reid_input(frame, bbox, mask)

                embedding = extract_embedding(prepared)

                color_descriptor = extract_color_descriptor(
                    frame, bbox, mask
                )

                if embedding is not None:
                    track["embedding"] = embedding

                if color_descriptor is not None:
                    track["color"] = color_descriptor

                face_embedding_this_frame = detection.get("face_embedding")

                if face_embedding_this_frame is not None:
                    track["face_embedding"] = face_embedding_this_frame
                    track["last_face_seen"] = now

                # ----------------------------------------------------
                # Identification interval
                #
                # IMPORTANT: body/colour NEVER get to decide who a
                # track is. If a fresh-enough face is available, face
                # matching against registered Person_ profiles runs.
                # If not, we do nothing at all here - identity
                # stickiness keeps whatever identity the track already
                # had. Only tracks that have NEVER shown a face at all
                # fall back to the old body/colour "someone is here"
                # anonymous bookkeeping.
                # ----------------------------------------------------

                if (
                    now - track["last_identification"]
                    >= IDENTIFICATION_INTERVAL
                ):

                    track["last_identification"] = now

                    face_for_id = None

                    if (
                        track.get("face_embedding") is not None
                        and now - track.get("last_face_seen", -999)
                        <= FACE_STALENESS_SECONDS
                    ):
                        face_for_id = track["face_embedding"]

                    if face_for_id is not None:

                        face_prediction = compare_faces_with_database(
                            face_for_id
                        )

                        update_track_identity(track, face_prediction)

                        if track.get("identity") is None:
                            process_unknown_face_track(
                                track, face_for_id, color_descriptor
                            )

                    else:

                        if track.get("identity") is None:
                            process_unknown_track(
                                track, embedding, color_descriptor
                            )

                save_track_sample(track, now)

            # ----------------------------------------------------
            # Manual registration / add samples (face-gated)
            # ----------------------------------------------------

            if register_mode and reid_valid:

                process_registration(
                    track,
                    embedding,
                    color_descriptor,
                    face_embedding_this_frame
                )

            if add_mode and reid_valid:

                process_add_samples(
                    track,
                    embedding,
                    color_descriptor,
                    face_embedding_this_frame
                )

            # ----------------------------------------------------
            # Draw
            # ----------------------------------------------------

            x1, y1, x2, y2 = bbox

            identity = track.get("identity")

            has_fresh_face = (
                now - track.get("last_face_seen", -999) <= 1.0
            )

            if identity is None:
                label = f"Track {track['track_id']} | Unknown?"
            else:
                label = f"{identity} | T:{track['track_id']}"

            if has_fresh_face:
                label += " [F]"

            if not reid_valid:
                label += " | TOO SMALL FOR REID"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            draw_text(
                frame,
                label,
                (x1, max(25, y1 - 10)),
                0.55,
                2
            )

            if register_mode and reid_valid:

                progress = len(register_face_samples)

                progress_text = (
                    f"REGISTERING {register_target_name}: "
                    f"{progress}/{NUM_SAMPLES} (face) "
                    f"| Track {register_target_track_id}"
                )

                cv2.putText(
                    frame, progress_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 255, 255), 2, cv2.LINE_AA
                )

            if add_mode:

                progress = len(add_face_samples)

                add_text = (
                    f"ADDING {add_target_name}: "
                    f"{progress}/{NUM_SAMPLES} (face) "
                    f"| Track {add_target_track_id}"
                )

                cv2.putText(
                    frame, add_text, (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                    (255, 255, 0), 2, cv2.LINE_AA
                )

        # ====================================================
        # TOP SYSTEM INFORMATION
        # ====================================================

        fps = 1.0 / max(now - last_frame_time, 1e-6)

        last_frame_time = now

        face_status = "ON" if face_app is not None else "OFF (!)"

        info = (
            f"FPS: {fps:.1f} | "
            f"Tracks: {len(tracks)} | "
            f"Database: {len(database)} | "
            f"Face: {face_status}"
        )

        cv2.putText(
            frame, info, (20, frame.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.60,
            (255, 255, 255), 2, cv2.LINE_AA
        )

        controls = "S:Register  A:Add samples  C:Cancel  Q:Quit"

        cv2.putText(
            frame, controls, (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 2, cv2.LINE_AA
        )

        if status_message and time.time() < status_message_until:

            text_size = cv2.getTextSize(
                status_message, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2
            )[0]

            tw, th = text_size

            cv2.rectangle(
                frame, (20, 95), (30 + tw, 110 + th), (0, 0, 0), -1
            )

            cv2.putText(
                frame, status_message, (25, 100 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                (0, 255, 0), 2, cv2.LINE_AA
            )

        cv2.imshow("Person ReID", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("s"):

            if face_app is None:
                set_status(
                    "FACE MODEL NOT LOADED - install insightface", 5
                )

            elif not register_mode and not add_mode:

                start_registration()

                valid_targets = [
                    t for t in tracks.values()
                    if t.get("reid_valid", False)
                ]

                if valid_targets:
                    target = max(
                        valid_targets,
                        key=lambda t: max(1, t["bbox"][2] - t["bbox"][0]) *
                                        max(1, t["bbox"][3] - t["bbox"][1])
                    )
                    register_target_track_id = target["track_id"]
                    set_status(
                        f"TARGET LOCKED: Track {register_target_track_id}",
                        3
                    )
                else:
                    cancel_registration()
                    set_status("S: NO VALID PERSON", 3)

            else:

                set_status("CANCEL CURRENT MODE FIRST", 3)

        elif key == ord("a"):

            if face_app is None:
                set_status(
                    "FACE MODEL NOT LOADED - install insightface", 5
                )

            elif not register_mode and not add_mode:

                start_add_samples()

            else:

                set_status("CANCEL CURRENT MODE FIRST", 3)

        elif key == ord("c"):

            if register_mode:

                cancel_registration()

            elif add_mode:

                add_mode = False
                add_samples = []
                add_color_samples = []
                add_face_samples = []
                add_target_name = None
                add_target_track_id = None

                set_status("ADD MODE CANCELLED", 3)

    cap.release()

    cv2.destroyAllWindows()

    print("\nSystem stopped.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
    
