import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time
import os

# =========================
# Config
# =========================
MODEL_PATH = "FER2013_best_modelV2.keras"   # .keras
CAM_INDEX = 0

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# prediction smoothing
SMOOTH = 6
HOLD_FRAMES = 3
pred_q = deque(maxlen=SMOOTH)

current_label = "neutral"
pending_label = None
pending_count = 0

PREPROCESS_MODE = "rescale"  # "rescale" or "zscore"

THRESH = {
    "happy": 0.12,
    "surprise": 0.15,
    "sad": 0.20,
    "neutral": 0.00,
    "angry": 0.14,
    "fear": 0.18,
    "disgust": 0.08
}
GAP_MIN = 0.02

# =========================
# Face stability config
# =========================
DETECT_EVERY = 6          # run cascade every N frames
TRACKER_TYPE = "CSRT"    # "CSRT" (best) or "KCF" (faster)
BBOX_SMOOTH = 0.75        # EMA: closer to 1 = smoother (less jitter)
MIN_FACE = (90, 90)

# CLAHE helps in bad lighting (for detection & ROI)
USE_CLAHE_FOR_DETECT = True
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Face detector (Haar)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# Load model
# =========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ ไม่พบไฟล์โมเดล: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Loaded Keras model:", MODEL_PATH)

# verify output classes
_dummy = tf.zeros([1, 48, 48, 1], tf.float32)
out = model(_dummy, training=False).numpy()
num_classes = out.shape[-1]
print("✅ Model output classes:", num_classes)

if len(emotion_labels) != num_classes:
    print("⚠️ emotion_labels length != num_classes -> ปรับให้เท่ากัน")
    if len(emotion_labels) > num_classes:
        emotion_labels = emotion_labels[:num_classes]
    else:
        emotion_labels = emotion_labels + [f"class_{i}" for i in range(len(emotion_labels), num_classes)]
    print("✅ adjusted labels:", emotion_labels)

# =========================
# Helpers
# =========================

def preprocess_face(gray_roi_48):
    roi = gray_roi_48.astype("float32")
    if PREPROCESS_MODE == "zscore":
        roi = (roi - np.mean(roi)) / (np.std(roi) + 1e-6)
    else:
        roi = roi / 255.0
    roi = roi.reshape(1, 48, 48, 1)
    return tf.convert_to_tensor(roi, dtype=tf.float32)


def to_probs(vec):
    vec = np.asarray(vec, dtype=np.float32)
    s = float(np.sum(vec))
    if not (0.9 <= s <= 1.1):
        e = np.exp(vec - np.max(vec))
        vec = e / (np.sum(e) + 1e-8)
    return vec


def pick_biggest_face(faces):
    if len(faces) == 0:
        return None
    areas = [w * h for (x, y, w, h) in faces]
    return faces[int(np.argmax(areas))]


def make_tracker():
    """Create an OpenCV tracker if available.

    Note: CSRT/KCF trackers require opencv-contrib-python (or are under cv2.legacy
    in some OpenCV versions). If unavailable, return None and we will fall back
    to running face detection more often.
    """
    t = TRACKER_TYPE.upper()

    def _get(name):
        # OpenCV may expose trackers either at top-level or under cv2.legacy
        if hasattr(cv2, name):
            return getattr(cv2, name)
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, name):
            return getattr(cv2.legacy, name)
        return None

    creators = {
        "CSRT": _get("TrackerCSRT_create"),
        "KCF": _get("TrackerKCF_create"),
        "MOSSE": _get("TrackerMOSSE_create"),
    }

    creator = creators.get(t) or creators.get("CSRT") or creators.get("KCF") or creators.get("MOSSE")
    if creator is None:
        print("⚠️ OpenCV tracker not available (need opencv-contrib-python). Falling back to detection-only mode.")
        return None
    return creator()

def ema_bbox(prev, new, a=BBOX_SMOOTH):
    # prev/new are (x, y, w, h)
    if prev is None:
        return new
    px, py, pw, ph = prev
    nx, ny, nw, nh = new
    x = int(a * px + (1 - a) * nx)
    y = int(a * py + (1 - a) * ny)
    w = int(a * pw + (1 - a) * nw)
    h = int(a * ph + (1 - a) * nh)
    return (x, y, w, h)


# =========================
# Webcam
# =========================
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("❌ เปิดกล้องไม่ได้ ลอง CAM_INDEX=1 หรือ 2")

print("✅ Camera opened. Press 'q' to quit.")

prev_t = time.time()
fps = 0.0

tracker = None
tracked_bbox = None   # (x,y,w,h)
smoothed_bbox = None
frame_i = 0
lost_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_i += 1

    # FPS
    now = time.time()
    dt = now - prev_t
    prev_t = now
    fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # optionally boost contrast for detection
    det_gray = clahe.apply(gray) if USE_CLAHE_FOR_DETECT else gray

    # 1) update tracker every frame if we have it
    if tracker is not None:
        ok, box = tracker.update(frame)
        if ok:
            x, y, w, h = [int(v) for v in box]
            tracked_bbox = (x, y, w, h)
            lost_count = 0
        else:
            lost_count += 1
            if lost_count >= 8:
                tracker = None
                tracked_bbox = None
                smoothed_bbox = None
                pred_q.clear()

    # 2) run detection periodically OR when no tracker
    if (tracker is None) or (frame_i % DETECT_EVERY == 0):
        faces = face_cascade.detectMultiScale(
            det_gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=MIN_FACE
        )
        face = pick_biggest_face(faces)
        if face is not None:
            x, y, w, h = face
            tracked_bbox = (x, y, w, h)
            _trk = make_tracker()
            if _trk is not None:
                tracker = _trk
                tracker.init(frame, (x, y, w, h))
            else:
                tracker = None
            lost_count = 0

    if tracked_bbox is not None:
        # smooth bbox to reduce jitter
        smoothed_bbox = ema_bbox(smoothed_bbox, tracked_bbox, a=BBOX_SMOOTH)
        x, y, w, h = smoothed_bbox

        # pad a bit
        pad = int(0.15 * w)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(gray.shape[1], x + w + pad)
        y2 = min(gray.shape[0], y + h + pad)

        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            pred_q.clear()
            cv2.putText(frame, f"Bad ROI | FPS:{fps:.1f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # mild CLAHE on ROI for robustness
            roi = clahe.apply(roi)
            roi = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)

            inp = preprocess_face(roi)
            vec = model(inp, training=False).numpy()[0]
            probs = to_probs(vec)

            pred_q.append(probs)
            p = np.mean(pred_q, axis=0)

            top3 = p.argsort()[-3:][::-1]
            i1, i2, i3 = [int(i) for i in top3]
            label1, label2, label3 = [emotion_labels[i] for i in top3]
            conf1, conf2, conf3 = float(p[i1]), float(p[i2]), float(p[i3])
            gap = conf1 - conf2

            proposed = label1 if conf1 >= THRESH.get(label1, 0.18) else "neutral"

            # angry override (anti sad swallowing angry)
            if (
                label1 == "sad" and label2 == "angry"
                and conf1 > 0.80 and conf2 > 0.02
            ):
                proposed = "angry"

            # disgust rescue
            if (
                "disgust" in [label1, label2, label3]
                and conf1 < 0.55
                and abs(conf1 - conf2) < 0.10
            ):
                proposed = "disgust"

            if proposed != "neutral" and gap < GAP_MIN:
                proposed = current_label

            # HOLD frames
            if proposed == current_label:
                pending_label = None
                pending_count = 0
            else:
                if pending_label != proposed:
                    pending_label = proposed
                    pending_count = 1
                else:
                    pending_count += 1
                if pending_count >= HOLD_FRAMES:
                    current_label = proposed
                    pending_label = None
                    pending_count = 0

            debug_txt = f"{label1}:{conf1*100:.0f}% {label2}:{conf2*100:.0f}% {label3}:{conf3*100:.0f}% | FPS:{fps:.1f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{current_label}  ({debug_txt})",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

    else:
        pred_q.clear()
        cv2.putText(frame, f"No face | FPS:{fps:.1f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Emotion Detector (FER2013) - Stable", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
