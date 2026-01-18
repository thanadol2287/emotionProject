import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time
import os

# =========================
# Config
# =========================
MODEL_PATH = "FER2013_best_model V3.keras"   # .keras (ไฟล์อยู่โฟลเดอร์เดียวกับสคริปต์)
CAM_INDEX = 0

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# =========================
# Smoothing + Stability tuning
# =========================
SMOOTH = 6                 # moving average window
pred_q = deque(maxlen=SMOOTH)

PROB_EMA = 0.75            # EMA บนเวกเตอร์ prob (ยิ่งสูงยิ่งนิ่ง)
SWITCH_MARGIN = 0.05       # ต้องชนะอันดับ 2 ห่างพอ ถึงจะสลับ
GAP_MIN = 0.03             # กันการสลับตอนคะแนนใกล้กันมาก

# เข้า/ออก คนละ threshold (Schmitt trigger)
ENTER_THRESH = {
    "happy": 0.14,
    "surprise": 0.18,
    "sad": 0.22,
    "neutral": 0.00,
    "angry": 0.20,
    "fear": 0.20,
    "disgust": 0.14
}
EXIT_THRESH = {
    "happy": 0.08,
    "surprise": 0.10,
    "sad": 0.12,
    "neutral": 0.00,
    "angry": 0.10,
    "fear": 0.12,
    "disgust": 0.06
}

# HOLD แยกตามคลาส (angry/disgust ให้หนึบขึ้น)
HOLD = {
    "angry": 5,
    "disgust": 6,
    "fear": 4,
    "sad": 3,
    "happy": 3,
    "surprise": 3,
    "neutral": 2
}

current_label = "neutral"
pending_label = None
pending_count = 0

# preprocessing
PREPROCESS_MODE = "rescale"  # "rescale" or "zscore"

# =========================
# Face stability config
# =========================
DETECT_EVERY = 6           # run cascade every N frames
TRACKER_TYPE = "CSRT"      # "CSRT" (best) or "KCF" (faster)
BBOX_SMOOTH = 0.75         # EMA: closer to 1 = smoother (less jitter)
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
    """Create an OpenCV tracker if available (needs opencv-contrib-python)."""
    t = TRACKER_TYPE.upper()

    def _get(name):
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
    if prev is None:
        return new
    px, py, pw, ph = prev
    nx, ny, nw, nh = new
    x = int(a * px + (1 - a) * nx)
    y = int(a * py + (1 - a) * ny)
    w = int(a * pw + (1 - a) * nw)
    h = int(a * ph + (1 - a) * nh)
    return (x, y, w, h)

def reset_state():
    """Reset prediction state when face is lost / ROI invalid."""
    global pending_label, pending_count, p_ema
    pred_q.clear()
    pending_label = None
    pending_count = 0
    p_ema = None

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

# EMA prob state
p_ema = None

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
                reset_state()

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
            reset_state()
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

            # --- EMA smoothing on probability vector ---
            if p_ema is None:
                p_ema = p.copy()
            else:
                p_ema = PROB_EMA * p_ema + (1.0 - PROB_EMA) * p

            top3 = p_ema.argsort()[-3:][::-1]
            i1, i2, i3 = [int(i) for i in top3]
            label1, label2, label3 = [emotion_labels[i] for i in top3]
            conf1, conf2, conf3 = float(p_ema[i1]), float(p_ema[i2]), float(p_ema[i3])
            gap = conf1 - conf2

            # current prob
            cur_idx = emotion_labels.index(current_label) if current_label in emotion_labels else None
            cur_p = float(p_ema[cur_idx]) if cur_idx is not None else 0.0

            # --- Schmitt trigger decision ---
            want_switch = (conf1 >= ENTER_THRESH.get(label1, 0.18)) and (gap >= SWITCH_MARGIN)
            stay_ok = (cur_p >= EXIT_THRESH.get(current_label, 0.0))

            if current_label != "neutral" and stay_ok:
                proposed = current_label
            else:
                proposed = label1 if want_switch else "neutral"

            # --- Anti sad swallowing angry ---
            if (
                label1 == "sad" and label2 == "angry"
                and conf2 >= ENTER_THRESH.get("angry", 0.20)
                and gap < 0.12
            ):
                proposed = "angry"

            # --- Disgust rescue ---
            if "disgust" in [label1, label2, label3] and "disgust" in emotion_labels:
                d_idx = emotion_labels.index("disgust")
                d_p = float(p_ema[d_idx])
                if d_p >= ENTER_THRESH.get("disgust", 0.14) and (abs(d_p - conf1) <= 0.10):
                    proposed = "disgust"

            # --- GAP guard ---
            if proposed != "neutral" and gap < GAP_MIN:
                proposed = current_label

            # --- HOLD frames (per-class) ---
            need_hold = HOLD.get(proposed, 3)

            if proposed == current_label:
                pending_label = None
                pending_count = 0
            else:
                if pending_label != proposed:
                    pending_label = proposed
                    pending_count = 1
                else:
                    pending_count += 1

                if pending_count >= need_hold:
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
        reset_state()
        cv2.putText(frame, f"No face | FPS:{fps:.1f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Emotion Detector (FER2013) - Stable", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
