import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time
import os

# =========================
# Config
# =========================
MODEL_PATH = "FER2013_best_modelV2.keras"   # ✅ ไฟล์ .keras
CAM_INDEX = 0

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

SMOOTH = 6
HOLD_FRAMES = 3
pred_q = deque(maxlen=SMOOTH)

current_label = "neutral"
pending_label = None
pending_count = 0

PREPROCESS_MODE = "rescale"  # "rescale" หรือ "zscore"

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

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# Load Keras model (.keras)
# =========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ ไม่พบไฟล์โมเดล: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Loaded Keras model:", MODEL_PATH)

# ตรวจจำนวนคลาสจาก output
dummy = tf.zeros([1, 48, 48, 1], tf.float32)
out = model(dummy, training=False).numpy()
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
# Helper
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
    # ถ้าไม่ใกล้ 1 -> logits -> softmax
    if not (0.9 <= s <= 1.1):
        e = np.exp(vec - np.max(vec))
        vec = e / (np.sum(e) + 1e-8)
    return vec

def pick_biggest_face(faces):
    if len(faces) == 0:
        return None
    areas = [w*h for (x,y,w,h) in faces]
    return faces[int(np.argmax(areas))]

# =========================
# Webcam
# =========================
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("❌ เปิดกล้องไม่ได้ ลอง CAM_INDEX=1 หรือ 2")

print("✅ Camera opened. Press 'q' to quit.")

prev_t = time.time()
fps = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS
    now = time.time()
    dt = now - prev_t
    prev_t = now
    fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(90, 90)
    )

    face = pick_biggest_face(faces)

    if face is not None:
        x, y, w, h = face

        pad = int(0.15 * w)
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(gray.shape[1], x + w + pad); y2 = min(gray.shape[0], y + h + pad)

        roi = gray[y1:y2, x1:x2]
        roi = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)

        inp = preprocess_face(roi)

        # ✅ inference ด้วย Keras model โดยตรง
        vec = model(inp, training=False).numpy()[0]
        probs = to_probs(vec)

        pred_q.append(probs)
        p = np.mean(pred_q, axis=0)

        # top3
        top3 = p.argsort()[-3:][::-1]
        i1, i2, i3 = [int(i) for i in top3]
        label1, label2, label3 = [emotion_labels[i] for i in top3]
        conf1, conf2, conf3 = float(p[i1]), float(p[i2]), float(p[i3])
        gap = conf1 - conf2

        # proposed (เริ่มจาก top1)
        proposed = label1 if conf1 >= THRESH.get(label1, 0.18) else "neutral"

        # ✅ DISGUST RESCUE RULE (อย่าให้มี proposed ซ้ำทับ)
        # 🔥 ANGRY OVERRIDE (แก้ sad กลืน angry)
        if (
            label1 == "sad"
            and label2 == "angry"
            and conf1 > 0.80          # sad มั่นใจผิด
            and conf2 > 0.02          # angry มีสัญญาณอยู่
        ):
            proposed = "angry"
            USE_CLAHE = True

        if (
            "disgust" in [label1, label2, label3]
            and conf1 < 0.55
            and abs(conf1 - conf2) < 0.10
        ):
            proposed = "disgust"

        # anti-bounce (ยกเว้น neutral)
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

    cv2.imshow("Emotion Detector (FER2013)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
