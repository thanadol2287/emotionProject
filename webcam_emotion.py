import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# =========================
# Config
# =========================
MODEL_PATH = "emotion_savedmodel_v2"   # โฟลเดอร์ SavedModel (ต้องมี saved_model.pb)
CAM_INDEX = 0

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Smooth + Hold (ยิ่งมากยิ่งนิ่ง แต่เปลี่ยนช้า)
SMOOTH = 10
HOLD_FRAMES = 5
pred_q = deque(maxlen=SMOOTH)

current_label = "neutral"
pending_label = None
pending_count = 0

# ✅ เลือก preprocessing ให้ตรงกับตอนเทรน
# - ถ้าตอนเทรนใช้ ImageDataGenerator(rescale=1./255) => ใช้ MODE="rescale"
# - ถ้าตอนเทรน normalize แบบ z-score => ใช้ MODE="zscore"
PREPROCESS_MODE = "rescale"  # "rescale" หรือ "zscore"

# ✅ Threshold ต่อคลาส (ปรับให้ไม่โหดเกินไป)
THRESH = {
    "happy": 0.12,
    "surprise": 0.15,
    "sad": 0.18,
    "neutral": 0.00,
    "angry": 0.18,
    "fear": 0.18,
    "disgust": 0.18
}

# ✅ top1 ต้องชนะ top2 เท่าไร (กันเด้ง)
GAP_MIN = 0.02

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# Load SavedModel (once)
# =========================
loaded = tf.saved_model.load(MODEL_PATH)
infer = loaded.signatures["serving_default"]

# ดึงชื่อ output จริงแบบอัตโนมัติ (กันพังเพราะไม่ใช่ output_0)
OUTPUT_KEY = list(infer.structured_outputs.keys())[0]
print("✅ Loaded SavedModel:", MODEL_PATH)
print("✅ Output key:", OUTPUT_KEY)

# =========================
# Webcam
# =========================
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("❌ เปิดกล้องไม่ได้ ลอง CAM_INDEX=1 หรือ 2")

print("✅ Camera opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(90, 90))

    for (x, y, w, h) in faces:
        pad = int(0.15 * w)
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(gray.shape[1], x + w + pad); y2 = min(gray.shape[0], y + h + pad)

        roi = gray[y1:y2, x1:x2]
        roi = cv2.resize(roi, (48, 48)).astype("float32")

        # ---- Preprocess (ต้องตรงกับตอนเทรน) ----
        if PREPROCESS_MODE == "zscore":
            roi = (roi - np.mean(roi)) / (np.std(roi) + 1e-6)
        else:
            roi = roi / 255.0

        roi = roi.reshape(1, 48, 48, 1)

        # ---- Inference ----
        out = infer(tf.convert_to_tensor(roi, dtype=tf.float32))

        # บางโมเดล export มาเป็น logits บางโมเดลเป็น probs แล้ว
        # วิธีที่ปลอดภัย: ถ้าผลรวมไม่ใกล้ 1 ให้ softmax
        vec = out[OUTPUT_KEY].numpy()[0]
        s = float(np.sum(vec))
        probs = vec if (0.9 <= s <= 1.1) else tf.nn.softmax(vec).numpy()

        pred_q.append(probs)
        p = np.mean(pred_q, axis=0)

        # ---- Decision logic ----
        top2 = p.argsort()[-2:][::-1]
        i1, i2 = int(top2[0]), int(top2[1])
        label1, label2 = emotion_labels[i1], emotion_labels[i2]
        conf1, conf2 = float(p[i1]), float(p[i2])
        gap = conf1 - conf2

        # threshold
        proposed = label1 if conf1 >= THRESH.get(label1, 0.18) else "neutral"

        # กันเด้ง: ถ้าชนะไม่ชัดพอ ให้คงค่าเดิม
        if proposed != "neutral" and gap < GAP_MIN:
            proposed = current_label

        # HOLD: ต้องเห็น label ใหม่ติดกันหลายเฟรมก่อนเปลี่ยนจริง
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

        debug_txt = f"{label1}:{conf1*100:.0f}%  {label2}:{conf2*100:.0f}%"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{current_label}  ({debug_txt})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("Emotion Detector (FER2013)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
