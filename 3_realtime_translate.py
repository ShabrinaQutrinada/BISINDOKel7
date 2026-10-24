import json
from collections import deque, Counter

import cv2
import numpy as np
import tensorflow as tf

import config
from utils import (LandmarkExtractor, preprocess_frame, extract_keypoints,
                   draw_landmarks, hands_detected)

# ===== Pengaturan pengenalan =====
CONFIDENCE_THRESHOLD = 0.80  # probabilitas minimum agar kata diterima
STABLE_WINDOW = 10           # jumlah prediksi terakhir yang dicek
STABLE_MIN = 8               # minimal sekian prediksi harus sama agar dianggap stabil
COOLDOWN_FRAMES = 20         # jeda setelah kata muncul sebelum kata berikutnya
PREDICT_EVERY = 2            # prediksi tiap 2 frame agar lebih ringan
MAX_WORDS_SHOWN = 5


def put_text(frame, text, pos, scale=0.8, color=(255, 255, 255), thick=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def main():
    model = tf.keras.models.load_model(config.MODEL_PATH)
    with open(config.LABELS_PATH) as f:
        labels = json.load(f)
    print("Kelas:", labels)

    extractor = LandmarkExtractor()
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cv2.namedWindow("BISINDO to Text", cv2.WINDOW_NORMAL)

    sequence = deque(maxlen=config.SEQUENCE_LENGTH)
    recent = deque(maxlen=STABLE_WINDOW)
    sentence = []
    frame_no = 0
    cooldown = 0
    current = ""

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Webcam tidak terbaca.")
            break
        frame_no += 1

        frame = preprocess_frame(frame)
        hand_res, pose_res = extractor.process(frame)
        sequence.append(extract_keypoints(hand_res, pose_res))
        draw_landmarks(frame, hand_res, pose_res)

        if cooldown > 0:
            cooldown -= 1

        if len(sequence) == config.SEQUENCE_LENGTH and frame_no % PREDICT_EVERY == 0:
            if hands_detected(hand_res):
                x = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)
                probs = model(x, training=False).numpy()[0]
                best = int(np.argmax(probs))
                recent.append(best if probs[best] >= CONFIDENCE_THRESHOLD else -1)
            else:
                recent.append(-1)

            # kata diterima jika prediksinya stabil di beberapa prediksi terakhir
            if len(recent) == STABLE_WINDOW:
                idx, count = Counter(recent).most_common(1)[0]
                if idx != -1 and count >= STABLE_MIN:
                    word = labels[idx]
                    current = config.DISPLAY_TEXT.get(word, word)
                    if word != "tidak_terdeteksi" and cooldown == 0 and (not sentence or sentence[-1] != current):
                        sentence.append(current)
                        sentence = sentence[-MAX_WORDS_SHOWN:]
                        cooldown = COOLDOWN_FRAMES
                        recent.clear()
                elif idx == -1:
                    current = ""

        # ----- tampilan -----
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 50), (40, 40, 40), -1)
        put_text(frame, " | ".join(sentence) if sentence else "(lakukan isyarat...)",
                 (10, 35), 1.0, (0, 255, 255))
        put_text(frame, f"Terdeteksi: {current}", (10, 78), 0.7, (0, 255, 0))
        put_text(frame, "q: keluar   c: hapus teks", (10, h - 15), 0.55, thick=1)

        cv2.imshow("BISINDO to Text", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            sentence = []

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()


if __name__ == "__main__":
    main()