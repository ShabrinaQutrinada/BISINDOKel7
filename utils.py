import os
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import config

# Titik pose tubuh bagian atas yang dipakai sebagai acuan posisi tangan
# 0 hidung, 11-12 bahu, 13-14 siku, 15-16 pergelangan tangan
POSE_UPPER_BODY = [0, 11, 12, 13, 14, 15, 16]

NUM_POSE_FEATURES = len(POSE_UPPER_BODY) * 3   # 21
NUM_HAND_FEATURES = 21 * 3                     # 63 per tangan
NUM_FEATURES = NUM_POSE_FEATURES + 2 * NUM_HAND_FEATURES  # 147

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # ibu jari
    (0, 5), (5, 6), (6, 7), (7, 8),          # telunjuk
    (5, 9), (9, 10), (10, 11), (11, 12),     # jari tengah
    (9, 13), (13, 14), (14, 15), (15, 16),   # jari manis
    (13, 17), (17, 18), (18, 19), (19, 20),  # kelingking
    (0, 17),
]
POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def ensure_models():
    """Mengunduh file model MediaPipe jika belum ada di folder models/."""
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    for path, url in [(config.HAND_MODEL_PATH, config.HAND_MODEL_URL),
                      (config.POSE_MODEL_PATH, config.POSE_MODEL_URL)]:
        if not os.path.exists(path):
            print(f"Mengunduh {os.path.basename(path)} ...")
            urllib.request.urlretrieve(url, path)
            print("  selesai.")


class LandmarkExtractor:
    """Menjalankan MediaPipe Hand Landmarker dan Pose Landmarker pada setiap frame."""

    def __init__(self):
        ensure_models()
        hand_options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=config.HAND_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        pose_options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=config.POSE_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands = vision.HandLandmarker.create_from_options(hand_options)
        self.pose = vision.PoseLandmarker.create_from_options(pose_options)
        self._last_ts = 0

    def _timestamp_ms(self):
        # Mode VIDEO mewajibkan timestamp yang selalu naik
        ts = int(time.monotonic() * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        return ts

    def process(self, frame_bgr):
        """Frame BGR (sudah di-flip) -> (hasil tangan, hasil pose)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # konversi warna BGR -> RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = self._timestamp_ms()
        hand_result = self.hands.detect_for_video(mp_image, ts)
        pose_result = self.pose.detect_for_video(mp_image, ts)
        return hand_result, pose_result

    def close(self):
        self.hands.close()
        self.pose.close()


def preprocess_frame(frame):
    """Flip horizontal agar tampilan seperti cermin (selfie view)."""
    return cv2.flip(frame, 1)


def extract_keypoints(hand_result, pose_result):
    """
    Mengubah hasil MediaPipe menjadi satu vektor fitur berukuran 147:
    [pose atas (7x3), tangan kiri (21x3), tangan kanan (21x3)].

    Koordinat dinormalisasi terhadap tubuh pengguna:
    - titik asal = titik tengah kedua bahu
    - skala      = jarak antar bahu
    Sehingga hasilnya tidak bergantung pada posisi duduk atau jarak ke kamera.
    Tangan yang tidak terdeteksi diisi nol.
    """
    origin = np.array([0.5, 0.5], dtype=np.float32)
    scale = 0.25

    pose = np.zeros((len(POSE_UPPER_BODY), 3), dtype=np.float32)
    if pose_result.pose_landmarks:
        lm = pose_result.pose_landmarks[0]
        l_sh = np.array([lm[11].x, lm[11].y], dtype=np.float32)
        r_sh = np.array([lm[12].x, lm[12].y], dtype=np.float32)
        origin = (l_sh + r_sh) / 2
        scale = max(float(np.linalg.norm(l_sh - r_sh)), 1e-3)
        for i, idx in enumerate(POSE_UPPER_BODY):
            pose[i] = [(lm[idx].x - origin[0]) / scale,
                       (lm[idx].y - origin[1]) / scale,
                       lm[idx].z / scale]

    left = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)
    for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
        pts = np.array([[(p.x - origin[0]) / scale,
                         (p.y - origin[1]) / scale,
                         p.z / scale] for p in landmarks], dtype=np.float32)
        if handedness[0].category_name == "Left":
            left = pts
        else:
            right = pts

    return np.concatenate([pose.flatten(), left.flatten(), right.flatten()])


def hands_detected(hand_result):
    return len(hand_result.hand_landmarks) > 0


def draw_landmarks(frame, hand_result, pose_result):
    """Menggambar kerangka tubuh atas dan tangan di atas frame."""
    h, w = frame.shape[:2]

    if pose_result.pose_landmarks:
        lm = pose_result.pose_landmarks[0]
        for a, b in POSE_CONNECTIONS:
            pa = (int(lm[a].x * w), int(lm[a].y * h))
            pb = (int(lm[b].x * w), int(lm[b].y * h))
            cv2.line(frame, pa, pb, (200, 200, 200), 2)
        for idx in POSE_UPPER_BODY:
            cv2.circle(frame, (int(lm[idx].x * w), int(lm[idx].y * h)), 4, (200, 200, 200), -1)

    for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
        color = (255, 128, 0) if handedness[0].category_name == "Left" else (0, 200, 255)
        pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], color, 2)
        for p in pts:
            cv2.circle(frame, p, 3, (255, 255, 255), -1)