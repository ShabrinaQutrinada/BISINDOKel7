import os

# ===== Daftar kelas (kata BISINDO) =====
ACTIONS = [
    "halo", "tolong", "terima_kasih", "maaf", "selamat_tinggal",
    "ya", "tidak", "sama_sama", "apa_kabar", "baik",
    "nama", "siapa", "makan", "minum", "selamat_pagi",
    "tidak_terdeteksi",
    ]

# Teks yang ditampilkan di layar untuk tiap kelas
DISPLAY_TEXT = {
    "halo": "Halo",
    "tolong": "Tolong",
    "terima_kasih": "Terima kasih",
    "maaf": "Maaf",
    "selamat_tinggal": "Selamat tinggal",
    "ya": "Ya",
    "tidak": "Tidak",
    "sama_sama": "Sama-sama",
    "apa_kabar": "Apa kabar",
    "baik": "Baik",
    "nama": "Nama",
    "siapa": "Siapa",
    "makan": "Makan",
    "minum": "Minum",
    "selamat_pagi": "Selamat pagi",
    "tidak_terdeteksi": "",
}

# ===== Dataset =====
SEQUENCE_LENGTH = 30   # jumlah frame per satu isyarat (~1 detik pada 30 FPS)
NUM_SEQUENCES = 40     # jumlah rekaman per kata (per orang)
DATA_PATH = "dataset"  # folder penyimpanan dataset (.npy)

# ===== Model =====
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "bisindo_lstm.keras")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

# File model MediaPipe (diunduh otomatis saat pertama kali dijalankan)
HAND_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
POSE_MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/latest/hand_landmarker.task")
POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")

# ===== Kamera =====
CAMERA_INDEX = 0
