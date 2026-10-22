# Real-Time BISINDO Word-Level Sign Language to Text Translation Using OpenCV

Prototipe penerjemah isyarat kata BISINDO menjadi teks secara real-time melalui webcam.
Sistem mengenali 15 kata sehari-hari ditambah kelas netral (tidak sedang berisyarat).

Kata yang dikenali: halo, tolong, terima kasih, maaf, selamat tinggal, ya, tidak,
sama-sama, apa kabar, baik, nama, siapa, makan, minum, selamat pagi.

## Cara Kerja

Webcam -> OpenCV (flip, BGR ke RGB) -> MediaPipe Hand Landmarker + Pose Landmarker
-> normalisasi koordinat terhadap bahu (147 fitur per frame) -> urutan 30 frame
-> model LSTM -> teks di layar.

| Tahap | Metode |
| --- | --- |
| Image preprocessing | OpenCV: flip horizontal, konversi BGR ke RGB |
| Hand landmark detection | MediaPipe Hand Landmarker, 21 titik per tangan, maksimal 2 tangan |
| Pose estimation | MediaPipe Pose Landmarker, 7 titik tubuh atas sebagai acuan |
| Normalisasi | Titik asal tengah bahu, skala jarak antar bahu |
| Dynamic gesture classification | LSTM 2 lapis atas urutan 30 frame landmark |

## Struktur File

| File | Fungsi |
| --- | --- |
| config.py | Daftar kata, panjang sekuens, path |
| utils.py | MediaPipe, preprocessing, ekstraksi fitur, gambar landmark |
| 1_collect_data.py | Merekam dataset lewat webcam |
| 2_train_model.py | Training model LSTM |
| 3_realtime_translate.py | Terjemahan real-time |
| dataset/ | Dataset landmark (.npy) per kata |
| models/ | Model LSTM hasil training |

## Kebutuhan

Python 3.10 sampai 3.13 (64-bit), webcam, dan koneksi internet saat pertama dijalankan
(untuk mengunduh model MediaPipe).

| Library | Versi |
| --- | --- |
| mediapipe | 0.10.32 |
| opencv-contrib-python | terbaru |
| tensorflow | 2.20.0 |
| numpy, scikit-learn, matplotlib | terbaru |

## Cara Menjalankan

1. Install library: `python -m pip install -r requirements.txt`
2. Rekam dataset: `python 1_collect_data.py`
3. Training model: `python 2_train_model.py`
4. Jalankan penerjemah: `python 3_realtime_translate.py` (tombol q keluar, c hapus teks)

## Dataset

Dataset direkam sendiri oleh anggota kelompok menggunakan 1_collect_data.py.

| Keterangan | Nilai |
| --- | --- |
| Jumlah kelas | 16 (15 kata + netral) |
| Rekaman per kelas | 40 |
| Panjang satu rekaman | 30 frame (sekitar 1 detik) |
| Fitur per frame | 147 (pose atas 7x3, tangan kiri 21x3, tangan kanan 21x3) |
| Format | dataset/<kata>/<nomor>.npy, ukuran (30, 147) |
| Pembagian data | 80% latih, 20% uji |
| Augmentasi | noise, skala, dan pergeseran posisi |

Yang disimpan hanya koordinat landmark, bukan gambar atau video.