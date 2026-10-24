import os
import time

import cv2
import numpy as np

import config
from utils import (
    LandmarkExtractor,
    preprocess_frame,
    extract_keypoints,
    draw_landmarks
)

LABEL = "maaf"
PERSON = "kezia"

TARGET_SEQUENCES = 40
MIN_HAND_FRAMES = 27
MIN_POSE_FRAMES = 27

COUNTDOWN_SECONDS = 3
DATASET_DIR = os.path.join(config.DATA_PATH, LABEL)

os.makedirs(DATASET_DIR, exist_ok=True)

def existing_recordings():
    prefix = f"{PERSON}_"
    return sorted(
        f for f in os.listdir(DATASET_DIR)
        if f.startswith(prefix)
        and f.endswith(".npy")
    )


def next_index():
    numbers = []

    for filename in existing_recordings():
        try:
            number = int(
                filename[len(PERSON) + 1:-4]
            )
            numbers.append(number)
        except ValueError:
            continue

    return max(numbers, default=-1) + 1

def put_text(frame, text, y, color=(255, 255, 255)):
    cv2.putText(
        frame,
        text,
        (25, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        4
    )

    cv2.putText(
        frame,
        text,
        (25, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2
    )

def validate_sequence(sequence):
    data = np.asarray(sequence, dtype=np.float32)

    if data.shape != (config.SEQUENCE_LENGTH, 147):
        return False, "Incorrect sequence shape"

    if not np.isfinite(data).all():
        return False, "NaN or Infinity detected"

    pose = data[:, :21]
    hands = data[:, 21:147]

    pose_frames = int(
        np.any(pose != 0, axis=1).sum()
    )

    hand_frames = int(
        np.any(hands != 0, axis=1).sum()
    )

    if hand_frames < MIN_HAND_FRAMES:
        return False, f"Hand detected {hand_frames}/30"

    if pose_frames < MIN_POSE_FRAMES:
        return False, f"Pose detected {pose_frames}/30"

    return True, (
        f"Hand {hand_frames}/30 | "
        f"Pose {pose_frames}/30"
    )

def main():

    existing = existing_recordings()
    saved = len(existing)
    index = next_index()

    print("=================================")
    print("BISINDO DATASET COLLECTION")
    print("Gesture:", LABEL.upper())
    print("Participant:", PERSON)
    print("Existing recordings:", saved)
    print("Target recordings:", TARGET_SEQUENCES)
    print("=================================")

    if saved >= TARGET_SEQUENCES:
        print("Target already reached.")
        return

    extractor = LandmarkExtractor()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    if not cap.isOpened():
        extractor.close()
        raise RuntimeError("Webcam cannot be opened")

    attempts = 0

    try:
        while saved < TARGET_SEQUENCES:

            attempts += 1

            countdown_start = time.monotonic()

            while (
                time.monotonic() - countdown_start
                < COUNTDOWN_SECONDS
            ):

                ret, frame = cap.read()

                if not ret:
                    raise RuntimeError(
                        "Cannot read webcam"
                    )

                frame = preprocess_frame(frame)

                hand_res, pose_res = extractor.process(
                    frame
                )

                draw_landmarks(
                    frame, hand_res, pose_res
                )

                remaining = max(
                    0,
                    COUNTDOWN_SECONDS - int(
                        time.monotonic()
                        - countdown_start
                    )
                )

                put_text(
                    frame,
                    "Gesture: MAAF",
                    40
                )

                put_text(
                    frame,
                    f"Saved: {saved}/{TARGET_SEQUENCES}",
                    80
                )

                put_text(
                    frame,
                    f"Starting in: {remaining}",
                    120,
                    (0, 255, 255)
                )

                cv2.imshow(
                    "BISINDO MAAF Collection",
                    frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by user.")
                    return

            sequence = []

            recording_start = time.monotonic()

            while len(sequence) < config.SEQUENCE_LENGTH:

                ret, frame = cap.read()

                if not ret:
                    raise RuntimeError(
                        "Cannot read webcam"
                    )

                frame = preprocess_frame(frame)

                hand_res, pose_res = extractor.process(
                    frame
                )

                keypoints = extract_keypoints(
                    hand_res, pose_res
                )

                sequence.append(keypoints)

                draw_landmarks(
                    frame, hand_res, pose_res
                )

                put_text(
                    frame,
                    "RECORDING: MAAF",
                    40,
                    (0, 0, 255)
                )

                put_text(
                    frame,
                    f"Frame: {len(sequence)}/30",
                    80
                )

                cv2.imshow(
                    "BISINDO MAAF Collection",
                    frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by user.")
                    return

            recording_duration = (
                time.monotonic() - recording_start
            )

            valid, message = validate_sequence(
                sequence
            )

            if not valid:
                print(
                    f"[REJECTED] Attempt {attempts}: "
                    f"{message}"
                )
                print("Repeating recording...")
                continue

            while True:
                filename = f"{PERSON}_{index:03d}.npy"

                filepath = os.path.join(
                    DATASET_DIR, filename
                )

                if not os.path.exists(filepath):
                    break

                index += 1

            data = np.asarray(
                sequence,
                dtype=np.float32
            )

            with open(filepath, "xb") as f:
                np.save(f, data)

            saved += 1
            index += 1

            print(
                f"[SAVED] {filepath} "
                f"({saved}/{TARGET_SEQUENCES}) | "
                f"{message} | "
                f"Duration: {recording_duration:.2f}s"
            )

    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()

    print("=================================")
    print("DATASET COLLECTION COMPLETED")
    print(f"Total saved: {saved}")
    print("=================================")


if __name__ == "__main__":
    main()
