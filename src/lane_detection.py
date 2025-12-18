# src/lane_detection.py

import os
import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K


# ---------- Metrics (needed to load the .h5) ----------

def dsc(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2.0 * (precision * recall) / (precision + recall + K.epsilon())


def dice_loss(y_true, y_pred):
    return 1.0 - dsc(y_true, y_pred)


def IOU(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    thresh = 0.5
    y_true = K.cast(K.greater_equal(y_true, thresh), "float32")
    y_pred = K.cast(K.greater_equal(y_pred, thresh), "float32")

    union = K.sum(K.maximum(y_true, y_pred)) + K.epsilon()
    intersection = K.sum(K.minimum(y_true, y_pred)) + K.epsilon()
    return intersection / union


# ---------- Helper classes & paths ----------

class Lanes:
    """Keeps a short history of predictions for temporal smoothing."""

    def __init__(self, max_history=5):
        self.max_history = max_history
        self.recent_fit = []

    def update(self, prediction):
        self.recent_fit.append(prediction)
        if len(self.recent_fit) > self.max_history:
            self.recent_fit = self.recent_fit[1:]

    @property
    def avg_fit(self):
        if not self.recent_fit:
            return None
        return np.mean(np.array(self.recent_fit), axis=0)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "LLDNet.h5")
INPUT_VIDEO = os.path.join(ROOT, "data", "Video.mp4")   # change to j.mp4 / road1.mp4 if you like
OUTPUT_VIDEO = os.path.join(ROOT, "outputs", "Predicted Video.mp4")


# ---------- Main inference loop ----------

def main():
    # Load model
    model = load_model(
        MODEL_PATH,
        custom_objects={
            "dice_loss": dice_loss,
            "IOU": IOU,
            "dsc": dsc,
            "precision_m": precision_m,
            "recall_m": recall_m,
            "f1_m": f1_m,
        },
    )

    lanes = Lanes(max_history=5)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Error: could not open input video: {INPUT_VIDEO}")
        return

    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 20.0, (width, height))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream or cannot read frame. Exiting.")
                break

            # Preprocess
            small_img = cv2.resize(frame, (160, 80))
            small_img = np.expand_dims(small_img.astype(np.float32), axis=0)

            # Inference
            start = time.time()
            prediction = model.predict(small_img, verbose=0)[0] * 255.0
            elapsed = time.time() - start
            fps = int(1.0 / elapsed) if elapsed > 0 else 0
            print(f"FPS: {fps}")

            # Temporal smoothing
            lanes.update(prediction)
            avg_pred = lanes.avg_fit
            if avg_pred is None:
                avg_pred = prediction

            # Build lane mask (G channel)
            blanks = np.zeros_like(avg_pred).astype(np.uint8)
            lane_drawn = np.dstack((blanks, avg_pred, blanks))

            # Resize back to frame size
            lane_image = cv2.resize(lane_drawn, (width, height)).astype(np.uint8)
            frame_resized = cv2.resize(frame, (width, height))

            # Overlay
            result = cv2.addWeighted(frame_resized, 1.0, lane_image, 1.0, 0)

            out.write(result)
            cv2.imshow("LLDNet Lane Detection", result)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
