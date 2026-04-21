"""
Task 2.8 — Grad-CAM module.

Usage:
    from gradcam import GradCAM
    cam = GradCAM(model)
    heatmap  = cam.compute(image_array)          # (224, 224) float32 in [0,1]
    overlay  = cam.overlay(original_image, heatmap)
    cam.save(overlay, "output/gradcam_result.png")
"""

import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


class GradCAM:
    def __init__(self, model: tf.keras.Model, layer_name: str = None):
        self.model = model

        # Find the base submodel (MobileNetV3Small) and head layers
        self.base_model = None
        base_idx = None
        for i, layer in enumerate(model.layers):
            if hasattr(layer, "layers"):
                self.base_model = layer
                base_idx = i
                break

        if self.base_model is None:
            raise ValueError("Could not find base submodel in model.")

        # Head layers = everything after the base model
        self.head_layers = model.layers[base_idx + 1:]
        self.layer_name = self.base_model.name
        print(f"Grad-CAM: base={self.layer_name}, head layers={len(self.head_layers)}")

    def compute(self, image: np.ndarray, class_idx: int = None) -> np.ndarray:
        """
        Args:
            image: float32 array shape (224, 224, 3), values in [0, 255]
            class_idx: target class (None = predicted class)
        Returns:
            heatmap: float32 array shape (224, 224), values in [0, 1]
        """
        img_tensor = tf.cast(tf.expand_dims(image, axis=0), tf.float32)

        # Step 1: get spatial features from base model
        base_out = self.base_model(img_tensor, training=False)

        # Step 2: wrap as Variable so tape tracks gradients through head layers
        feature_var = tf.Variable(base_out, trainable=True)

        if class_idx is None:
            pred_full = self.model(img_tensor, training=False)
            class_idx = int(tf.argmax(pred_full[0]))

        with tf.GradientTape() as tape:
            x = feature_var
            for layer in self.head_layers:
                x = layer(x, training=False)
            loss = x[:, class_idx]

        grads        = tape.gradient(loss, feature_var)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_out = feature_var[0]
        heatmap  = conv_out @ pooled_grads[..., tf.newaxis]
        heatmap  = tf.squeeze(heatmap).numpy()

        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        heatmap = cv2.resize(heatmap, (224, 224))
        return heatmap.astype(np.float32)

    def overlay(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Args:
            original_image: uint8 BGR array (224, 224, 3) — as loaded by cv2
                            OR float32 RGB [0,1] array — auto-detected
            heatmap: float32 (224, 224) from compute()
            alpha: heatmap blend strength
        Returns:
            overlay: uint8 BGR array (224, 224, 3)
        """
        # Normalise original to uint8 BGR
        if original_image.dtype != np.uint8:
            img = np.clip(original_image, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = original_image.copy()

        # Apply color map to heatmap
        heatmap_uint8  = np.uint8(255 * heatmap)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        result = cv2.addWeighted(colored_heatmap, alpha, img, 1 - alpha, 0)
        return result

    @staticmethod
    def save(overlay: np.ndarray, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(path, overlay)
        print(f"Grad-CAM overlay saved -> {path}")


# ── CLI demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    ROOT = Path(__file__).parent

    model_path = ROOT / "models" / "best_phase2.h5"
    if not model_path.exists():
        model_path = ROOT / "models" / "model.h5"
    if not model_path.exists():
        raise SystemExit("No trained model found. Run train_model.py first.")

    with open(ROOT / "labels.json") as f:
        label_map = json.load(f)

    print(f"Loading {model_path} ...")
    model = tf.keras.models.load_model(str(model_path))

    cam = GradCAM(model)

    # Pick a random test image
    import pandas as pd
    test_df = pd.read_csv(ROOT / "data" / "splits" / "test.csv")
    row = test_df.sample(1, random_state=7).iloc[0]
    fp, true_label = row["filepath"], int(row["label"])

    # Load and preprocess
    img_bgr   = cv2.imread(fp)
    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = cv2.resize(img_rgb, (224, 224)).astype(np.float32)  # keep [0,255]

    heatmap = cam.compute(img_float)
    overlay = cam.overlay(img_float, heatmap)

    out_dir = ROOT / "logs"
    out_dir.mkdir(exist_ok=True)
    cam.save(overlay, str(out_dir / "gradcam_sample.png"))

    # Print prediction
    pred = model.predict(np.expand_dims(img_float, 0), verbose=0)[0]
    pred_idx = int(np.argmax(pred))
    print(f"True:      {label_map[str(true_label)]}")
    print(f"Predicted: {label_map[str(pred_idx)]}  ({pred[pred_idx]*100:.1f}%)")
