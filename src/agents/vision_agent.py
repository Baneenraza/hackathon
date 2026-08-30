"""VisionAgent - casting defect detection from an image.

run(image) -> {"defect": bool, "severity": float, "confidence": float}
`image` may be a file path, a PIL image, or an HxWx3 uint8 numpy array.
Optionally returns a Grad-CAM heatmap when return_cam=True.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

_CFG = os.path.join(C.MODELS, "casting_cnn_config.json")
_MODEL = os.path.join(C.MODELS, "casting_cnn.keras")


class VisionAgent:
    def __init__(self):
        import tensorflow as tf
        self.tf = tf
        self.cfg = json.load(open(_CFG))
        self.img = self.cfg["img"]
        self.model = tf.keras.models.load_model(_MODEL)
        self.pre = tf.keras.applications.mobilenet_v2.preprocess_input

    def _load(self, image):
        tf = self.tf
        if isinstance(image, str):
            raw = tf.keras.utils.load_img(image, target_size=(self.img, self.img))
            arr = tf.keras.utils.img_to_array(raw)
        elif hasattr(image, "size") and not isinstance(image, np.ndarray):  # PIL
            raw = image.convert("RGB").resize((self.img, self.img))
            arr = np.asarray(raw, dtype="float32")
        else:
            arr = np.asarray(image, dtype="float32")
            if arr.shape[:2] != (self.img, self.img):
                arr = tf.image.resize(arr, (self.img, self.img)).numpy()
        return arr

    def run(self, image, return_cam=False):
        arr = self._load(image)
        x = self.pre(np.expand_dims(arr.copy(), 0))
        prob_def = float(self.model.predict(x, verbose=0).ravel()[0])  # P(defect)
        defect = prob_def >= 0.5
        # severity: how deep into the defect region the score sits (0 at threshold)
        severity = float(np.clip((prob_def - 0.5) / 0.5, 0.0, 1.0)) if defect else 0.0
        confidence = float(max(prob_def, 1.0 - prob_def))
        out = {"defect": bool(defect),
               "severity": round(severity, 4),
               "confidence": round(confidence, 4),
               "p_defect": round(prob_def, 4)}
        if return_cam:
            out["cam"] = self._gradcam(arr)
        return out

    # Grad-CAM on the last MobileNetV2 conv block (head = GAP -> Dropout -> Dense)
    def _gradcam(self, arr, last_conv="out_relu"):
        tf = self.tf
        base = self.model.get_layer("mobilenetv2_1.00_160")
        dense = self.model.layers[-1]
        W, b = dense.get_weights()
        W = tf.constant(W, tf.float32)
        conv_model = tf.keras.models.Model(base.inputs,
                                           base.get_layer(last_conv).output)
        x = self.pre(np.expand_dims(arr.copy(), 0))
        with tf.GradientTape() as tape:
            conv = conv_model(x, training=False)
            pooled = tf.reduce_mean(conv, axis=(1, 2))
            logit = tf.matmul(pooled, W) + b
        grads = tape.gradient(logit, conv)[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = tf.reduce_sum(conv[0] * weights, axis=-1)
        cam = tf.maximum(cam, 0)
        cam = (cam / (tf.reduce_max(cam) + 1e-8)).numpy()
        return cam


if __name__ == "__main__":
    import glob
    va = VisionAgent()
    for cls in ["def_front", "ok_front"]:
        p = sorted(glob.glob(os.path.join(C.CASTING_DIR, "test", cls, "*.jpeg")))[0]
        print(cls, "->", va.run(p))
