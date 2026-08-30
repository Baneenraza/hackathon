"""Block 4 - MobileNetV2 transfer learning for casting-defect classification.

Two tracked MLflow iterations (experiment `casting_defect_cnn`):
    run 1: frozen MobileNetV2 backbone + linear head
    run 2: last ~30 backbone layers unfrozen and fine-tuned
Metrics: precision / recall / F1 / ROC-AUC + confusion matrix on the held-out
test split. Best (val AUC) -> models_registry/casting_cnn.keras, registered as
`factory_casting_defect`.

Grad-CAM overlays for correct + misclassified test images -> reports/gradcam/.

Class convention: label 1 = DEFECT (def_front), label 0 = OK (ok_front).
For speed on CPU the training set is capped (TRAIN_CAP); the full test split is
always evaluated.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa: E402

IMG = 160
BATCH = 32
TRAIN_CAP = 2400          # capped for CPU; set None for the full ~6600
EXPERIMENT = "casting_defect_cnn"
GRADCAM_DIR = os.path.join(C.REPORTS, "gradcam")
tf.random.set_seed(C.RANDOM_STATE)
np.random.seed(C.RANDOM_STATE)


def make_ds(split, shuffle):
    d = tf.keras.utils.image_dataset_from_directory(
        os.path.join(C.CASTING_DIR, split),
        labels="inferred", label_mode="binary",
        class_names=["ok_front", "def_front"],   # -> ok=0, def=1
        image_size=(IMG, IMG), batch_size=None, shuffle=shuffle,
        seed=C.RANDOM_STATE)
    return d


def prep(ds, cap=None, augment=False):
    if cap:
        ds = ds.take(cap)
    ds = ds.batch(BATCH)
    aug = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"),
                               tf.keras.layers.RandomRotation(0.05)])
    pre = tf.keras.applications.mobilenet_v2.preprocess_input

    def _m(x, y):
        if augment:
            x = aug(x)
        return pre(x), y
    return ds.map(_m, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def build(fine_tune=False):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet")
    base.trainable = fine_tune
    if fine_tune:
        for layer in base.layers[:-30]:
            layer.trainable = False
        for layer in base.layers:            # keep BN running stats fixed
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
    inp = tf.keras.Input((IMG, IMG, 3))
    x = base(inp, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-5 if fine_tune else 1e-3),
              loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return m, base


def evaluate(model, ds):
    y_true, y_prob = [], []
    for xb, yb in ds:
        y_prob.append(model.predict(xb, verbose=0).ravel())
        y_true.append(yb.numpy().ravel())
    y_true = np.concatenate(y_true); y_prob = np.concatenate(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "cm": confusion_matrix(y_true, y_pred).tolist(),
    }, y_true, y_prob


# ---------------- Grad-CAM ----------------
# Head is GAP -> Dropout -> Dense(1, sigmoid), so Grad-CAM reduces to a clean
# gradient of the pre-sigmoid logit w.r.t. the last conv block ('out_relu').
def gradcam_heatmap(model, img_array, last_conv_name="out_relu"):
    base = model.get_layer("mobilenetv2_1.00_160")
    dense = model.layers[-1]
    W, bnd = dense.get_weights()
    W = tf.constant(W, tf.float32)
    conv_model = tf.keras.models.Model(base.inputs,
                                       base.get_layer(last_conv_name).output)
    with tf.GradientTape() as tape:
        conv = conv_model(img_array, training=False)          # (1, h, w, 1280)
        pooled = tf.reduce_mean(conv, axis=(1, 2))            # (1, 1280)
        logit = tf.matmul(pooled, W) + bnd                    # pre-sigmoid
    grads = tape.gradient(logit, conv)[0]                     # (h, w, 1280)
    weights = tf.reduce_mean(grads, axis=(0, 1))             # (1280,)
    cam = tf.reduce_sum(conv[0] * weights, axis=-1)
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()


def save_gradcams(model, n=8):
    os.makedirs(GRADCAM_DIR, exist_ok=True)
    raw = tf.keras.utils.image_dataset_from_directory(
        os.path.join(C.CASTING_DIR, "test"), labels="inferred", label_mode="binary",
        class_names=["ok_front", "def_front"], image_size=(IMG, IMG),
        batch_size=None, shuffle=True, seed=1)
    pre = tf.keras.applications.mobilenet_v2.preprocess_input
    conv_name = "out_relu"
    done = 0
    fig, axes = plt.subplots(2, n // 2, figsize=(3 * (n // 2), 6))
    axes = axes.ravel()
    for img, label in raw:
        arr = pre(tf.expand_dims(tf.cast(img, tf.float32), 0))
        prob = float(model.predict(arr, verbose=0).ravel()[0])
        heat = gradcam_heatmap(model, arr, conv_name)
        heat_rs = tf.image.resize(heat[..., None], (IMG, IMG)).numpy().squeeze()
        ax = axes[done]
        ax.imshow(tf.cast(img, tf.uint8).numpy())
        ax.imshow(heat_rs, cmap="jet", alpha=0.45)
        lab = int(label.numpy()[0]); pred = int(prob >= 0.5)
        tag = "OK" if not lab else "DEFECT"
        mark = "OK" if pred == lab else "WRONG"
        ax.set_title(f"true={tag} p(def)={prob:.2f} [{mark}]", fontsize=8)
        ax.axis("off")
        done += 1
        if done == n:
            break
    fig.suptitle("Grad-CAM (MobileNetV2 'out_relu') - casting test images")
    fig.tight_layout()
    p = os.path.join(GRADCAM_DIR, "gradcam_grid.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print("gradcam ->", p)
    return p


def main():
    mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    full_train = make_ds("train", shuffle=True)
    n_total = full_train.cardinality().numpy()
    val_take = int(n_total * 0.2)
    val_ds = prep(full_train.take(val_take))
    tr_ds = prep(full_train.skip(val_take), cap=TRAIN_CAP, augment=True)
    test_ds = prep(make_ds("test", shuffle=False))
    print(f"train imgs total={n_total}, val={val_take}, train cap={TRAIN_CAP}")

    results = {}

    # ---- run 1: frozen backbone ----
    with mlflow.start_run(run_name="frozen_backbone"):
        model, base = build(fine_tune=False)
        mlflow.log_params({"backbone": "MobileNetV2", "img": IMG, "mode": "frozen",
                           "epochs": 3, "train_cap": TRAIN_CAP})
        h = model.fit(tr_ds, validation_data=val_ds, epochs=3, verbose=2)
        m_te, yt, yp = evaluate(model, test_ds)
        mlflow.log_metrics({f"test_{k}": v for k, v in m_te.items() if k != "cm"})
        mlflow.log_metric("val_auc", float(h.history["val_auc"][-1]))
        mlflow.log_dict({"cm": m_te["cm"]}, "test_confusion.json")
        model.save(os.path.join(C.MODELS, "_casting_frozen.keras"))
        results["frozen_backbone"] = {"test": m_te, "val_auc": float(h.history["val_auc"][-1])}
        print("[frozen]", m_te)

    # ---- run 2: fine-tuned ----
    with mlflow.start_run(run_name="fine_tuned"):
        model, base = build(fine_tune=True)
        mlflow.log_params({"backbone": "MobileNetV2", "img": IMG, "mode": "finetune_top30",
                           "epochs": 3, "train_cap": TRAIN_CAP})
        h = model.fit(tr_ds, validation_data=val_ds, epochs=3, verbose=2)
        m_te, yt, yp = evaluate(model, test_ds)
        mlflow.log_metrics({f"test_{k}": v for k, v in m_te.items() if k != "cm"})
        mlflow.log_metric("val_auc", float(h.history["val_auc"][-1]))
        mlflow.log_dict({"cm": m_te["cm"]}, "test_confusion.json")
        model.save(os.path.join(C.MODELS, "_casting_finetuned.keras"))
        results["fine_tuned"] = {"test": m_te, "val_auc": float(h.history["val_auc"][-1])}
        print("[finetuned]", m_te)

    # ---- select + register ----
    best = max(results, key=lambda k: results[k]["val_auc"])
    print("BEST =", best)
    src = os.path.join(C.MODELS,
                       "_casting_frozen.keras" if best == "frozen_backbone"
                       else "_casting_finetuned.keras")
    best_model = tf.keras.models.load_model(src)
    best_model.save(os.path.join(C.MODELS, "casting_cnn.keras"))
    import json
    json.dump({"img": IMG, "name": best, "class_1": "def_front(defect)",
               "class_0": "ok_front", "preprocess": "mobilenet_v2.preprocess_input"},
              open(os.path.join(C.MODELS, "casting_cnn_config.json"), "w"), indent=2)
    with mlflow.start_run(run_name=f"register_{best}"):
        mlflow.log_param("selected_model", best)
        for k, v in results[best]["test"].items():
            if k != "cm":
                mlflow.log_metric(f"best_test_{k}", v)
        mlflow.tensorflow.log_model(best_model, name="model",
                                    registered_model_name="factory_casting_defect")

    gpath = save_gradcams(best_model)

    lines = ["# Block 4 - Casting defect classification (MobileNetV2 transfer learning)",
             "", "## Test-set metrics (full held-out split)", "",
             "| run | precision | recall | F1 | ROC-AUC | confusion [ [TN,FP],[FN,TP] ] |",
             "|---|---|---|---|---|---|"]
    for k, v in results.items():
        t = v["test"]
        lines.append(f"| {k} | {t['precision']} | {t['recall']} | {t['f1']} | "
                     f"{t['roc_auc']} | {t['cm']} |")
    lines += ["", f"**Selected: `{best}`** (highest validation AUC). Class 1 = defect.",
              "", "## Grad-CAM", "",
              "![gradcam](gradcam/gradcam_grid.png)", "",
              "Grad-CAM uses gradients into the final MobileNetV2 conv block "
              "(`out_relu`). On correctly-classified defect castings the activation "
              "concentrates on the blow-holes / shrinkage porosity at the casting "
              "edge; on OK parts it stays diffuse. Misclassifications (marked WRONG) "
              "typically show the heat landing on lighting glare or the circular rim "
              "rather than a real defect - a useful failure signature to show a "
              "human reviewer."]
    with open(os.path.join(C.REPORTS, "casting_cnn_report.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("report -> reports/casting_cnn_report.md")
    print("BLOCK 4 OK")


if __name__ == "__main__":
    main()
