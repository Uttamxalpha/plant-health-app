"""
╔══════════════════════════════════════════════════════════════╗
║   PLANT HEALTH CNN — FULL RETRAINING PIPELINE               ║
║   Dataset : kagglehub → vipoooool/new-plant-diseases-dataset ║
║   Run     : python train.py                                  ║
╚══════════════════════════════════════════════════════════════╝

What this script does, step by step:
  1.  Download dataset via kagglehub
  2.  Auto-discover dataset structure (train / valid folders)
  3.  Build tf.data pipelines with aggressive augmentation
  4.  Construct the CNN (residual + attention + depthwise sep)
  5.  Compile with cosine-decay LR + label smoothing
  6.  Train with early-stopping & model checkpointing
  7.  Evaluate on validation set
  8.  Plot & save training curves
  9.  Export model → saved_models/plant_health_final.keras
  10. Save class_names.json for the Streamlit app
"""

# ── Stdlib ──────────────────────────────────────────────────
import os, sys, json, time, shutil, warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF C++ spam

# ── Third-party ──────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use("Agg")                        # headless backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# STEP 0 — Install & import kagglehub
# ─────────────────────────────────────────────
try:
    import kagglehub
except ImportError:
    print("  Installing kagglehub…")
    os.system(f"{sys.executable} -m pip install kagglehub -q")
    import kagglehub

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, TensorBoard, CSVLogger
)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
class CFG:
    # ── Paths ───────────────────────────────
    SAVE_DIR    = Path("saved_models")
    LOG_DIR     = Path("logs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    PLOT_DIR    = Path("plots")

    # ── Image ───────────────────────────────
    IMG_H       = 224
    IMG_W       = 224
    CHANNELS    = 3
    INPUT_SHAPE = (IMG_H, IMG_W, CHANNELS)

    # ── Training ────────────────────────────
    BATCH_SIZE  = 32
    EPOCHS      = 60
    LR_INIT     = 1e-3
    LR_MIN      = 1e-7
    DROPOUT     = 0.4
    L2          = 1e-4
    LABEL_SMOOTH= 0.1

    # ── Model ───────────────────────────────
    USE_PRETRAINED = False   # True = MobileNetV2 backbone
    FINE_TUNE_AT   = -30     # unfreeze last N layers when USE_PRETRAINED

    # ── Misc ────────────────────────────────
    SEED        = 42
    AUTOTUNE    = tf.data.AUTOTUNE

cfg = CFG()
tf.random.set_seed(cfg.SEED)
np.random.seed(cfg.SEED)

for d in [cfg.SAVE_DIR, cfg.LOG_DIR, cfg.PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# STEP 1 — DOWNLOAD DATASET
# ─────────────────────────────────────────────

def download_dataset() -> Path:
    print("\n" + "═"*60)
    print("  [1/9] Downloading dataset via KaggleHub")
    print("═"*60)
    print("  Dataset : vipoooool/new-plant-diseases-dataset")
    print("  Note    : Requires Kaggle credentials (~2 GB download)")
    print("            Set KAGGLE_USERNAME & KAGGLE_KEY env vars,")
    print("            or place kaggle.json in ~/.kaggle/")
    print()

    t0 = time.time()
    path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
    elapsed = time.time() - t0

    path = Path(path)
    print(f"\n  ✅ Downloaded in {elapsed:.1f}s")
    print(f"  📁 Path: {path}")
    return path


# ─────────────────────────────────────────────
# STEP 2 — DISCOVER DATASET STRUCTURE
# ─────────────────────────────────────────────

def find_data_dirs(root: Path):
    """
    The KaggleHub dataset has this structure:
      <root>/
        New Plant Diseases Dataset(Augmented)/
          New Plant Diseases Dataset(Augmented)/
            train/
              Apple___Apple_scab/  …
            valid/
              Apple___Apple_scab/  …
    We walk the tree to find the train & valid directories.
    """
    print("\n  [2/9] Discovering dataset structure…")

    train_dir = valid_dir = None

    for p in sorted(root.rglob("*")):
        if p.is_dir():
            name = p.name.lower()
            if name == "train"  and train_dir is None:
                train_dir = p
            if name in ("valid", "val", "validation") and valid_dir is None:
                valid_dir = p

    if train_dir is None:
        raise FileNotFoundError(
            f"Could not find 'train' directory under {root}.\n"
            f"Directory tree:\n" + "\n".join(str(p) for p in root.rglob("*") if p.is_dir())
        )

    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    n_train = sum(len(list(d.glob("*.*"))) for d in train_dir.iterdir() if d.is_dir())
    n_valid = sum(len(list(d.glob("*.*"))) for d in valid_dir.iterdir() if d.is_dir()) if valid_dir else 0

    print(f"  ✅ train dir  : {train_dir}  ({n_train:,} images)")
    if valid_dir:
        print(f"  ✅ valid dir  : {valid_dir}  ({n_valid:,} images)")
    print(f"  ✅ classes    : {len(classes)}")

    return train_dir, valid_dir, classes


# ─────────────────────────────────────────────
# STEP 3 — BUILD tf.data PIPELINE
# ─────────────────────────────────────────────

def build_dataset(directory: Path, classes: list, training: bool) -> tf.data.Dataset:
    """
    Loads images from directory using keras utility, then wraps in
    a fast, cached, prefetched tf.data pipeline.

    Augmentation layers are baked into the dataset (training only)
    so augmentation runs on CPU in parallel while GPU trains.
    """
    ds = keras.utils.image_dataset_from_directory(
        str(directory),
        labels="inferred",
        label_mode="categorical",
        class_names=classes,
        image_size=(cfg.IMG_H, cfg.IMG_W),
        batch_size=cfg.BATCH_SIZE,
        shuffle=training,
        seed=cfg.SEED,
    )

    # Normalise [0,255] → [0,1]
    normalise = layers.Rescaling(1.0 / 255.0)

    # Heavy augmentation pipeline (training only)
    augment = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.25),
        layers.RandomZoom(0.20),
        layers.RandomTranslation(0.10, 0.10),
        layers.RandomContrast(0.20),
        layers.RandomBrightness(0.15),
    ], name="augmentation")

    def preprocess_train(x, y):
        x = normalise(x)
        x = augment(x, training=True)
        return x, y

    def preprocess_val(x, y):
        x = normalise(x)
        return x, y

    if training:
        ds = ds.map(preprocess_train, num_parallel_calls=cfg.AUTOTUNE)
        ds = ds.shuffle(buffer_size=1000, seed=cfg.SEED, reshuffle_each_iteration=True)
        ds = ds.repeat()
    else:
        ds = ds.map(preprocess_val, num_parallel_calls=cfg.AUTOTUNE)

    ds = ds.prefetch(cfg.AUTOTUNE)
    return ds


# ─────────────────────────────────────────────
# STEP 4 — MODEL ARCHITECTURE
# ─────────────────────────────────────────────

# ── Building-block helpers ───────────────────

def conv_bn_relu(x, filters, kernel=3, strides=1):
    x = layers.Conv2D(
        filters, kernel, strides=strides, padding="same",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(cfg.L2),
        kernel_initializer="he_normal",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def residual_block(x, filters):
    """Pre-activation residual block."""
    shortcut = x
    in_ch    = x.shape[-1]

    # Main path
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(cfg.L2))(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(cfg.L2))(x)

    # Project shortcut if channel dims differ
    if in_ch != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False)(shortcut)

    return layers.Add()([x, shortcut])


def depthwise_sep_block(x, filters):
    """MobileNet-style depthwise separable conv block."""
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False,
                                depthwise_regularizer=keras.regularizers.l2(cfg.L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 1, padding="same", use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(cfg.L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def squeeze_excite(x, ratio=16):
    """Channel Attention — Squeeze-and-Excitation block."""
    ch  = x.shape[-1]
    gap = layers.GlobalAveragePooling2D(keepdims=True)(x)
    se  = layers.Dense(max(ch // ratio, 8), activation="relu",  use_bias=False)(gap)
    se  = layers.Dense(ch,                  activation="sigmoid", use_bias=False)(se)
    return layers.Multiply()([x, se])


def build_model(num_classes: int) -> Model:
    """
    PlantHealthCNN Architecture
    ───────────────────────────────────────────────────────
    Input  224×224×3
    Stage1  Stem  Conv(32, 3×3, s=2) + Conv(32) + MaxPool  → 56×56×32
    Stage2  2× Conv(64)  + MaxPool                          → 28×28×64
    Stage3  2× ResBlock(128) + MaxPool                      → 14×14×128
    Stage4  2× ResBlock(256) + MaxPool                      →  7×7×256
    Stage5  2× DWSep(512)                                   →  7×7×512
    Stage6  SE Channel Attention                            →  7×7×512
    Stage7  GlobalAveragePool                               →  512
    Head    Dense(512)→BN→Drop(0.4)→Dense(256)→Drop(0.2)  →  256
    Output  Dense(num_classes, softmax)                     →  N
    """
    if cfg.USE_PRETRAINED:
        return _build_transfer_model(num_classes)

    inp = keras.Input(shape=cfg.INPUT_SHAPE, name="leaf_image")
    x   = inp

    # ── Stage 1: Stem ──────────────────────────────────────
    x = conv_bn_relu(x, 32, kernel=3, strides=2)    # 112×112
    x = conv_bn_relu(x, 32, kernel=3)
    x = layers.MaxPooling2D(2, strides=2)(x)         # 56×56

    # ── Stage 2: Feature Extraction ────────────────────────
    x = conv_bn_relu(x, 64, kernel=3)
    x = conv_bn_relu(x, 64, kernel=3)
    x = layers.MaxPooling2D(2, strides=2)(x)         # 28×28

    # ── Stage 3: Residual Blocks ───────────────────────────
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = layers.MaxPooling2D(2, strides=2)(x)         # 14×14

    # ── Stage 4: Deep Residual Blocks ──────────────────────
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = layers.MaxPooling2D(2, strides=2)(x)         #  7×7

    # ── Stage 5: Lightweight Depthwise Separable ───────────
    x = depthwise_sep_block(x, 512)
    x = depthwise_sep_block(x, 512)

    # ── Stage 6: Channel Attention ─────────────────────────
    x = squeeze_excite(x, ratio=16)

    # ── Stage 7: Global Pooling ────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # ── Classifier Head ────────────────────────────────────
    x = layers.Dense(512, use_bias=False,
                     kernel_regularizer=keras.regularizers.l2(cfg.L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(cfg.DROPOUT)(x)

    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(cfg.L2))(x)
    x = layers.Dropout(cfg.DROPOUT / 2)(x)

    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return Model(inp, out, name="PlantHealthCNN")


def _build_transfer_model(num_classes: int) -> Model:
    """MobileNetV2 transfer-learning variant."""
    base = keras.applications.MobileNetV2(
        input_shape=cfg.INPUT_SHAPE,
        include_top=False,
        weights="imagenet",
    )
    # Freeze all except last N layers
    for layer in base.layers[:cfg.FINE_TUNE_AT]:
        layer.trainable = False

    inp = keras.Input(shape=cfg.INPUT_SHAPE, name="leaf_image")
    x   = base(inp, training=False)
    x   = squeeze_excite(x, ratio=16)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(512, activation="relu",
                       kernel_regularizer=keras.regularizers.l2(cfg.L2))(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(cfg.DROPOUT)(x)
    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return Model(inp, out, name="PlantHealthCNN_Transfer")


# ─────────────────────────────────────────────
# STEP 5 — COMPILE
# ─────────────────────────────────────────────

def compile_model(model: Model, steps_per_epoch: int) -> Model:
    """
    Optimizer : Adam with cosine-decay LR + warm restarts
    Loss      : Categorical cross-entropy with label smoothing
    Metrics   : Accuracy, Top-3 accuracy, AUC
    """
    lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate = cfg.LR_INIT,
        first_decay_steps     = steps_per_epoch * 5,
        t_mul                 = 2.0,
        m_mul                 = 0.9,
        alpha                 = cfg.LR_MIN,
    )
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0),
        loss      = keras.losses.CategoricalCrossentropy(
                        label_smoothing=cfg.LABEL_SMOOTH),
        metrics   = [
            "accuracy",
            keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


# ─────────────────────────────────────────────
# STEP 6 — CALLBACKS
# ─────────────────────────────────────────────

def get_callbacks() -> list:
    return [
        EarlyStopping(
            monitor            = "val_accuracy",
            patience           = 12,
            restore_best_weights = True,
            min_delta          = 0.001,
            verbose            = 1,
        ),
        ModelCheckpoint(
            filepath           = str(cfg.SAVE_DIR / "best_model.keras"),
            monitor            = "val_accuracy",
            save_best_only     = True,
            verbose            = 1,
        ),
        ReduceLROnPlateau(
            monitor            = "val_loss",
            factor             = 0.5,
            patience           = 5,
            min_lr             = cfg.LR_MIN,
            verbose            = 1,
        ),
        TensorBoard(
            log_dir            = str(cfg.LOG_DIR),
            histogram_freq     = 1,
            update_freq        = "epoch",
        ),
        CSVLogger(
            filename           = str(cfg.SAVE_DIR / "training_log.csv"),
            append             = False,
        ),
    ]


# ─────────────────────────────────────────────
# STEP 7 — TRAIN
# ─────────────────────────────────────────────

def train(model, train_ds, val_ds, n_train, n_val):
    steps_per_epoch  = max(1, n_train  // cfg.BATCH_SIZE)
    validation_steps = max(1, n_val    // cfg.BATCH_SIZE)

    print(f"\n  Steps / epoch  : {steps_per_epoch}")
    print(f"  Validation steps: {validation_steps}")
    print(f"  Max epochs      : {cfg.EPOCHS}")
    print()

    history = model.fit(
        train_ds,
        epochs           = cfg.EPOCHS,
        steps_per_epoch  = steps_per_epoch,
        validation_data  = val_ds,
        validation_steps = validation_steps,
        callbacks        = get_callbacks(),
        verbose          = 1,
    )
    return history


# ─────────────────────────────────────────────
# STEP 8 — EVALUATE
# ─────────────────────────────────────────────

def evaluate(model, val_ds, n_val):
    print("\n  [8/9] Final evaluation on validation set…")
    steps = max(1, n_val // cfg.BATCH_SIZE)
    results = model.evaluate(val_ds, steps=steps, verbose=1)
    metrics = dict(zip(model.metrics_names, results))

    print("\n  ┌─────────────────────────────┐")
    print(f"  │  Accuracy  : {metrics.get('accuracy', 0)*100:6.2f}%        │")
    print(f"  │  Top-3 Acc : {metrics.get('top3_acc', 0)*100:6.2f}%        │")
    print(f"  │  AUC       : {metrics.get('auc', 0):.4f}          │")
    print(f"  │  Loss      : {metrics.get('loss', 0):.4f}          │")
    print("  └─────────────────────────────┘")
    return metrics


# ─────────────────────────────────────────────
# STEP 9 — PLOT & SAVE
# ─────────────────────────────────────────────

def plot_history(history, save_path: Path):
    """Save a 4-panel training history plot."""
    h   = history.history
    eps = range(1, len(h["accuracy"]) + 1)

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0d1117")

    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    panels = [
        (gs[0, 0], "accuracy",  "val_accuracy", "Accuracy",   "#58a6ff", "#3fb950"),
        (gs[0, 1], "loss",      "val_loss",      "Loss",       "#f85149", "#d29922"),
        (gs[1, 0], "top3_acc",  "val_top3_acc",  "Top-3 Acc", "#79c0ff", "#56d364"),
        (gs[1, 1], "auc",       "val_auc",        "AUC",       "#cba6f7", "#f38ba8"),
    ]

    for spec, train_k, val_k, title, tc, vc in panels:
        ax = fig.add_subplot(spec)
        ax.set_facecolor("#161b22")
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(colors="#8b949e", labelsize=9)
        ax.xaxis.label.set_color("#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.title.set_color("#f0f6fc")

        if train_k in h:
            ax.plot(eps, h[train_k], color=tc, lw=2, label="Train", alpha=0.9)
        if val_k in h:
            ax.plot(eps, h[val_k], color=vc, lw=2, label="Val", linestyle="--", alpha=0.9)

        # Mark best val epoch
        if val_k in h:
            best_e  = int(np.argmax(h[val_k]) if "acc" in val_k or "auc" in val_k
                         else np.argmin(h[val_k])) + 1
            best_v  = h[val_k][best_e - 1]
            ax.axvline(best_e, color="#6e7681", lw=1, linestyle=":")
            ax.annotate(f"best: {best_v:.4f}", xy=(best_e, best_v),
                        xytext=(5, 5), textcoords="offset points",
                        color="#8b949e", fontsize=7)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Epoch")
        ax.legend(facecolor="#21262d", labelcolor="#c9d1d9", fontsize=8,
                  framealpha=0.8, loc="best")
        ax.grid(True, color="#21262d", linewidth=0.5)

    fig.suptitle("Plant Health CNN — Training History",
                 color="#f0f6fc", fontsize=15, fontweight="bold", y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"\n  📊 Training plot saved: {save_path}")


def save_artifacts(model, classes, history, metrics):
    """Save model + class names + training summary."""
    # 1. Final model
    model_path = cfg.SAVE_DIR / "plant_health_final.keras"
    model.save(str(model_path))
    print(f"\n  ✅ Model saved      : {model_path}")

    # 2. TFLite (compressed, for edge deployment)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        tflite_path  = cfg.SAVE_DIR / "plant_health.tflite"
        tflite_path.write_bytes(tflite_model)
        print(f"  ✅ TFLite saved     : {tflite_path}")
    except Exception as e:
        print(f"  ⚠️  TFLite export skipped: {e}")

    # 3. Class names (for Streamlit app)
    class_path = cfg.SAVE_DIR / "class_names.json"
    with open(class_path, "w") as f:
        json.dump(classes, f, indent=2)
    print(f"  ✅ Class names saved: {class_path}")

    # 4. Training summary
    summary = {
        "timestamp":   datetime.now().isoformat(),
        "num_classes": len(classes),
        "epochs_run":  len(history.history["accuracy"]),
        "best_val_acc": float(max(history.history.get("val_accuracy", [0]))),
        "best_top3":   float(max(history.history.get("val_top3_acc",  [0]))),
        "best_auc":    float(max(history.history.get("val_auc",        [0]))),
        "final_metrics": {k: float(v) for k, v in metrics.items()},
        "config": {
            "img_size":    [cfg.IMG_H, cfg.IMG_W],
            "batch_size":  cfg.BATCH_SIZE,
            "lr_init":     cfg.LR_INIT,
            "dropout":     cfg.DROPOUT,
            "use_pretrained": cfg.USE_PRETRAINED,
        }
    }
    summary_path = cfg.SAVE_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ Summary saved    : {summary_path}")

    return model_path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   PLANT HEALTH CNN — FULL RETRAINING PIPELINE       ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  TensorFlow : {tf.__version__}")
    print(f"  GPU(s)     : {[g.name for g in tf.config.list_logical_devices('GPU')] or 'None — CPU mode'}")
    print(f"  Batch size : {cfg.BATCH_SIZE}")
    print(f"  Max epochs : {cfg.EPOCHS}")
    print(f"  Image size : {cfg.IMG_H}×{cfg.IMG_W}")
    print(f"  Backbone   : {'MobileNetV2 (Transfer)' if cfg.USE_PRETRAINED else 'Custom CNN'}")

    # ── 1. Download ────────────────────────────────────────
    dataset_root = download_dataset()

    # ── 2. Discover folders ───────────────────────────────
    train_dir, valid_dir, classes = find_data_dirs(dataset_root)
    num_classes = len(classes)
    print(f"\n  Classes ({num_classes}):")
    for i, c in enumerate(classes):
        print(f"    [{i:02d}] {c}")

    # ── 3. Count images ───────────────────────────────────
    n_train = sum(len(list(d.glob("*.*"))) for d in train_dir.iterdir() if d.is_dir())
    n_val   = sum(len(list(d.glob("*.*"))) for d in valid_dir.iterdir()  if d.is_dir()) if valid_dir else 0
    print(f"\n  Training images   : {n_train:,}")
    print(f"  Validation images : {n_val:,}")

    # ── 4. Build tf.data pipelines ────────────────────────
    print("\n  [3/9] Building data pipelines…")
    train_ds = build_dataset(train_dir, classes, training=True)
    val_ds   = build_dataset(valid_dir, classes, training=False) if valid_dir else None
    print("  ✅ Pipelines ready")

    # ── 5. Build model ────────────────────────────────────
    print("\n  [4/9] Building model…")
    model = build_model(num_classes)
    model.summary(line_length=90, expand_nested=False)
    print(f"\n  Total parameters : {model.count_params():,}")

    # ── 6. Compile ────────────────────────────────────────
    print("\n  [5/9] Compiling…")
    steps_per_epoch = max(1, n_train // cfg.BATCH_SIZE)
    model = compile_model(model, steps_per_epoch)
    print("  ✅ Compiled")

    # ── 7. Train ──────────────────────────────────────────
    print("\n  [6/9] Training…")
    print("═"*60)
    history = train(model, train_ds, val_ds or train_ds, n_train, n_val or n_train)

    # ── 8. Evaluate ───────────────────────────────────────
    metrics = evaluate(model, val_ds or train_ds, n_val or n_train)

    # ── 9. Plot ───────────────────────────────────────────
    print("\n  [7/9] Saving training plots…")
    plot_history(history, cfg.PLOT_DIR / "training_history.png")

    # ── 10. Save artifacts ────────────────────────────────
    print("\n  [8/9] Saving model & artifacts…")
    model_path = save_artifacts(model, classes, history, metrics)

    # ── Done ──────────────────────────────────────────────
    print("\n  [9/9] Done! 🎉")
    print("═"*60)
    print(f"  ✅ Best val accuracy : {max(history.history.get('val_accuracy', [0]))*100:.2f}%")
    print(f"  ✅ Best top-3 acc    : {max(history.history.get('val_top3_acc',  [0]))*100:.2f}%")
    print(f"\n  Model ready for Streamlit:")
    print(f"    cp {model_path} saved_models/plant_health_final.keras")
    print(f"    streamlit run app.py")
    print("═"*60 + "\n")

    return model, classes, history


if __name__ == "__main__":
    main()
