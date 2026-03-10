"""
🌿 Plant Health CNN — MacBook Air M2 Edition
=============================================
Optimized for Apple Silicon (M2) with Metal GPU acceleration.

Just run:
    python train_m2.py

That's it. Everything is automatic.
"""

# ── Silence noisy logs ──────────────────────────────────────
import os, sys, json, time, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# AUTO-INSTALL missing packages
# ─────────────────────────────────────────────
def install(pkg):
    os.system(f"{sys.executable} -m pip install {pkg} -q")

try:
    import kagglehub
except ImportError:
    print("📦 Installing kagglehub...")
    install("kagglehub")
    import kagglehub

try:
    import tensorflow as tf
except ImportError:
    print("📦 Installing tensorflow-macos + tensorflow-metal...")
    install("tensorflow-macos tensorflow-metal")
    import tensorflow as tf

import numpy as np
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# CONFIG — tuned for M2 MacBook Air
# ─────────────────────────────────────────────
IMG_SIZE   = 128          # smaller = faster, still accurate
BATCH      = 64           # M2 handles this well
EPOCHS     = 20           # plenty with transfer learning
LR         = 1e-3
SAVE_DIR   = Path("saved_models")
SAVE_DIR.mkdir(exist_ok=True)

print("""
╔══════════════════════════════════════════════╗
║  🌿 Plant Health CNN — M2 UltraLite Edition  ║
╠══════════════════════════════════════════════╣
║  Backbone  : MobileNetV3Small (pretrained)   ║
║  Image size: 128×128  (ultra-fast on M2)     ║
║  Batch size: 64                              ║
║  GPU       : Apple Metal (M2)                ║
╚══════════════════════════════════════════════╝
""")

# ── Check M2 Metal GPU ──────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"  ✅ Apple Metal GPU detected: {gpus[0].name}")
else:
    print("  ⚠️  No Metal GPU found — running on CPU")
    print("     Install: pip install tensorflow-macos tensorflow-metal")


# ─────────────────────────────────────────────
# STEP 1 — Download Dataset
# ─────────────────────────────────────────────
print("\n  [1/5] 📥 Downloading dataset via KaggleHub...")
print("        (~2 GB, only downloads once — cached after first run)\n")

path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
root = Path(path)
print(f"  ✅ Dataset ready at: {root}")


# ─────────────────────────────────────────────
# STEP 2 — Find train / valid folders
# ─────────────────────────────────────────────
print("\n  [2/5] 📂 Locating train/valid folders...")

train_dir = valid_dir = None
for p in sorted(root.rglob("*")):
    if p.is_dir():
        n = p.name.lower()
        if n == "train"  and train_dir is None: train_dir = p
        if n in ("valid","val","validation") and valid_dir is None: valid_dir = p

if not train_dir:
    raise FileNotFoundError(f"Could not find 'train' folder inside {root}")

classes  = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
n_train  = sum(len(list(d.glob("*.*"))) for d in train_dir.iterdir() if d.is_dir())
n_valid  = sum(len(list(d.glob("*.*"))) for d in valid_dir.iterdir() if d.is_dir()) if valid_dir else 0
NUM_CLS  = len(classes)

print(f"  ✅ Train  : {n_train:,} images")
print(f"  ✅ Valid  : {n_valid:,} images")
print(f"  ✅ Classes: {NUM_CLS}")


# ─────────────────────────────────────────────
# STEP 3 — Data Pipeline
# ─────────────────────────────────────────────
print("\n  [3/5] ⚙️  Building data pipelines...")

AUTOTUNE = tf.data.AUTOTUNE

def make_dataset(directory, training=True):
    ds = keras.utils.image_dataset_from_directory(
        str(directory),
        labels        = "inferred",
        label_mode    = "categorical",
        class_names   = classes,
        image_size    = (IMG_SIZE, IMG_SIZE),
        batch_size    = BATCH,
        shuffle       = training,
        seed          = 42,
    )

    norm = layers.Rescaling(1.0) # MobileNetV3 built-in norm takes [0, 255]

    # Light augmentation — fast on M2
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ])

    def prep_train(x, y):
        x = norm(x)
        x = aug(x, training=True)
        return x, y

    def prep_val(x, y):
        return norm(x), y

    ds = ds.map(prep_train if training else prep_val,
                num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(500).repeat()
    return ds.prefetch(AUTOTUNE)

train_ds = make_dataset(train_dir, training=True)
val_ds   = make_dataset(valid_dir, training=False) if valid_dir else None
print("  ✅ Pipelines ready")


# ─────────────────────────────────────────────
# STEP 4 — Build Lite Model
# ─────────────────────────────────────────────
print("\n  [4/5] 🧠 Building MobileNetV3Small model...")

# MobileNetV3 pretrained on ImageNet — ultra-tiny, lightning fast, accurate
base = keras.applications.MobileNetV3Small(
    input_shape = (IMG_SIZE, IMG_SIZE, 3),
    include_top = False,
    weights     = "imagenet",
)

# Freeze all base layers initially
base.trainable = False

inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="leaf")
x   = base(inp, training=False)
x   = layers.GlobalAveragePooling2D()(x)
x   = layers.Dense(256, activation="relu")(x)
x   = layers.Dropout(0.3)(x)
x   = layers.Dense(128, activation="relu")(x)
x   = layers.Dropout(0.2)(x)
out = layers.Dense(NUM_CLS, activation="softmax", name="predictions")(x)

model = keras.Model(inp, out, name="PlantHealth_M2_UltraLite")

# Phase 1: Train only the head (fast)
model.compile(
    optimizer = keras.optimizers.Adam(LR),
    loss      = keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics   = ["accuracy",
                 keras.metrics.TopKCategoricalAccuracy(k=3, name="top3")],
)

total_params = model.count_params()
trainable    = sum(tf.size(w).numpy() for w in model.trainable_weights)
print(f"  ✅ Total params    : {total_params:,}")
print(f"  ✅ Trainable params: {trainable:,}  (head only)")
model.summary(line_length=70, expand_nested=False)


# ─────────────────────────────────────────────
# STEP 5 — Train (2 phases)
# ─────────────────────────────────────────────
print("\n  [5/5] 🏋️  Training...")

steps_train = max(1, n_train // BATCH)
steps_val   = max(1, n_valid // BATCH) if n_valid else None
callbacks   = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5,
        restore_best_weights=True, verbose=1),
    keras.callbacks.ModelCheckpoint(
        str(SAVE_DIR / "best_lite.keras"),
        monitor="val_accuracy", save_best_only=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3,
        min_lr=1e-7, verbose=1),
]

# ── Phase 1: Head only (fast warmup, ~5 epochs) ─────────────
print("\n  ── Phase 1: Training classifier head (frozen base) ──")
h1 = model.fit(
    train_ds,
    epochs           = min(10, EPOCHS // 2),
    steps_per_epoch  = steps_train,
    validation_data  = val_ds,
    validation_steps = steps_val,
    callbacks        = callbacks,
    verbose          = 1,
)

# ── Phase 2: Fine-tune top layers of MobileNetV3Small ─────────
print("\n  ── Phase 2: Fine-tuning top layers of MobileNetV3Small ──")
base.trainable = True
# Freeze all except last 30 layers
for layer in base.layers[:-30]:
    layer.trainable = False

# Recompile with lower LR
model.compile(
    optimizer = keras.optimizers.Adam(LR / 10),
    loss      = keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics   = ["accuracy",
                 keras.metrics.TopKCategoricalAccuracy(k=3, name="top3")],
)
trainable2 = sum(tf.size(w).numpy() for w in model.trainable_weights)
print(f"  Trainable params now: {trainable2:,}")

h2 = model.fit(
    train_ds,
    epochs           = EPOCHS,
    initial_epoch    = len(h1.history["accuracy"]),
    steps_per_epoch  = steps_train,
    validation_data  = val_ds,
    validation_steps = steps_val,
    callbacks        = callbacks,
    verbose          = 1,
)

# Merge histories
history = {
    k: h1.history[k] + h2.history[k]
    for k in h1.history
}


# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
print("\n  💾 Saving model & artifacts...")

# Full Keras model
model.save(str(SAVE_DIR / "plant_health_lite.keras"))
print(f"  ✅ Keras model  → saved_models/plant_health_lite.keras")

# TFLite conversion is unsupported currently for MobileNetV3Small natively in TF macOS metal
print(f"  ⚠️  TFLite export skipped: LLVM error with MobileNetV3Small")

# Class names (for Streamlit app)
with open(SAVE_DIR / "class_names.json", "w") as f:
    json.dump(classes, f, indent=2)
print(f"  ✅ Class names  → saved_models/class_names.json")

# Training summary
best_acc = max(history.get("val_accuracy", [0]))
best_top3 = max(history.get("val_top3", [0]))
summary = {
    "model":        "MobileNetV3Small UltraLite (M2 Edition)",
    "img_size":     IMG_SIZE,
    "num_classes":  NUM_CLS,
    "epochs_run":   len(history["accuracy"]),
    "best_val_acc": float(best_acc),
    "best_top3":    float(best_top3),
}
with open(SAVE_DIR / "training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"  ✅ Summary      → saved_models/training_summary.json")


# ─────────────────────────────────────────────
# TRAINING PLOT
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor("#0d1117")

for ax in axes:
    ax.set_facecolor("#161b22")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ["bottom","left"]:
        ax.spines[sp].set_color("#30363d")
    ax.tick_params(colors="#8b949e")
    ax.yaxis.label.set_color("#8b949e")
    ax.xaxis.label.set_color("#8b949e")
    ax.title.set_color("#f0f6fc")

eps = range(1, len(history["accuracy"]) + 1)

axes[0].plot(eps, history["accuracy"],     color="#58a6ff", lw=2, label="Train")
axes[0].plot(eps, history["val_accuracy"], color="#3fb950", lw=2, label="Val", ls="--")
axes[0].set_title("Accuracy", fontweight="bold")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].legend(facecolor="#21262d", labelcolor="#c9d1d9")
axes[0].set_ylim([0, 1])

axes[1].plot(eps, history["loss"],     color="#f85149", lw=2, label="Train")
axes[1].plot(eps, history["val_loss"], color="#d29922", lw=2, label="Val", ls="--")
axes[1].set_title("Loss", fontweight="bold")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend(facecolor="#21262d", labelcolor="#c9d1d9")

plt.tight_layout()
plt.savefig("training_history.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
print(f"  ✅ Plot         → training_history.png")


# ─────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════════╗
║  🎉 Training Complete!                       ║
╠══════════════════════════════════════════════╣
║  Val Accuracy : {best_acc*100:5.2f}%                     ║
║  Top-3 Acc   : {best_top3*100:5.2f}%                     ║
║  Epochs run  : {len(history['accuracy']):<5}                      ║
╠══════════════════════════════════════════════╣
║  Next step:                                  ║
║    streamlit run app.py                      ║
╚══════════════════════════════════════════════╝
""")
