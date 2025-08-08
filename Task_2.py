# Task 2: Deep Learning Image Classification with CIFAR-10 Dataset

print("=" * 70)
print("DEEP LEARNING IMAGE CLASSIFICATION PROJECT")
print("CIFAR-10 Dataset with Convolutional Neural Network")
print("=" * 70)

# Core libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# TensorFlow and Keras imports
try:
    import tensorflow as tf
    import keras
    from keras import layers, models, optimizers, callbacks
    from keras.datasets import cifar10
    from keras.utils import to_categorical
    from keras.preprocessing.image import ImageDataGenerator

    print("✅ TensorFlow imported successfully!")
    print(f"TensorFlow version: {tf.__version__}")
except ImportError as e:
    print("❌ TensorFlow import error:")
    print("Please install TensorFlow using: pip install tensorflow")
    print("For GPU support: pip install tensorflow-gpu")
    exit()

# Additional utilities
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print()

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# =============================================================================
# STEP 2: DATA LOADING AND EXPLORATION
# =============================================================================

print("STEP 1: DATA LOADING AND EXPLORATION")
print("-" * 50)

# Load CIFAR-10 dataset
print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Class names for CIFAR-10
class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]

print("Dataset loaded successfully!")
print(f"Training images shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test images shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")
print(f"Number of classes: {len(class_names)}")
print(f"Image dimensions: {x_train.shape[1]}x{x_train.shape[2]}x{x_train.shape[3]}")

# Display class distribution
print("\nClass Distribution in Training Set:")
unique, counts = np.unique(y_train, return_counts=True)
for i, (class_id, count) in enumerate(zip(unique, counts)):
    print(
        f"  {class_names[class_id]:<12}: {count:,} images ({count/len(y_train)*100:.1f}%)"
    )

# =============================================================================
# STEP 3: DATA PREPROCESSING
# =============================================================================

print(f"\nSTEP 2: DATA PREPROCESSING")
print("-" * 50)

# Normalize pixel values to [0, 1] range
print("Normalizing pixel values...")
x_train_normalized = x_train.astype("float32") / 255.0
x_test_normalized = x_test.astype("float32") / 255.0

print(f"Original pixel range: [{x_train.min()}, {x_train.max()}]")
print(
    f"Normalized pixel range: [{x_train_normalized.min():.3f}, {x_train_normalized.max():.3f}]"
)

# Convert labels to categorical (one-hot encoding)
print("Converting labels to categorical format...")
num_classes = len(class_names)
y_train_categorical = to_categorical(y_train, num_classes)
y_test_categorical = to_categorical(y_test, num_classes)

print(f"Original label shape: {y_train.shape}")
print(f"Categorical label shape: {y_train_categorical.shape}")
print(f"Sample original label: {y_train[0]}")
print(f"Sample categorical label: {y_train_categorical[0]}")

# Data augmentation for better generalization
print("\nSetting up data augmentation...")
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode="nearest",
)

print("Data augmentation configured:")
print("  - Rotation: ±15 degrees")
print("  - Width/Height shift: ±10%")
print("  - Horizontal flip: Enabled")
print("  - Zoom: ±10%")

# =============================================================================
# STEP 4: MODEL ARCHITECTURE DESIGN
# =============================================================================

print(f"\nSTEP 3: CONVOLUTIONAL NEURAL NETWORK DESIGN")
print("-" * 50)


def create_cnn_model(input_shape, num_classes):
    """
    Create a Convolutional Neural Network for image classification

    Fixed Architecture for CIFAR-10 (32x32 images):
    - 3 Convolutional blocks with proper spatial dimension handling
    - Batch normalization and dropout for regularization
    - Global average pooling to reduce parameters
    - Dense layers for classification
    """

    model = models.Sequential(
        [
            # First Convolutional Block
            layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
                input_shape=input_shape,
                name="conv1",
            ),
            layers.BatchNormalization(name="bn1"),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv2"),
            layers.MaxPooling2D((2, 2), name="pool1"),  # 32x32 -> 16x16
            layers.Dropout(0.25, name="drop1"),
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv3"),
            layers.BatchNormalization(name="bn2"),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv4"),
            layers.MaxPooling2D((2, 2), name="pool2"),  # 16x16 -> 8x8
            layers.Dropout(0.25, name="drop2"),
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv5"),
            layers.BatchNormalization(name="bn3"),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv6"),
            layers.MaxPooling2D((2, 2), name="pool3"),  # 8x8 -> 4x4
            layers.Dropout(0.25, name="drop3"),
            # Optional Fourth Block for deeper features
            layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="conv7"),
            layers.BatchNormalization(name="bn5"),
            layers.Dropout(0.25, name="drop5"),
            # Classification Head
            layers.GlobalAveragePooling2D(name="global_pool"),  # 4x4x256 -> 256
            layers.Dense(512, activation="relu", name="dense1"),
            layers.BatchNormalization(name="bn4"),
            layers.Dropout(0.5, name="drop4"),
            layers.Dense(num_classes, activation="softmax", name="output"),
        ]
    )

    return model


# Create the model
input_shape = (32, 32, 3)  # CIFAR-10 image dimensions
model = create_cnn_model(input_shape, num_classes)

# Display model architecture
print("Model Architecture:")
model.summary()

# Count total parameters
total_params = model.count_params()
trainable_params = sum(
    [tf.keras.backend.count_params(w) for w in model.trainable_weights]
)
print(f"\nModel Statistics:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Model size (approx): {total_params * 4 / 1024 / 1024:.2f} MB")

# =============================================================================
# STEP 5: MODEL COMPILATION AND CALLBACKS
# =============================================================================

print(f"\nSTEP 4: MODEL COMPILATION AND TRAINING SETUP")
print("-" * 50)

# Compile the model
optimizer = optimizers.Adam(learning_rate=0.001)

# Use available metrics based on TensorFlow version
try:
    # Try to use top_k_categorical_accuracy for newer versions
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
        ],
    )
    metrics_used = "Accuracy, Top-3 accuracy"
except:
    # Fallback to basic metrics for compatibility
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    metrics_used = "Accuracy"

print("Model compiled with:")
print(f"  Optimizer: Adam (lr=0.001)")
print(f"  Loss function: Categorical crossentropy")
print(f"  Metrics: {metrics_used}")

# Setup callbacks for training
callbacks_list = [
    # Reduce learning rate when loss plateaus
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
    ),
    # Early stopping to prevent overfitting
    callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    ),
    # Save best model
    callbacks.ModelCheckpoint(
        "best_cifar10_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1
    ),
]

print("Training callbacks configured:")
print("  - Learning rate reduction on plateau")
print("  - Early stopping (patience=10)")
print("  - Model checkpointing")

# =============================================================================
# STEP 6: MODEL TRAINING
# =============================================================================

print(f"\nSTEP 5: MODEL TRAINING")
print("-" * 50)

# Training parameters
batch_size = 32
epochs = 50  # Will likely stop early due to early stopping
validation_split = 0.2

print(f"Training configuration:")
print(f"  Batch size: {batch_size}")
print(f"  Maximum epochs: {epochs}")
print(f"  Validation split: {validation_split}")
print(f"  Data augmentation: Enabled")

print(f"\nStarting training at {datetime.now().strftime('%H:%M:%S')}...")

# Fit the data generator to training data
datagen.fit(x_train_normalized)

# Train the model with data augmentation
history = model.fit(
    datagen.flow(x_train_normalized, y_train_categorical, batch_size=batch_size),
    steps_per_epoch=len(x_train_normalized) // batch_size,
    epochs=epochs,
    validation_data=(x_test_normalized, y_test_categorical),
    callbacks=callbacks_list,
    verbose=1,
)

print(f"Training completed at {datetime.now().strftime('%H:%M:%S')}")

# =============================================================================
# STEP 7: MODEL EVALUATION
# =============================================================================

print(f"\nSTEP 6: MODEL EVALUATION")
print("-" * 50)

# Evaluate on test set
test_results = model.evaluate(x_test_normalized, y_test_categorical, verbose=0)
test_loss = test_results[0]
test_accuracy = test_results[1]

# Check if top-3 accuracy is available
if len(test_results) > 2:
    test_top3_accuracy = test_results[2]
    print(f"Test Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(
        f"  Test Top-3 Accuracy: {test_top3_accuracy:.4f} ({test_top3_accuracy*100:.2f}%)"
    )
else:
    test_top3_accuracy = None
    print(f"Test Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Make predictions
print("\nGenerating predictions...")
y_pred_proba = model.predict(x_test_normalized, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test_categorical, axis=1)

# Calculate detailed metrics
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, average=None, labels=range(num_classes)
)

print(f"\nPer-Class Performance:")
print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
print("-" * 55)
for i, class_name in enumerate(class_names):
    print(
        f"{class_name:<12} {precision[i]:<10.3f} {recall[i]:<10.3f} {f1[i]:<10.3f} {support[i]:<8}"
    )

# Overall metrics
macro_precision = np.mean(precision)
macro_recall = np.mean(recall)
macro_f1 = np.mean(f1)

print(f"\nOverall Performance:")
print(f"  Macro Precision: {macro_precision:.3f}")
print(f"  Macro Recall: {macro_recall:.3f}")
print(f"  Macro F1-Score: {macro_f1:.3f}")

# =============================================================================
# STEP 8: VISUALIZATION OF RESULTS (FIXED)
# =============================================================================

print(f"\nSTEP 7: CREATING VISUALIZATIONS")
print("-" * 50)

# Set up the plotting style
plt.style.use("default")
plt.rcParams["figure.figsize"] = (15, 10)

# Create a comprehensive visualization with 4x4 grid (FIXED)
fig = plt.figure(figsize=(20, 16))

# 1. Training History
plt.subplot(4, 4, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy", linewidth=2)
plt.plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
plt.title("Model Accuracy Over Time", fontsize=14, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(4, 4, 2)
plt.plot(history.history["loss"], label="Training Loss", linewidth=2, color="red")
plt.plot(
    history.history["val_loss"], label="Validation Loss", linewidth=2, color="orange"
)
plt.title("Model Loss Over Time", fontsize=14, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Confusion Matrix (spanning 2 positions)
plt.subplot(4, 4, (3, 4))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# 3. Per-class Performance Bar Chart (spanning 2 positions)
plt.subplot(4, 4, (5, 6))
x_pos = np.arange(len(class_names))
plt.bar(x_pos - 0.2, precision, 0.2, label="Precision", alpha=0.8)
plt.bar(x_pos, recall, 0.2, label="Recall", alpha=0.8)
plt.bar(x_pos + 0.2, f1, 0.2, label="F1-Score", alpha=0.8)
plt.xlabel("Classes")
plt.ylabel("Score")
plt.title("Per-Class Performance Metrics", fontsize=14, fontweight="bold")
plt.xticks(x_pos, class_names, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Learning Rate Schedule (if available)
if "lr" in history.history:
    plt.subplot(4, 4, 7)
    plt.plot(history.history["lr"])
    plt.title("Learning Rate Schedule", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)

# 5. Top-3 Accuracy (if available) or Validation Accuracy
plt.subplot(4, 4, 8)
if "top_3_accuracy" in history.history:
    plt.plot(history.history["top_3_accuracy"], label="Training Top-3")
    if "val_top_3_accuracy" in history.history:
        plt.plot(history.history["val_top_3_accuracy"], label="Validation Top-3")
    plt.title("Top-3 Accuracy Over Time", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Top-3 Accuracy")
    plt.legend()
else:
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
plt.grid(True, alpha=0.3)

# 6. Sample Predictions Visualization (FIXED - now uses positions 9-16)
sample_indices = np.random.choice(len(x_test), 8, replace=False)
sample_images = x_test[sample_indices]
sample_true = y_test[sample_indices].flatten()
sample_pred = y_pred[sample_indices]
sample_proba = y_pred_proba[sample_indices]

# Create a grid of sample predictions (FIXED)
for i in range(8):
    plt.subplot(4, 4, 9 + i)  # Now positions 9-16 are valid in 4x4 grid
    plt.imshow(sample_images[i])
    true_label = class_names[sample_true[i]]
    pred_label = class_names[sample_pred[i]]
    confidence = sample_proba[i][sample_pred[i]]

    # Color code: green for correct, red for incorrect
    color = "green" if sample_true[i] == sample_pred[i] else "red"
    plt.title(
        f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}",
        fontsize=9,
        color=color,
    )
    plt.axis("off")

plt.tight_layout()
plt.savefig("cifar10_classification_results.png", dpi=300, bbox_inches="tight")
plt.show()

# Additional detailed visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Class-wise Accuracy
class_accuracy = []
for i in range(num_classes):
    class_mask = y_true == i
    if np.sum(class_mask) > 0:
        class_acc = accuracy_score(y_true[class_mask], y_pred[class_mask])
        class_accuracy.append(class_acc)
    else:
        class_accuracy.append(0)

axes[0, 0].bar(range(num_classes), class_accuracy, color="skyblue", alpha=0.7)
axes[0, 0].set_title("Per-Class Accuracy")
axes[0, 0].set_xlabel("Class")
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].set_xticks(range(num_classes))
axes[0, 0].set_xticklabels(class_names, rotation=45)

# Prediction Confidence Distribution
axes[0, 1].hist(np.max(y_pred_proba, axis=1), bins=50, alpha=0.7, color="lightcoral")
axes[0, 1].set_title("Prediction Confidence Distribution")
axes[0, 1].set_xlabel("Maximum Prediction Probability")
axes[0, 1].set_ylabel("Frequency")

# Training Loss vs Validation Loss (zoomed)
axes[1, 0].plot(history.history["loss"], label="Training Loss", linewidth=2)
axes[1, 0].plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
axes[1, 0].set_title("Loss Curves (Detailed View)")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Loss")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Model Performance Summary
axes[1, 1].axis("off")
summary_text = f"""
Model Performance Summary
========================
Test Accuracy: {test_accuracy*100:.2f}%
Training Epochs: {len(history.history['loss'])}
Total Parameters: {total_params:,}
Model Size: {total_params * 4 / 1024 / 1024:.2f} MB

Best Performing Class:
{class_names[np.argmax(f1)]} (F1: {np.max(f1):.3f})

Most Challenging Class:
{class_names[np.argmin(f1)]} (F1: {np.min(f1):.3f})
"""
axes[1, 1].text(
    0.1,
    0.5,
    summary_text,
    fontsize=12,
    verticalalignment="center",
    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
)

plt.tight_layout()
plt.savefig("cifar10_detailed_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

# =============================================================================
# STEP 9: MODEL INSIGHTS AND RECOMMENDATIONS
# =============================================================================

print(f"\nSTEP 8: MODEL INSIGHTS AND RECOMMENDATIONS")
print("-" * 50)

print("MODEL PERFORMANCE ANALYSIS:")
print(f"✅ Final test accuracy: {test_accuracy*100:.2f}%")
print(f"✅ Training completed in {len(history.history['loss'])} epochs")

# Identify best and worst performing classes
best_class_idx = np.argmax(f1)
worst_class_idx = np.argmin(f1)

print(f"\nCLASS PERFORMANCE INSIGHTS:")
print(
    f"🏆 Best performing class: {class_names[best_class_idx]} (F1: {f1[best_class_idx]:.3f})"
)
print(
    f"📉 Most challenging class: {class_names[worst_class_idx]} (F1: {f1[worst_class_idx]:.3f})"
)

# Analyze confusion patterns
print(f"\nCONFUSION ANALYSIS:")
for i in range(num_classes):
    confused_with = np.argsort(cm[i])[-2]  # Second highest (excluding self)
    if confused_with != i and cm[i][confused_with] > 0:
        confusion_rate = cm[i][confused_with] / np.sum(cm[i]) * 100
        print(
            f"  {class_names[i]} often confused with {class_names[confused_with]} ({confusion_rate:.1f}%)"
        )

print(f"\nRECOMMENDATIONS FOR IMPROVEMENT:")
print("1. MODEL ARCHITECTURE:")
print("   • Consider deeper networks (ResNet, DenseNet) for better feature extraction")
print("   • Experiment with attention mechanisms")
print("   • Try transfer learning with pre-trained models")

print("\n2. DATA AUGMENTATION:")
print("   • Add more aggressive augmentation for challenging classes")
print("   • Consider mixup or cutmix techniques")
print("   • Collect more data for underperforming classes")

print("\n3. TRAINING OPTIMIZATION:")
print("   • Experiment with different optimizers (AdamW, RMSprop)")
print("   • Try cosine annealing learning rate schedule")
print("   • Implement label smoothing for better generalization")

print("\n4. DEPLOYMENT CONSIDERATIONS:")
print("   • Model is ready for deployment with current performance")
print("   • Consider model quantization for mobile deployment")
print("   • Implement ensemble methods for production use")

# =============================================================================
# STEP 10: SAVE MODEL AND RESULTS
# =============================================================================

print(f"\nSTEP 9: SAVING MODEL AND RESULTS")
print("-" * 50)

# Save the trained model
model.save("cifar10_cnn_model.h5")
print("✅ Model saved as 'cifar10_cnn_model.h5'")

# Save training history
np.save("training_history.npy", history.history)
print("✅ Training history saved as 'training_history.npy'")

# Save predictions and metrics
results = {
    "test_accuracy": test_accuracy,
    "test_loss": test_loss,
    "predictions": y_pred,
    "true_labels": y_true,
    "class_names": class_names,
    "confusion_matrix": cm,
    "classification_report": classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    ),
}

np.save("model_results.npy", results)
print("✅ Results saved as 'model_results.npy'")

print(f"\n" + "=" * 70)
print("🎉 DEEP LEARNING PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 70)
print(f"📊 Final Model Performance: {test_accuracy*100:.2f}% accuracy")
print(f"📁 Files created:")
print(f"   • cifar10_cnn_model.h5 (trained model)")
print(f"   • training_history.npy (training metrics)")
print(f"   • model_results.npy (evaluation results)")
print(f"   • cifar10_classification_results.png (visualizations)")
print(f"   • cifar10_detailed_analysis.png (detailed analysis)")
print(f"\n🚀 Model is ready for deployment and further experimentation!")

# =============================================================================
# BONUS: EXAMPLE USAGE FOR NEW PREDICTIONS
# =============================================================================

print(f"\nBONUS: EXAMPLE CODE FOR MAKING NEW PREDICTIONS")
print("-" * 50)

example_code = """
# Load the saved model for future predictions
from tensorflow.keras.models import load_model
import numpy as np

# Load model
model = load_model('cifar10_cnn_model.h5')

# Predict on new image (assuming image is preprocessed)
# new_image = preprocess_your_image()  # Shape should be (1, 32, 32, 3)
# prediction = model.predict(new_image)
# predicted_class = class_names[np.argmax(prediction)]
# confidence = np.max(prediction)

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2f}")
"""

print(example_code)
print("This completes the comprehensive deep learning image classification project!")
