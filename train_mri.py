"""
Train Brain MRI Tumor Classifier — MobileNetV2 Transfer Learning
Dataset: brain_mri_dataset (4 classes: glioma, meningioma, no_tumor, pituitary)
Trains for 15 epochs with fine-tuning for accurate predictions.
"""
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import json

# ── Config ──
TRAIN_DIR = 'brain_mri_dataset/Training'
TEST_DIR  = 'brain_mri_dataset/Testing'
IMG_SIZE  = (128, 128)
BATCH_SIZE = 32

print("=" * 50)
print("  BRAIN MRI TUMOR CLASSIFIER — Training")
print("=" * 50)

# ── Data Augmentation ──
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training'
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation'
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

print(f"\nClasses: {train_gen.class_indices}")
print(f"Training samples: {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")
print(f"Test samples: {test_gen.samples}")

# Save class mapping
with open('mri_classes.json', 'w') as f:
    json.dump({v: k for k, v in train_gen.class_indices.items()}, f)

# ── Phase 1: Train with frozen base (5 epochs) ──
print("\n-- Phase 1: Training classification head (frozen base) --")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_gen, epochs=5, validation_data=val_gen)

# ── Phase 2: Fine-tune top layers of MobileNetV2 (10 more epochs) ──
print("\n-- Phase 2: Fine-tuning top layers --")
# Unfreeze the last 30 layers
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
]

model.fit(train_gen, epochs=10, validation_data=val_gen, callbacks=callbacks)

# ── Evaluate on test set ──
print("\n-- Evaluating on test set --")
test_loss, test_acc = model.evaluate(test_gen)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# ── Save ──
model.save('mri_model.h5')
print("\n[OK] Model saved -> mri_model.h5")
print("[OK] Classes saved -> mri_classes.json")
print("Training Complete!")
