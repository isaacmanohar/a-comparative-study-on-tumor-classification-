"""
Train Breast Ultrasound Classifier — MobileNetV2 Transfer Learning
Dataset: Dataset_BUSI_with_GT (3 classes: benign, malignant, normal)
Filters out mask images. Trains for 15 epochs with fine-tuning.
"""
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pandas as pd
import json

# ── Config ──
DATA_DIR  = 'Dataset_BUSI_with_GT'
IMG_SIZE  = (128, 128)
BATCH_SIZE = 32

print("=" * 50)
print("  BREAST ULTRASOUND CLASSIFIER — Training")
print("=" * 50)

# ── Collect files (filter out masks) ──
data = []
for label in ['benign', 'malignant', 'normal']:
    folder = os.path.join(DATA_DIR, label)
    if not os.path.exists(folder):
        continue
    for file in os.listdir(folder):
        if 'mask' not in file.lower() and file.lower().endswith(('.png', '.jpg', '.jpeg')):
            data.append({'filename': os.path.abspath(os.path.join(folder, file)), 'class': label})

df = pd.DataFrame(data)
print(f"\nTotal images (no masks): {len(df)}")
print(df['class'].value_counts().to_string())

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

train_gen = train_datagen.flow_from_dataframe(
    df, x_col='filename', y_col='class',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training'
)

val_gen = train_datagen.flow_from_dataframe(
    df, x_col='filename', y_col='class',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation'
)

print(f"\nClasses: {train_gen.class_indices}")
print(f"Training samples: {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")

# Save class mapping
with open('breast_classes.json', 'w') as f:
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
predictions = Dense(len(train_gen.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_gen, epochs=5, validation_data=val_gen)

# ── Phase 2: Fine-tune top layers of MobileNetV2 (10 more epochs) ──
print("\n-- Phase 2: Fine-tuning top layers --")
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
]

model.fit(train_gen, epochs=10, validation_data=val_gen, callbacks=callbacks)

# ── Save ──
model.save('breast_model.h5')
print("\n[OK] Model saved -> breast_model.h5")
print("[OK] Classes saved -> breast_classes.json")
print("Training Complete!")
