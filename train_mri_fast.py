import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
import json

TRAIN_DIR = 'brain_mri_dataset/Training'
IMG_SIZE = (64, 64)
BATCH_SIZE = 64

datagen = ImageDataGenerator(rescale=1./255)
train_gen = datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

# Tiny, ultra-fast CNN
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, epochs=2)

with open('mri_classes.json', 'w') as f:
    json.dump({v: k for k, v in train_gen.class_indices.items()}, f)

model.save('mri_model.h5')
print("Fast Training Complete!")
