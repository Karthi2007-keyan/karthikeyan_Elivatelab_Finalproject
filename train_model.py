import os
import numpy as np
import cv2
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Configuration
DATA_DIR = "dataset"
CATEGORIES = ["with_mask", "without_mask"]
IMG_SIZE = 100

data = []
labels = []

# Load and preprocess dataset
for category in CATEGORIES:
    path = os.path.join(DATA_DIR, category)
    label = CATEGORIES.index(category)

    for img_name in tqdm(os.listdir(path), desc=f"Loading {category}"):
        img_path = os.path.join(path, img_name)
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(label)

if len(data) == 0:
    raise ValueError("No training images found!")

# Prepare data
X = np.array(data, dtype="float32") / 255.0
y = to_categorical(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"[INFO] Training on {len(X_train)} images, Testing on {len(X_test)} images")

# Augmentation
train_aug = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(2, activation="softmax")
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train
model.fit(train_aug.flow(X_train, y_train, batch_size=32),
          epochs=20,
          validation_data=(X_test, y_test))

# Save model
os.makedirs("model", exist_ok=True)
model.save("model/mask_detector.keras")
print("[INFO] Model trained and saved successfully.")
