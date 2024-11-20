import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from PIL import Image
from sklearn.model_selection import train_test_split

# Set image size and other parameters
data_dir = 'FishDataset'
image_size = (224, 224)
num_classes = 10
batch_size = 32
epochs = 75

# Initialize lists to store images and labels
images = []
labels = []

# Get class labels from directory names
class_labels = sorted(os.listdir(data_dir))

# Load and preprocess images
for label in class_labels:
    class_dir = os.path.join(data_dir, label)
    if os.path.isdir(class_dir):
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, filename)
                img = load_img(img_path, target_size=image_size)
                img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
                images.append(img_array)
                labels.append(label)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Convert string labels into integer labels
labels_encoded = label_encoder.fit_transform(labels)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# Standardize the data by subtracting mean and dividing by standard deviation
mean = np.mean(x_train, axis=(0, 1, 2), keepdims=True)
std = np.std(x_train, axis=(0, 1, 2), keepdims=True)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Data augmentation with .repeat() to prevent data shortage
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)

# Set up and compile the modified model with fine-tuning, batch normalization, and lower learning rate
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = True

# Adjusted model setup
for layer in base_model.layers[:100]:  # Freeze initial layers, unfreeze the rest
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),  # Additional dense layer
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Optimizer
opt = optimizers.Adam(learning_rate=1e-4)  # Start with a slightly higher learning rate

# Adjusted model setup with more dropout and Adam optimizer
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.6),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.6),
    layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

def scheduler(epoch, lr):
    if epoch < 10:
        return lr + 1e-6
    else:
        return float(lr * tf.math.exp(-0.1).numpy())
cyclic_lr = LearningRateScheduler(scheduler)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
cyclic_lr = LearningRateScheduler(scheduler)  # Using cyclical learning rate

# Train with augmented data and modified callbacks
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping, lr_scheduler, cyclic_lr],
    verbose=1
)


# Save the model
model.save('fish_model_transfer_learningv2.h5')

# Plot training history
def plot_metrics(history):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid()

    plt.tight_layout()
    plt.show()

plot_metrics(history)