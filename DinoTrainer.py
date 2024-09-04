import os
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
from keras import models, layers, Input
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Path to the images
image_folder = 'Images'

# Image parameters
img_height = 128
img_width = 72
img_channels = 3

# Model's parameters
batch_size = 32
epoch = 12

# Initialize lists for images and labels
images = []
labels = []

# Load images and labels
for filename in os.listdir(image_folder):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # Extract label from filename
        label_str = filename.split(' ')[0]
        if label_str == '0_0':
            label = 0  # No button
        elif label_str == '1_0':
            label = 1  # Jump
        elif label_str == '0_1':
            label = 2  # Crouch
        else:
            continue  # Skip files with unexpected labels

        # Load image
        img_path = os.path.join(image_folder, filename)
        img = load_img(img_path, target_size=(img_height, img_width))
        img_array = img_to_array(img)

        # Append image and label to lists
        images.append(img_array)
        labels.append(label)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Normalize pixel values
images = images / 255.0

# Convert labels to categorical format
labels = to_categorical(labels, num_classes=3)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the InputShape
input_shape = (img_height, img_width, img_channels)

# Define the model
model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    #layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    #layers.Dropout(0.2),
    layers.Dense(3, activation='softmax')  # 3 classes: no button, jump, crouch
])

# Define a learning rate schedule
def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.1

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with LearningRateScheduler callback
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epoch,
    batch_size=batch_size,
    callbacks=[keras.callbacks.LearningRateScheduler(lr_schedule)]
)

# Save the model
model.save('Dino_AI.keras')

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()