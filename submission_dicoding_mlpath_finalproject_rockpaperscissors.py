# -*- coding: utf-8 -*-
"""Submission_NisrinaIshmahMahira_DicodingFinalProject_RockPaperScissors.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19qCjXXXheEBdocgnAx27vQnf5r28820F

# **Dicoding: Machine Learning Final Project Submission for Beginners**
**Learning Path : Machine Learning**

**Project name    : Rock, Paper and Scissors**

---



---

= Name    : Nisrina Ishmah Mahira / <a href="https://www.linkedin.com/in/nisrinaishmah/" target="_blank">LinkedIn</a>

= Email   : nisrinaishmahmahira@gmail.com

= ID Dicoding : <a href='https://www.dicoding.com/users/ishmanis/' target="_blank">ishmanis</a>

= <a href="https://github.com/nisrinaishmh/Artificial-Neural-Network-Program-to-Recognize-Hand-Shapes-that-Form-Scissors-Rock-or-Paper" target="_blank">Github</a>

---

## Library Import
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
import time

print('Versi tensorflow: ', tf.__version__)

"""## Dataset Import"""

!wget --no-check-certificate \
https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip \
-O /tmp/rockpaperscissors.zip

# Extract the zip file
import zipfile,os
local_zip = '/tmp/rockpaperscissors.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

base_dir = '/tmp/rockpaperscissors/rps-cv-images'

"""## ImageDataGenerator for Image Augmentation

*   List item
*   List item


"""

# Initialize ImageDataGenerator for image augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.4  # Division of dataset into training and validation set(40%)

)

# Loading dataset using ImageDataGenerator
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

"""## Sequential Model"""

# Building a Sequential model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(), # Flatten (Flatten) data before entering the hidden layer
    Dense(512, activation='relu'),  # First hidden layer
    Dense(256, activation='relu'),  # Second hidden layer
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 class: Rock, Paper, Scissors
])

# Compile model with optimizer and loss function
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=categorical_crossentropy,
              metrics=['accuracy'])

# Stop training when accuracy is 95% (min_delta=0.05)
early_stopping = EarlyStopping(monitor='accuracy', min_delta=0.05, patience=5, mode='max', verbose=1)

# Save an initial time before training begins
start_time = time.time()

# Model Training
history = model.fit(
    train_generator,
    steps_per_epoch=32, #len(train_generator),
    epochs=100,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[early_stopping]  # callback stops training when it is reached
)

# Save the model
model.save('rps_model.keras')

"""## Required Model Training Time

*   List item
*   List item


"""

# Calculate training time total
end_time = time.time()
training_time_seconds = end_time - start_time

# from seconds to minutes
training_time_minutes = training_time_seconds / 60

print(f"Total waktu pelatihan: {training_time_minutes:.2f} menit")

"""## Accuracy and Loss Plots on Training and Validation Sets

"""

import plotly.graph_objs as go
import plotly.express as px

# Accuracy plot on training and validation sets
trace1 = go.Scatter(x=list(range(1, len(history.history['accuracy']) + 1)),
                    y=history.history['accuracy'],
                    mode='lines',
                    name='Akurasi Pelatihan')
trace2 = go.Scatter(x=list(range(1, len(history.history['val_accuracy']) + 1)),
                    y=history.history['val_accuracy'],
                    mode='lines',
                    name='Akurasi Validasi')

layout1 = go.Layout(title='Akurasi Pelatihan dan Validasi',
                    xaxis=dict(title='Epoch'),
                    yaxis=dict(title='Akurasi'))
fig1 = go.Figure(data=[trace1, trace2], layout=layout1)

# Loss plot on training and validation sets
trace3 = go.Scatter(x=list(range(1, len(history.history['loss']) + 1)),
                    y=history.history['loss'],
                    mode='lines',
                    name='Loss Pelatihan')
trace4 = go.Scatter(x=list(range(1, len(history.history['val_loss']) + 1)),
                    y=history.history['val_loss'],
                    mode='lines',
                    name='Loss Validasi')

layout2 = go.Layout(title='Loss Pelatihan dan Validasi',
                    xaxis=dict(title='Epoch'),
                    yaxis=dict(title='Loss'))
fig2 = go.Figure(data=[trace3, trace4], layout=layout2)

fig1.show()
fig2.show()

"""# Testing Machine Learning Models

"""

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from google.colab import files
from IPython.display import Image, display
import numpy as np

# Load the trained model
model = load_model('rps_model.keras')

# Uploading images
uploaded_image = files.upload()

# Retrieves the uploaded image file name
image_files = list(uploaded_image.keys())

if len(image_files) == 0:
    print("Tidak ada gambar yang diunggah.")
else:
    # Takes the first image (if there are many images uploaded)
    uploaded_image_path = image_files[0]

    # Load and process the image to be predicted
    img = image.load_img(uploaded_image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalization

    # Displays uploaded images
    display(Image(uploaded_image_path))

    # Make predictions
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)

    # Displays prediction results
    if class_idx == 0:
        print("Paper")
    elif class_idx == 1:
        print("Rock")
    else:
        print("Scissors")

# Uploading images
uploaded_image = files.upload()

# Retrieves the uploaded image file name
image_files = list(uploaded_image.keys())

if len(image_files) == 0:
    print("Tidak ada gambar yang diunggah.")
else:
    # Takes the first image (if there are many images uploaded)
    uploaded_image_path = image_files[0]

    # Load and process the image to be predicted
    img = image.load_img(uploaded_image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalization


    # Displays uploaded images
    display(Image(uploaded_image_path))

    # Make predictions
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)

    # Displays prediction results
    if class_idx == 0:
        print("Paper")
    elif class_idx == 1:
        print("Rock")
    else:
        print("Scissors")

# Uploading images
uploaded_image = files.upload()

# Retrieves the uploaded image file name
image_files = list(uploaded_image.keys())

if len(image_files) == 0:
    print("Tidak ada gambar yang diunggah.")
else:
    # Takes the first image (if there are many images uploaded)
    uploaded_image_path = image_files[0]

    # Load and process the image to be predicted
    img = image.load_img(uploaded_image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalization


    # Displays uploaded images
    display(Image(uploaded_image_path))

    # Make predictions
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)

    # Displays prediction results
    if class_idx == 0:
        print("Paper")
    elif class_idx == 1:
        print("Rock")
    else:
        print("Scissors")

# Uploading images
uploaded_image = files.upload()

# Retrieves the uploaded image file name
image_files = list(uploaded_image.keys())

if len(image_files) == 0:
    print("Tidak ada gambar yang diunggah.")
else:
    # Takes the first image (if there are many images uploaded)
    uploaded_image_path = image_files[0]

    # Load and process the image to be predicted
    img = image.load_img(uploaded_image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalization


    # Displays uploaded images
    display(Image(uploaded_image_path))

    # Make predictions
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)

    # Displays prediction results
    if class_idx == 0:
        print("Paper")
    elif class_idx == 1:
        print("Rock")
    else:
        print("Scissors")

# Uploading images
uploaded_image = files.upload()

# Retrieves the uploaded image file name
image_files = list(uploaded_image.keys())

if len(image_files) == 0:
    print("Tidak ada gambar yang diunggah.")
else:
    # Takes the first image (if there are many images uploaded)
    uploaded_image_path = image_files[0]

    # Load and process the image to be predicted
    img = image.load_img(uploaded_image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalization


    # Displays uploaded images
    display(Image(uploaded_image_path))

    # Make predictions
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)

    # Displays prediction results
    if class_idx == 0:
        print("Paper")
    elif class_idx == 1:
        print("Rock")
    else:
        print("Scissors")

# Uploading images
uploaded_image = files.upload()

# Retrieves the uploaded image file name
image_files = list(uploaded_image.keys())

if len(image_files) == 0:
    print("Tidak ada gambar yang diunggah.")
else:
    # Takes the first image (if there are many images uploaded)
    uploaded_image_path = image_files[0]

    # Load and process the image to be predicted
    img = image.load_img(uploaded_image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalization


    # Displays uploaded images
    display(Image(uploaded_image_path))

    # Make predictions
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)

    # Displays prediction results
    if class_idx == 0:
        print("Paper")
    elif class_idx == 1:
        print("Rock")
    else:
        print("Scissors")