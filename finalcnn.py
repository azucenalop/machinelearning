from PIL import Image
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator

# set up paths to the dataset
train_data_dir = 'C:/Users/alg-1/maachine/train'
test_data_dir = 'C:/Users/alg-1/maachine/test'

# set up the image size
img_width, img_height = 256, 256

# load the images
def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = Image.open(os.path.join(folder_path, filename))
        img = img.resize((img_width, img_height))
        img_array = np.array(img)
        images.append(img_array)
    return images

# load training images
train_crack_path = os.path.join(train_data_dir, "crack")
train_nocrack_path = os.path.join(train_data_dir, "nocrack")
train_crack_images = load_images(train_crack_path)
train_nocrack_images = load_images(train_nocrack_path)

# load testing images
test_crack_path = os.path.join(test_data_dir, "crack")
test_nocrack_path = os.path.join(test_data_dir, "nocrack")
test_crack_images = load_images(test_crack_path)
test_nocrack_images = load_images(test_nocrack_path)

# create the labels
train_labels = np.concatenate([np.ones(len(train_crack_images)), np.zeros(len(train_nocrack_images))])
test_labels = np.concatenate([np.ones(len(test_crack_images)), np.zeros(len(test_nocrack_images))])

# concatenate the images and labels
train_images = np.concatenate([train_crack_images, train_nocrack_images])
test_images = np.concatenate([test_crack_images, test_nocrack_images])

# Define the batch size and number of epochs
batch_size = 32
epochs = 12

# Create the image data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(train_images, train_labels, batch_size=batch_size)
test_generator = test_datagen.flow(test_images, test_labels, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=epochs, validation_data=test_generator)

# Evaluate the model on the test set

test_loss, test_acc = model.evaluate(test_generator)

# Print the test accuracy
print('Test accuracy:', test_acc)

# Get the predicted probabilities for the test images
test_probabilities = []
for i in range(len(test_images)):
    x = np.expand_dims(test_images[i], axis=0)
    x = x / 255.0
    preds = model.predict(x)
    test_probabilities.append(preds[0][0])

# Print the predicted probabilities for the first 10 test images
print('Predicted probabilities for the first 10 test images:', test_probabilities[:10])
