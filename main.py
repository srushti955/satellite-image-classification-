import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pathlib

data_path = pathlib.Path("2750")

image_height = 64
image_width = 64
batch_size = 32

train_data = tf.keras.utils.image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(image_height, image_width),
    batch_size=batch_size
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(image_height, image_width),
    batch_size=batch_size
)

class_names = train_data.class_names

print("Image Categories:")
print(class_names)

cnn_model = models.Sequential()

cnn_model.add(layers.Rescaling(1.0 / 255, input_shape=(64, 64, 3)))

cnn_model.add(layers.Conv2D(32, (3, 3), activation='relu'))
cnn_model.add(layers.MaxPooling2D())

cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(layers.MaxPooling2D())

cnn_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
cnn_model.add(layers.MaxPooling2D())

cnn_model.add(layers.Flatten())

cnn_model.add(layers.Dense(128, activation='relu'))

cnn_model.add(layers.Dropout(0.5))

cnn_model.add(layers.Dense(len(class_names), activation='softmax'))

cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

training_history = cnn_model.fit(
    train_data,
    validation_data=validation_data,
    epochs=10
)

cnn_model.save("satellite_image_model.h5")

plt.plot(training_history.history['accuracy'])
plt.plot(training_history.history['val_accuracy'])

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend(["Training Accuracy", "Validation Accuracy"])

plt.show()