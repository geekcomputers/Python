import pickle

import tensorflow as tf

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 3)),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation='relu'),
                                    tf.keras.layers.Dense(1, activation="sigmoid")
                                    ])
model.summary()
from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1 / 255)
train_generator = train_datagen.flow_from_directory('../Classification_human-or-horse',
                                                    target_size=(200, 200),
                                                    batch_size=222,
                                                    class_mode='binary')
model.fit_generator(train_generator, steps_per_epoch=6, epochs=1, verbose=1)
filename = "myTf1.sav"
pickle.dump(model, open(filename, 'wb'))

from tkinter import Tk
from tkinter.filedialog import askopenfilename
from keras.preprocessing import image
import numpy as np

Tk().withdraw()
filename = askopenfilename()
print(filename)
img = image.load_img(filename, target_size=(200, 200))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes[0])
if classes[0] > 0.5:
    print(filename + " is a human")
else:
    print(filename + " is a horse")
