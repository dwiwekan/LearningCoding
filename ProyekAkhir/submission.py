import os
import zipfile
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

list_os = os.listdir('train/')
print(list_os)

# print each class of image in train folder
count = 0
for i in list_os:
    print(f'total {i} images :', len(os.listdir(f'train/{i}')))
    count += len(os.listdir(f'train/{i}'))
print('\ntotal images in train folder :', count)
# print total images in train folder

  
train_dir = os.path.join('train/')
train_datagen = ImageDataGenerator(
    rescale=1./255,
    featurewise_center=True,         # Perform feature scaling across the dataset
    zoom_range=0.2,                  # Randomly zoom image
    rotation_range=20,               # Randomly rotate image
    fill_mode='nearest',
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.20
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_dir, # same directory as training data
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation')


labels = [k for k in train_generator.class_indices]
sample_generate = train_generator.__next__()

images = sample_generate[0]
titles = sample_generate[1]
plt.figure(figsize = (20 , 20))

for i in range(20):
    plt.title(f'Class: {labels[np.argmax(titles[i],axis=0)]}')
    plt.subplot(5 , 5, i+1)
    plt.subplots_adjust(hspace = 0.2 , wspace = 0.2)
    plt.imshow(images[i])
    plt.axis("off")
    

from tensorflow.keras.applications import MobileNet

# Load the pre-trained MobileNet model (excluding the top fully-connected layers)
base_model = MobileNet(include_top=False, input_shape=(150, 150, 3))

# Freeze the layers in the base model so they're not updated during training
base_model.trainable = False

# Create a new model by adding your classifier on top of the base model
model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # Additional Conv2D layer
    tf.keras.layers.MaxPooling2D(2, 2),  # Additional MaxPooling2D layer
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(20, activation='softmax')
])

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics = ['accuracy'])

# Callbacks 
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.85 and logs.get('val_accuracy')>0.85):
            print("\nReached 85% accuracy so cancelling training!")
            self.model.stop_training = True
            
history = model.fit(train_generator,
                    validation_data=validation_generator,
                    epochs=50,
                    batch_size=64,
                    callbacks=[myCallback()],
                    verbose=1)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Akurasi Model')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Model')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# save model
model.save('model_mobilenet.h5')
model.save_weights('model_weights_mobilenet.h5')

# convert model to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model_mobilenet.tflite', 'wb') as f:
    f.write(tflite_model)