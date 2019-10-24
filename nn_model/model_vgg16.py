import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

from settings import DIR_straitified_dataset, DIR_weight, CURRENT_TIME

# fix OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'

STRAITIFIED_NAME = 'KFold.5.20191020093348'
STRAITIFIED_DIR_NAME = os.path.join(STRAITIFIED_NAME, '0')
DATASET_DIR = os.path.join(DIR_straitified_dataset, STRAITIFIED_DIR_NAME)

TRAIN_SAMPLE_NUM = 0
VALIDATION_SAMPLE_NUM = 0

for root, dir_name, file_name in os.walk(os.path.join(DATASET_DIR, 'train')):
    TRAIN_SAMPLE_NUM += len(file_name)

for root, dir_name, file_name in os.walk(
        os.path.join(DATASET_DIR, 'validation')):
    VALIDATION_SAMPLE_NUM += len(file_name)

BATCH_SIZE = 64
EPOCHS = 50

IMG_WIDTH = 224
IMG_HEIGHT = 224

model = Sequential()
model.add(
    VGG16(include_top=False,
          weights='imagenet',
          input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

top_model = Sequential()
# vgg16 not include_top's output_shape: (None, 7, 7, 512)
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(9, activation='softmax'))
model.add(top_model)

sgd = SGD(lr=1e-4)

model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'train'),
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'validation'),
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True)

checkpoint = ModelCheckpoint(filepath=os.path.join(f'{DIR_weight}.0.{BATCH_SIZE}.{EPOCHS}.{CURRENT_TIME}.h5'),
                             monitor="val_acc",
                             verbose=2,
                             save_best_only=False,
                             save_weights_only=False,
                             mode='max',
                             period=1)

model.fit_generator(train_generator,
                    steps_per_epoch=TRAIN_SAMPLE_NUM // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=validation_generator,
                    validation_steps=VALIDATION_SAMPLE_NUM // BATCH_SIZE,
                    callbacks=[checkpoint],
                    verbose=1)
