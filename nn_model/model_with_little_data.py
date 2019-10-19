import os
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

from nn_model.sequential_model import Sequential
from settings import DIR_straitified_dataset

STRAITIFIED_DIR_NAME = 'KFold.5.20191019215945'
DATASET_DIR = os.path.join(DIR_straitified_dataset, STRAITIFIED_DIR_NAME)

# TRAIN_SAMPLE_NUM = len(os.listdir(os.path.join(DATASET_DIR, 'train')))
TRAIN_SAMPLE_NUM = 2000
# VALIDATION_SAMPLE_NUM = len(os.listdir(os.path.join(DATASET_DIR,
                                                    # 'validation')))
VALIDATION_SAMPLE_NUM = 2000

BATCH_SIZE = 64
EPOCHS = 50

IMG_WIDTH = 224
IMG_HEIGHT = 224

model = Sequential(STRAITIFIED_DIR_NAME)

model.add(Conv2D(32, (3, 3), input_shape=(3, IMG_WIDTH, IMG_HEIGHT)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add_conv_act_maxpool_2x2(32, (3, 3), 'relu', (2, 2))
model.add_conv_act_maxpool_2x2(64, (3, 3), 'relu', (2, 2))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# train_datagen = ImageDataGenerator(rescale=1. / 255,
#                                    shear_range=0.2,
#                                    zoom_range=0.2,
#                                    horizontal_flip=True)
# validation_datagen = ImageDataGenerator(rescale=1. / 255)

# train_generator = train_datagen.flow_from_directory(
#     os.path.join(DATASET_DIR, 'train'),
#     target_size=(IMG_WIDTH, IMG_HEIGHT),
#     batch_size=BATCH_SIZE,
#     class_mode='binary')
# validation_generator = validation_datagen.flow_from_directory(
#     os.path.join(DATASET_DIR, 'validation'),
#     target_size=(IMG_WIDTH, IMG_HEIGHT),
#     batch_size=BATCH_SIZE,
#     class_mode='binary')

# model.fit_generator(train_generator,
#                     steps_per_epoch=TRAIN_SAMPLE_NUM // BATCH_SIZE,
#                     epochs=EPOCHS,
#                     validation_data=validation_generator,
#                     validation_steps=VALIDATION_SAMPLE_NUM // BATCH_SIZE)

# model.save_weights(f'{STRAITIFIED_DIR_NAME}.weights.h5')
