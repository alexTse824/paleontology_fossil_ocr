import os
import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard

sys.path.append('.')
from settings import DIR_straitified_dataset, DIR_weight, DIR_log

# fix OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'

BATCH_SIZE = 64
EPOCHS = 100

IMG_WIDTH = 224
IMG_HEIGHT = 224

TRAIN_SAMPLE_NUM = 0
VALIDATION_SAMPLE_NUM = 0

STRAITIFIED_NAME = 'KFold.5'
SET_INDEX = '0'
STRAITIFIED_DIR_NAME = os.path.join(STRAITIFIED_NAME, SET_INDEX)
DATASET_DIR = os.path.join(DIR_straitified_dataset, STRAITIFIED_DIR_NAME)
WEIGHT_FILE = os.path.join(DIR_weight, f'{STRAITIFIED_NAME}.SET{SET_INDEX}.BATCH{BATCH_SIZE}.EPOCHS{EPOCHS}.h5')

for root, dir_name, file_name in os.walk(os.path.join(DATASET_DIR, 'train')):
    TRAIN_SAMPLE_NUM += len(file_name)

for root, dir_name, file_name in os.walk(
        os.path.join(DATASET_DIR, 'validation')):
    VALIDATION_SAMPLE_NUM += len(file_name)


model = Sequential()
# vgg16 not include_top's output_shape: (None, 7, 7, 512)
model.add(
    VGG16(include_top=False,
          weights='imagenet',
          input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
model.add(Flatten(input_shape=model.output_shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(9, activation='softmax'))

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

checkpoint = ModelCheckpoint(filepath=WEIGHT_FILE,
                             monitor="val_acc",
                             verbose=2,
                             save_best_only=False,
                             save_weights_only=False,
                             mode='max',
                             period=1)

tbCallBack = TensorBoard(log_dir=DIR_log,
                         histogram_freq=0,
                         batch_size=BATCH_SIZE,
                         write_grads=True,
                         write_graph=True,
                         write_images=True,
                         embeddings_freq=0,
                         embeddings_layer_names=None,
                         embeddings_metadata=None)

if os.path.exists(WEIGHT_FILE):
    model.load_weights(WEIGHT_FILE)
# TODO: validation_data为验证集, 应从训练集中分离, 使用validaiton_split比例划分(fit方法), 另使用test_set进行predict进行测试accuracy
model.fit_generator(train_generator,
                    steps_per_epoch=TRAIN_SAMPLE_NUM // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=validation_generator,
                    validation_steps=VALIDATION_SAMPLE_NUM // BATCH_SIZE,
                    callbacks=[checkpoint, tbCallBack],
                    verbose=1)
