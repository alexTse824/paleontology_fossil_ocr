import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard

from preprocess.tfrecords_convert import _parse_record

os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 100

train_tfre_file = '/Users/xie/Code/paleontology_fossil_ocr/data/tfrecords_data/raw_data_224_mixed_tfs/0/train.0.tfrecords'
validation_tfre_file = '/Users/xie/Code/paleontology_fossil_ocr/data/tfrecords_data/raw_data_224_mixed_tfs/0/validation.0.tfrecords'

ds_train = tf.data.TFRecordDataset(filenames=train_tfre_file).map(_parse_record)
ds_train = ds_train.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch()




model = Sequential()
# vgg16 not include_top's output_shape: (None, 7, 7, 512)
model.add(
    VGG16(include_top=False,
          weights='imagenet',
          input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(Flatten(input_shape=model.output_shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(9, activation='softmax'))

model.compile(optimizer=SGD(lr=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(ds_train, )
# train_datagen = ImageDataGenerator(rescale=1. / 255,
#                                    shear_range=0.2,
#                                    zoom_range=0.2,
#                                    horizontal_flip=True)
#
# validation_datagen = ImageDataGenerator(rescale=1. / 255)
#
#
# train_generator = train_datagen.fl
#
# train_generator = train_datagen.flow_from_directory(
#     os.path.join(DATASET_DIR, 'train'),
#     target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     shuffle=True)
#
# validation_generator = validation_datagen.flow_from_directory(
#     os.path.join(DATASET_DIR, 'validation'),
#     target_size=(IMG_WIDTH, IMG_HEIGHT),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     shuffle=True)

# checkpoint = ModelCheckpoint(filepath=WEIGHT_FILE,
#                              monitor="val_acc",
#                              verbose=2,
#                              save_best_only=False,
#                              save_weights_only=False,
#                              mode='max',
#                              period=1)

# tbCallBack = TensorBoard(log_dir=DIR_log,
#                          histogram_freq=0,
#                          batch_size=BATCH_SIZE,
#                          write_grads=True,
#                          write_graph=True,
#                          write_images=True,
#                          embeddings_freq=0,
#                          embeddings_layer_names=None,
#                          embeddings_metadata=None)


# model.fit_generator(train_generator,
#                     steps_per_epoch=TRAIN_SAMPLE_NUM // BATCH_SIZE,
#                     epochs=EPOCHS,
#                     validation_data=validation_generator,
#                     validation_steps=VALIDATION_SAMPLE_NUM // BATCH_SIZE,
#                     callbacks=[checkpoint, tbCallBack],
#                     verbose=1)

