import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'

weight_file = '/Users/xie/Code/paleontology_fossil_ocr/data/nn_data/KFold.5/weight/KFold.5.SET0.BATCH64.EPOCHS100.h5'
label = ['Graptolite', 'Radiolaria', 'Fossil Insect', 'Trilobite', 'Coral', 'Bivalvia', 'Cephalopoda', 'Brachiopoda',
         'Fossil Fish']


def predict_image(image):
    model = load_model(weight_file)
    img = load_img(image, target_size=(224, 224))
    x = img_to_array(img)
    x = x.astype('float32')
    x /= 255
    x = np.expand_dims(x, axis=0)
    y = model.predict(x).tolist()[0]

    predict_proba = max(y)
    predict_label = label[y.index(predict_proba)]

    return predict_label, predict_proba


ret = predict_image('/Users/xie/Code/paleontology_fossil_ocr/data/raw_data/Bivalvia/1.jpg')
print(ret)
