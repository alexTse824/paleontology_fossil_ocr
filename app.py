import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

tf.logging.set_verbosity(tf.logging.ERROR)

UPLOAD_FOLDER = '/home/ubuntu/code/fossil_ocr_server/upload'
WEIGHT = '/home/ubuntu/code/fossil_ocr_server/KFold.5.SET0.BATCH64.EPOCHS100.h5'
TEST_IMG = '/home/ubuntu/code/fossil_ocr_server/example.jpg'

label = ['Graptolite', 'Radiolaria', 'Fossil Insect', 'Trilobite', 'Coral', 'Bivalvia', 'Cephalopoda', 'Brachiopoda', 'Fossil Fish']

model = None

def load_weight(weight_file):
	global model
	model = load_model(weight_file)
	predict(TEST_IMG)

def predict(file):
	img = load_img(file, target_size=(224, 224))
	x = img_to_array(img)
	x /= 255
	x = np.expand_dims(x, axis=0)
	y = model.predict(x).tolist()[0]

	predict_proba = max(y)
	predict_label = label[y.index(predict_proba)]

	print('-' * 100)
	print(f'File: {file}')
	print(f'Lable: {predict_label}')
	print(f'Probability: {predict_proba}')
	print('-' * 100)

	return predict_label, round(predict_proba * 100, 2)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['POST'])
def index():
	file = request.files['file']
	file_path = os.path.join(UPLOAD_FOLDER, file.filename)
	file_path = os.path.join(UPLOAD_FOLDER, file.filename)
	file.save(file_path)
	label, proba = predict(file_path)
	
	return jsonify({'label': label, 'proba': str(proba)})
if __name__ == '__main__':
	load_weight(WEIGHT)
	app.run(
		debug=True,
		host='0.0.0.0',
		port='8000',
		ssl_context='adhoc'
	)
