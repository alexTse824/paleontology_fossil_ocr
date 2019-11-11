import os
import time
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

	top_3_proba = sorted(y, reverse=True)[:3]
	top_3_label = [label[y.index(i)] for i in top_3_proba]

	predict_proba = max(y)
	predict_label = label[y.index(predict_proba)]
	
	top_3_proba = [round(i * 100, 2) for i in top_3_proba]

	return [list(i) for i in zip(top_3_label, top_3_proba)]

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
	start_time = time.time()
	if request.method == 'POST':
		file = request.files['file']
		file_path = os.path.join(UPLOAD_FOLDER, file.filename)
		file_path = os.path.join(UPLOAD_FOLDER, file.filename)
		file.save(file_path)
		ret = predict(file_path)
		print('-' * 100)
		cost = round(time.time() - start_time, 3) * 1000
		print('Time cost: {} ms'.format(cost))
		print('-' * 100)
		return jsonify({'ret': ret, 'predictCost': cost})
	return '<h1>Welcome to FossilOCR Server</h1>'

if __name__ == '__main__':
	load_weight(WEIGHT)
	app.run(
		debug=True,
		host='0.0.0.0',
		port='8000',
	)
