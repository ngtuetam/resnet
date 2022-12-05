from flask import Flask, flash, request, redirect, url_for
from flask_cors import CORS, cross_origin
from predictimg import predict
import cv2 as cv
import numpy as np
from keras.utils import load_img

app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = ''


@app.route('/', methods=['POST', 'GET'])
@cross_origin(origin='*')
def process():
    if request.method == 'POST':
        img = request.files['file']
        img.save('image.jpg')
        path ="image.jpg"
        image = load_img(path, target_size=(224,224))
        res = predict(image)
        return res
    if request.method == 'GET':
        return 'get'
    return ''

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=4000)

if __name__ == '__main__':
    app.run()