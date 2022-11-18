from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import tensorflow as tf
from argparse import ArgumentParser
import numpy as np
import pickle
import cv2 as cv




# if __name__ == "__main__":
def predict(image):
    # parser = ArgumentParser()
    # parser.add_argument("--test-file-path", type=str, required=True)
    # parser.add_argument("--model-path", default="best_model.h5", type=str)
    # parser.add_argument("--class-names-path", default='class_names.pkl', type=str)
   
    # args = parser.parse_args()
    
    class_names_path = "class_names.pkl"
    model_path = "checkpoints/best_model.h5"

    # print('Predict using ResNet model for test file path {0}'.format(args.test_file_path)) # fix
    print('Predict using ResNet') # fix
    print('===============================================================================')

    # Loading class names
    # with open (args.class_names_path, 'rb') as fp:
    #   class_names = pickle.load(fp)

    with open (class_names_path, 'rb') as fp:
      class_names = pickle.load(fp)

    # Loading model
    # model=load_model(args.model_path)
    model = load_model(model_path)

    # Load test images
    # image = preprocessing.image.load_img(args.test_file_path, target_size=(224,224))
    # image = preprocessing.image.load_img(image, target_size=(224,224))
    input_arr = preprocessing.image.img_to_array(image)/225
    x = np.expand_dims(input_arr, axis=0)

    predictions = model.predict(x)
    label=np.argmax(predictions)
    print('Result: {}'.format(class_names[label]))

# if __name__ == "__main__":
#   path = "/home/ngtuetam/workspace/resnet/test_img/corgi.JPG"
#   image = preprocessing.image.load_img(path, target_size=(224,224))
#   predict(image)