# import packages
from keras.applications import imagenet_utils
from imutils import paths
import pickle
import imutils
import h5py
import cv2
import keras
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing import image
import utils

import tensorflow as tf
from tensorflow.python.framework import ops


from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import argparse

from keras.applications import VGG16
from keras import backend as K

def deprocess_image(x):
    # Util function to convert a tensor into a valid image.
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def load_image(path, target_size=(224, 224)):
    x = image.load_img(path, target_size=target_size)
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x


def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)


def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        new_model = VGG16(weights='imagenet')
    return new_model


def guided_backpropagation(img_tensor, model, activation_layer):
    model_input = model.input
    layer_output = model.get_layer(activation_layer).output

    max_output = K.max(layer_output, axis=3)

    get_output = K.function([model_input], [K.gradients(max_output, model_input)[0]])
    saliency = get_output([img_tensor])

    return saliency[0]

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required = True,
    help = "path to HDF5 database")
ap.add_argument("-i", "--dataset", required = True,
    help = "path to the input images dataset")
ap.add_argument("-m", "--model", required = True,
    help = "path to trained orientation model")
args = vars(ap.parse_args())

# load the label names (angles) from HDF5 dataset
db = h5py.File(args["db"])
labelNames = [int(angle) for angle in db["label_names"][:]]
db.close()

# grab the paths to the testing images and randomly sample them
print("[INFO] sampling images...")

imagePaths = list(paths.list_images(args["dataset"]))
imagePaths = np.random.choice(imagePaths, size = (10,), replace = False)

# load the VGG16 network
print("[INFO] loading network...")
vgg = VGG16(weights = "imagenet", include_top = False)

# load orientation model
print("[INFO] loading model...")
model = pickle.loads(open(args["model"], "rb").read())
print(model.summary())

# loop over the image paths
for imagePath in imagePaths:
    # load the image via OpenCV
    orig = cv2.imread(imagePath)

    # load the input image using Keras util function
    # and make sure to resize the image to (224, 224)
    image = load_img(imagePath, target_size = (224, 224))
    image = img_to_array(image)

    # preprocess the image by
    # (1) expanding dimensions
    # (2) subtracting the mean RGB pixel intensity from ImageNet dataset
    image = np.expand_dims(image, axis = 0)
    image = imagenet_utils.preprocess_input(image)

    # pass the image through the network to obatin the feature vector
    features = vgg.predict(image)
    features = features.reshape((features.shape[0], 512 * 7 * 7))

    # pass the CNN features thorugh Logistic Regression classififer
    # to obatin the orientation predictions
    angle = model.predict(features)
    angle = labelNames[angle[0]]

    # correct the image based on the predictions
    rotated = imutils.rotate_bound(orig, 360 - angle)

    register_gradient()
    model1 = VGG16(weights='imagenet')
    guided_model = modify_backprop(model1, 'GuidedBackProp')
    gradient = guided_backpropagation(orig, guided_model, "block5_pool")
    # display the original and corrected images
    cv2.imshow("Original", orig)
    cv2.imshow("Corrected", rotated)
    cv2.imshow(deprocess_image(gradient))
    cv2.waitKey(0)

cv2.destroyAllWindows()
