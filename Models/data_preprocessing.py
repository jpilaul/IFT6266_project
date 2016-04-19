"""
You will find here a series of data augmentation and preprocessing operations
Inspired by César Laurent's code

Based on the papaer from A. Krizhevsky, I. Sutskever, G. Hinton,
"ImageNet Classification with Deep Convolutional Neural Networks"
http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
presentation data augmentation:
http://rogerioferis.com/VisualRecognitionAndSearch2014/material/presentations/GuangnanAndMajaDeepLearning.pdf

"""


import numpy as np
from skimage.transform import rotate
from skimage.util import random_noise
from skimage.transform import resize


def RGB_PCA(image):
    pixels = images.reshape(-1, images.shape[-1])
    idx = np.random.random_integers(0, pixels.shape[0], 1000000)
    pixels = [pixels[i] for i in idx]
    pixels = np.array(pixels, dtype=np.uint8).T
    m = np.mean(pixels)/256.
    C = np.cov(pixels)/(256.*256.)
    l, v = np.linalg.eig(C)
    return l, v, m


def RGB_variations(image):
    eigen_vectors = np.array([[-0.57070609,-0.58842455,-0.57275746],
                           [ 0.72758705,-0.03900872,-0.68490539],
                           [ 0.38067261,-0.8076106,0.4503926 ]],"float32").T
    eigen_values = np.array([ 187.01200611,9.4903197,2.52593616 ], "float32")
    alpha = np.random.randn(3)/10
    var = eigen_values*alpha
    variation = np.dot(eigen_vectors, var)
    return image + variation


def convert_to_grayscale(image):
    return PIL.Image.fromarray(img).convert("L")



def noise(image):
    r = np.random.rand(1)[0]
    # TODO randomize parameters of the noises; check how to init seed
    if r < 0.33:
        return random_noise(x, 's&p', seed=np.random.randint(1000000))
    if r < 0.66:
        return random_noise(x, 'gaussian', seed=np.random.randint(1000000))
    return random_noise(x, 'speckle', seed=np.random.randint(1000000))


def data_preprocessing1(image, image_size):
    height = image_size[0]
    width = image_size[1]
    # 1. Resize
    if np.random.rand(1)[0] > 0.2:
        # Randomly zoom
        x = np.random.randint(height, height*1.2)
        image = resize(image, (height + x, width + x, 3))
        # Ranodmly crop
        x, y = np.random.randint(0, x, 2)
        image = image[x:height+x, y:width+y, :]
    else:
        image = resize(image, (height, width, 3))
    # 2. Normalize
    image = image.astype(floatX)
    # 3. Equalize
    image = RGB_variations(image, params.RGB_eig_val, params.RGB_eig_vec)
    # 4. Remove mean
    image = image - params.RGB_mean
    # 5. Reshape for theano's convolutoins
    image = np.rollaxis(image, 2, 0)
    return image


def data_preprocessing2(image, image_size):
    height = image_size[0]
    width = image_size[1]
    # 1. Resize
    if np.random.rand(1)[0] > 0.2:
        # Randomly zoom
        x = np.random.randint(height, height*1.2)
        image = resize(image, (height + x, width + x, 3))
        # Ranodmly crop
        x, y = np.random.randint(0, x, 2)
        image = image[x:height+x, y:width+y, :]
    else:
    # 2. Convert to grayscale
    image = convert_to_grayscale(image)
    # 3. Add noise
    if np.random.rand(1)[0] > 0.5:
        image = noise(image)
    # 4. Reshape for theano's convolutoins
    image = np.rollaxis(image, 2, 0)
    return image
