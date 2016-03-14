# Trying out Blocks
# Code is based of https://ift6266h16.wordpress.com/class-project/getting-started/

from fuel.datasets.dogs_vs_cats import DogsVsCats
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers.image import RandomFixedSizeCrop
from fuel.transformers import Flatten
from fuel.transformers.image import MinimumImageDimensions

import theano
from theano import tensor
import numpy as np

# Use Blocks to train this network
from blocks.algorithms import GradientDescent, Momentum
from blocks.extensions import Printing,FinishAfter
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop

# Load the training set
train = DogsVsCats(('train',), subset=slice(0, 20000))

# We now create a "stream" over the dataset which will return shuffled batches
# of size 128. Using the DataStream.default_stream constructor will turn our
# 8-bit images into floating-point decimals in [0, 1].
stream = DataStream.default_stream(
    train,
    iteration_scheme=ShuffledScheme(train.num_examples, 128))

# Enlarge images that are too small
downnscale_stream = MinimumImageDimensions(
    stream,(64,64),which_sources=('image_features',))


# Our images are of different sizes, so we'll use a Fuel transformer
# to take random crops of size (32 x 32) from each image
cropped_stream = RandomFixedSizeCrop(
    downnscale_stream, (32, 32), which_sources=('image_features',))

# We'll use a simple MLP, so we need to flatten the images
# from (channel, width, height) to simply (features,)
flattened_stream = Flatten(
    cropped_stream, which_sources=('image_features',))

# Create the Theano MLP
import theano
from theano import tensor
import numpy

X = tensor.matrix('image_features')
T = tensor.lmatrix('targets')

W = theano.shared(
    numpy.random.uniform(low=-0.01, high=0.01, size=(3072, 500)), 'W')
b = theano.shared(numpy.zeros(500))
V = theano.shared(
    numpy.random.uniform(low=-0.01, high=0.01, size=(500, 2)), 'V')
c = theano.shared(numpy.zeros(2))
params = [W, b, V, c]

H = tensor.nnet.sigmoid(tensor.dot(X, W) + b)
Y = tensor.nnet.softmax(tensor.dot(H, V) + c)

loss = tensor.nnet.categorical_crossentropy(Y, T.flatten()).mean()

# Use Blocks to train this network
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions import Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop

algorithm = GradientDescent(cost=loss, parameters=params,
                            step_rule=Scale(learning_rate=0.1))

# We want to monitor the cost as we train
loss.name = 'loss'
extensions = [TrainingDataMonitoring([loss], every_n_batches=1),
              Printing(every_n_batches=1)]

main_loop = MainLoop(data_stream=flattened_stream, algorithm=algorithm,
                     extensions=extensions)
main_loop.run()
