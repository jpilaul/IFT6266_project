# Trying out Blocks using MLP and convolution

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
from blocks.initialization import IsotropicGaussian, Uniform, Constant
from blocks.extensions import Printing,FinishAfter
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop

from toolz.itertoolz import interleave

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

#X = tensor.matrix('image_features')
X = tensor.tensor4('image_features')
T = tensor.lmatrix('targets')


"""
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
"""

"""
# Setting up the Layers
from blocks.bricks import Linear, Rectifier, Softmax, Logistic
input_to_hidden = Linear(name='input_to_hidden', input_dim=3072, output_dim=500)
H = Logistic().apply(input_to_hidden.apply(X))
hidden_to_output = Linear(name='hidden_to_output', input_dim=500, output_dim=2)
Y = Softmax().apply(hidden_to_output.apply(H))

# initializing Weights and Bias
input_to_hidden.weights_init = hidden_to_output.weights_init = IsotropicGaussian(0.01)
input_to_hidden.biases_init = hidden_to_output.biases_init = Constant(0)
input_to_hidden.initialize()
hidden_to_output.initialize()

"""


from blocks.bricks import MLP, Linear, Rectifier, Softmax, Logistic
from blocks.bricks.conv import (ConvolutionalSequence, Flattener, Convolutional, MaxPooling)
import blocks_bricks_conv as b

# Convolutional layers

filter_sizes = [5, 5]
num_filters = [50, 256]
pooling_sizes = [2, 2]
conv_step = (1,1)
border_mode = 'full'
conv_activations = [Logistic() for _ in num_filters]

conv_layers =   list(interleave([
                (Convolutional(filter_size=filter_size,
                               num_filters=num_filter,
                               step=conv_step,
                               border_mode=border_mode,
                               name='conv_{}'.format(i))
                 for i, (filter_size, num_filter)
                 in enumerate(zip(filter_sizes, num_filters))),
                 conv_activations,
                (MaxPooling(pooling_sizes, name='pool_{}'.format(i))
                for i, size in enumerate(pooling_sizes))]))



convnet = ConvolutionalSequence(conv_layers, num_channels=3,
                                image_size=(32, 32),
                                weights_init=Uniform(0, 0.2),
                                biases_init=Constant(0.))

convnet.push_initialization_config()

convnet.initialize()
conv_features = Flattener().apply(convnet.apply(X))

# MLP

mlp = MLP(activations=[Logistic(name='sigmoid_0'),
          Softmax(name='softmax_1')], dims=[256, 256, 256, 2],
          weights_init=IsotropicGaussian(0.01), biases_init=Constant(0))
[child.name for child in mlp.children]
['linear_0', 'sigmoid_0', 'linear_1', 'softmax_1']
Y = mlp.apply(conv_features)
mlp.initialize()


# Setting up the cost function
from blocks.bricks.cost import CategoricalCrossEntropy
cost = CategoricalCrossEntropy().apply(T.flatten(), Y)

from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
cg = ComputationGraph(cost)
print(VariableFilter(roles=[WEIGHT])(cg.variables))
W1, W2, W3  = VariableFilter(roles=[WEIGHT])(cg.variables)
# cost with L2 regularization
cost = cost + 0.005 * (W1 ** 2).sum() + 0.005 * (W2 ** 2).sum()
cost.name = 'cost_with_regularization'

#print(cg.variables)
#print(VariableFilter(roles=[WEIGHT])(cg.variables))

# Use Blocks to train this network
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions import Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop

algorithm = GradientDescent(cost=cost, parameters=cg.parameters,
                            step_rule=Scale(learning_rate=0.1))

# We want to monitor the cost as we train

extensions = [TrainingDataMonitoring([cost], every_n_batches=1),
              Printing(every_n_batches=1)]

main_loop = MainLoop(data_stream=flattened_stream, algorithm=algorithm,
                     extensions=extensions)
main_loop.run()
