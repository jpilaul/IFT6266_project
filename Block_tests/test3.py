
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

from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.bricks import Identity
from blocks import initialization

class FeedbackRNN(BaseRecurrent):

    def __init__(self, dim, **kwargs):

        super(FeedbackRNN, self).__init__(**kwargs)
        self.dim = dim
        self.first_recurrent_layer = SimpleRecurrent(
         dim=self.dim, activation=Identity(), name='first_recurrent_layer',
         weights_init=initialization.Identity())
        self.second_recurrent_layer = SimpleRecurrent(
         dim=self.dim, activation=Identity(), name='second_recurrent_layer',
         weights_init=initialization.Identity())
        self.children = [self.first_recurrent_layer,
                  self.second_recurrent_layer]

    @recurrent(sequences=['inputs'], contexts=[],
            states=['first_states', 'second_states'],
            outputs=['first_states', 'second_states'])
    def apply(self, inputs, first_states=None, second_states=None):
        first_h = self.first_recurrent_layer.apply(
        inputs=inputs, states=first_states + second_states, iterate=False)
        second_h = self.second_recurrent_layer.apply(
        inputs=first_h, states=second_states, iterate=False)
        return first_h, second_h

    def get_dim(self, name):
        return (self.dim if name in ('inputs', 'first_states', 'second_states')
            else super(FeedbackRNN, self).get_dim(name))


X = tensor.tensor3('image_features')
T = tensor.lmatrix('targets')


feedback = FeedbackRNN(dim=3)
feedback.initialize()
first_h, second_h = feedback.apply(inputs=X)
f = theano.function([X], [first_h, second_h])

for states in f(np.ones((3, 1, 3), dtype=theano.config.floatX)):
    print(states)
