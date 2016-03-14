

"""convolutional network example.
The original version of this code come from : https://github.com/mila-udem/blocks-examples/blob/master/mnist_lenet/
This code have been developed with the help of  :
Vincent Antaki (the code of the ScikitResize come from him)
"""

import logging
import numpy
from argparse import ArgumentParser

from theano import tensor

from blocks.algorithms import GradientDescent, Scale
from blocks.bricks import (MLP, Rectifier, Initializable, FeedforwardSequence,Softmax, Activation)
from blocks.bricks.conv import (Convolutional, ConvolutionalSequence, Flattener, MaxPooling)
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import (DataStreamMonitoring,TrainingDataMonitoring)
from blocks.extensions.saveload import Checkpoint
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, Uniform
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from toolz.itertoolz import interleave
#from plot import Plot
from ScikitResize import ScikitResize
from fuel.streams import ServerDataStream
import socket
import datetime

class LeNet(FeedforwardSequence, Initializable):

    def __init__(self, conv_activations, num_channels, image_shape,
                 filter_sizes, feature_maps, pooling_sizes,
                 top_mlp_activations, top_mlp_dims,
                 conv_step=None, border_mode='valid', **kwargs):
        if conv_step is None:
            self.conv_step = (1, 1)
        else:
            self.conv_step = conv_step
        self.num_channels = num_channels
        self.image_shape = image_shape
        self.top_mlp_activations = top_mlp_activations
        self.top_mlp_dims = top_mlp_dims
        self.border_mode = border_mode

        conv_parameters = zip(filter_sizes, feature_maps)

        # Construct convolutional layers with corresponding parameters
        self.layers = list(interleave([
            (Convolutional(filter_size=filter_size,
                           num_filters=num_filter,
                           step=self.conv_step,
                           border_mode=self.border_mode,
                           name='conv_{}'.format(i))
             for i, (filter_size, num_filter)
             in enumerate(conv_parameters)),
            conv_activations,
            (MaxPooling(size, name='pool_{}'.format(i))
             for i, size in enumerate(pooling_sizes))]))

        self.conv_sequence = ConvolutionalSequence(self.layers, num_channels,
                                                   image_size=image_shape)

        # Construct a top MLP
        self.top_mlp = MLP(top_mlp_activations, top_mlp_dims)

        # We need to flatten the output of the last convolutional layer.
        # This brick accepts a tensor of dimension (batch_size, ...) and
        # returns a matrix (batch_size, features)
        self.flattener = Flattener()
        application_methods = [self.conv_sequence.apply, self.flattener.apply,
                               self.top_mlp.apply]
        super(LeNet, self).__init__(application_methods, **kwargs)

    @property
    def output_dim(self):
        return self.top_mlp_dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.top_mlp_dims[-1] = value

    def _push_allocation_config(self):
        self.conv_sequence._push_allocation_config()
        conv_out_dim = self.conv_sequence.get_dim('output')

        self.top_mlp.activations = self.top_mlp_activations
        self.top_mlp.dims = [numpy.prod(conv_out_dim)] + self.top_mlp_dims


def main():
    feature_maps = [20, 50]
    mlp_hiddens = [50]
    conv_sizes = [5,5]
    pool_sizes = [3,3]
    save_to="DvC.pkl"
    batch_size=500
    image_size = (32, 32)
    output_size = 2
    learningRate=0.1
    num_epochs=10
    num_batches=None
    #host_plot = 'http://localhost:5006/'

    # Use ReLUs everywhere and softmax for the final prediction
    conv_activations = [Rectifier() for _ in feature_maps]
    mlp_activations = [Rectifier() for _ in mlp_hiddens] + [Softmax()]
    convnet = LeNet(conv_activations, 3, image_size,
                    filter_sizes=zip(conv_sizes, conv_sizes),
                    feature_maps=feature_maps,
                    pooling_sizes=zip(pool_sizes, pool_sizes),
                    top_mlp_activations=mlp_activations,
                    top_mlp_dims=mlp_hiddens + [output_size],
                    border_mode='full',
                    weights_init=Uniform(width=.2),
                    biases_init=Constant(0))
    # We push initialization config to set different initialization schemes
    # for convolutional layers.
    convnet.push_initialization_config()
    convnet.layers[0].weights_init = Uniform(width=.2)
    convnet.layers[1].weights_init = Uniform(width=.09)
    convnet.top_mlp.linear_transformations[0].weights_init = Uniform(width=.08)
    convnet.top_mlp.linear_transformations[1].weights_init = Uniform(width=.11)
    convnet.initialize()
    logging.info("Input dim: {} {} {}".format(*convnet.children[0].get_dim('input_')))
    for i, layer in enumerate(convnet.layers):
        if isinstance(layer, Activation):
            logging.info("Layer {} ({})".format(i, layer.__class__.__name__))
        else:
            logging.info("Layer {} ({}) dim: {} {} {}".format(
                i, layer.__class__.__name__, *layer.get_dim('output')))
    x = tensor.tensor4('image_features')
    y = tensor.lmatrix('targets')

    # Normalize input and apply the convnet
    probs = convnet.apply(x)
    cost = (CategoricalCrossEntropy().apply(y.flatten(), probs)
            .copy(name='cost'))
    error_rate = (MisclassificationRate().apply(y.flatten(), probs)
                  .copy(name='error_rate'))

    cg = ComputationGraph([cost, error_rate])

    ########### GET THE DATA #####################

    from fuel.datasets.dogs_vs_cats import DogsVsCats
    from fuel.streams import DataStream, ServerDataStream
    from fuel.schemes import ShuffledScheme
    from fuel.transformers.image import RandomFixedSizeCrop, MinimumImageDimensions, Random2DRotation
    from fuel.transformers import Flatten, Cast, ScaleAndShift

    def create_data(data):
        stream = DataStream.default_stream(data, iteration_scheme=ShuffledScheme(data.num_examples, batch_size))
        stream_downscale = MinimumImageDimensions(stream, image_size, which_sources=('image_features',))
        #stream_rotate = Random2DRotation(stream_downscale, which_sources=('image_features',))
        stream_max = ScikitResize(stream_downscale, image_size, which_sources=('image_features',))
        stream_scale = ScaleAndShift(stream_max, 1./255, 0, which_sources=('image_features',))
        stream_cast = Cast(stream_scale, dtype='float32', which_sources=('image_features',))
        #stream_flat = Flatten(stream_scale, which_sources=('image_features',))
        return stream_cast


    stream_data_train = create_data(DogsVsCats(('train',), subset=slice(0, 20000)))
    stream_data_test = create_data(DogsVsCats(('train',), subset=slice(20000, 25000)))

    # Train with simple SGD
    algorithm = GradientDescent(cost=cost, parameters=cg.parameters,step_rule=Scale(learning_rate=learningRate))


    # `Timing` extension reports time for reading data, aggregating a batch
    # and monitoring;
    # `ProgressBar` displays a nice progress bar during training.
    extensions = []
    extensions.append(Timing())
    extensions.append(FinishAfter(after_n_epochs=num_epochs,after_n_batches=num_batches))
    extensions.append(DataStreamMonitoring([cost, error_rate],stream_data_test,prefix="valid"))
    extensions.append(TrainingDataMonitoring([cost, error_rate,aggregation.mean(algorithm.total_gradient_norm)],prefix="train",after_epoch=True))
    extensions.append(Checkpoint(save_to))
    extensions.append(ProgressBar())
    extensions.append(Printing())

    #Adding a live plot with the bokeh server
    #type http://localhost:5006 in your browser to visualise the result (you need also to install bokeh 0.10.0 and not 0.11.0)

    """
    extensions.append(Plot('%s %s @ %s' % ('CNN ', datetime.datetime.now(), socket.gethostname()),
			channels=[['valid_cost', 'valid_error_rate'],
			 ['train_total_gradient_norm']], after_epoch=True, server_url=host_plot))
    """


    model = Model(cost)

    main_loop = MainLoop(
        algorithm,
        stream_data_train,
        model=model,
        extensions=extensions)

    main_loop.run()


if __name__ == "__main__":
    main()
