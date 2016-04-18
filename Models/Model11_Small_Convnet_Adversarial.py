"""
Convolutional network example using adversial Networks
Inspired from GoogleNet inception and based on the following paper
Ian J. Goodfellow, Jonathon Shlens & Christian Szegedy, "Explaining and harnessing adversarial examples", March 2015, ICLR 2015
ADVERSARIAL EXAMPLES
The original version of this code come from LeNetConvNet.py
https://github.com/mila-udem/blocks-examples/blob/master/mnist_lenet/
"""

import logging
import numpy
from argparse import ArgumentParser

from theano import tensor

from blocks.algorithms import GradientDescent, Scale, Adam, Momentum
from Algo_Momentum import NesterovAdam, NesterovMomentum
from blocks.bricks import (MLP, BatchNormalization, SpatialBatchNormalization, BatchNormalizedMLP,
Rectifier, Initializable, FeedforwardSequence,Softmax, Activation)
from blocks.bricks.conv import (Convolutional, ConvolutionalSequence, Flattener, MaxPooling)
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import (DataStreamMonitoring,TrainingDataMonitoring)
from blocks.extensions.saveload import Checkpoint
from blocks.graph import (ComputationGraph, apply_batch_normalization,
get_batch_normalization_updates, apply_dropout, apply_noise)
from blocks.filter import get_brick, VariableFilter
from blocks.roles import add_role, PARAMETER, WEIGHT, FILTER, INPUT, OUTPUT, DROPOUT
from blocks.initialization import Constant, Uniform, Orthogonal, IsotropicGaussian
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from toolz.itertoolz import interleave
from fuel_transformers_image import MaximumImageDimensions, RandomHorizontalSwap
from RRELU_ELU import Rand_Leaky_Rectifier, ELU
#from ScikitResize import ScikitResize
import socket
import datetime

logger = logging.getLogger(__name__)


def intranet(i, j, out, image_size, filter_size, num_filter, num_channels, pooling_size, conv_step, border_mode, conv_activation):


    conv_layersA = [] #first intra convolutional sequence

    conv_layersA.append(
        Convolutional(
            filter_size=filter_size[j],
            num_filters=num_filter[j],
            step=conv_step,
            border_mode=border_mode,
            name='conv_A{}({})'.format(i,j)))
    conv_layersA.append(BatchNormalization(name='BNconv_A{}({})'.format(i,j)))
    conv_layersA.append(conv_activation[0])

    j = j + 1 #next sub layer
    conv_layersA.append(
        Convolutional(
            filter_size=filter_size[j],
            num_filters=num_filter[j],
            step=conv_step,
            border_mode=border_mode,
            name='conv_A{}({})'.format(i,j)))
    conv_layersA.append(BatchNormalization(name='BNconv_A{}({})'.format(i,j)))
    conv_layersA.append(conv_activation[0])


    conv_sequenceA = ConvolutionalSequence(conv_layersA, num_channels=num_channels, image_size=image_size,
    weights_init=Uniform(width=0.2), use_bias=False, name='convSeq_A{}'.format(i))
    out1 = conv_sequenceA.apply(out)


    conv_layersB = [] #second intra convolutional sequence

    j = j + 1 #next sub layer
    conv_layersB.append(
        Convolutional(
            filter_size=filter_size[j],
            num_filters=num_filter[j],
            step=conv_step,
            border_mode=border_mode,
            name='conv_B{}({})'.format(i,j)))
    conv_layersB.append(BatchNormalization(name='BNconv_B{}({})'.format(i,j)))
    conv_layersB.append(conv_activation[0])

    j = j + 1 #next sub layer
    conv_layersB.append(
        Convolutional(
            filter_size=filter_size[j],
            num_filters=num_filter[j],
            step=conv_step,
            border_mode=border_mode,
            name='conv_B{}({})'.format(i,j)))
    conv_layersB.append(BatchNormalization(name='BNconv_B{}({})'.format(i,j)))
    conv_layersB.append(conv_activation[0])

    conv_sequenceB = ConvolutionalSequence(conv_layersB, num_channels=num_channels, image_size=image_size,
    weights_init=Uniform(width=0.2), use_bias=False, name='convSeq_B{}'.format(i))
    out2 = conv_sequenceB.apply(out)

    #Merge
    return tensor.concatenate([out1, out2], axis=1)


# Global parameters
num_epochs = 500
learning_rate = 0.0005
batch_size = 64
num_batches = None
image_size = (128,128)
output_size = 2
mode = "GPU_run" # "CPU_test" or "GPU_run"
host_plot='http://hades:5090'
save_to = "ModelSimpleConvNet128_Nesterov.pkl"
graph_name = 'CNN_convnet_128_Nesterov'

# Activation functions
conv_activation = [Rectifier()]
mlp_activation = [Rectifier()] + [Softmax()]

# data variables
x = tensor.tensor4('image_features')
y = tensor.lmatrix('targets')


################# Convolutional Sequence 1 #################
# conv_layers1 parameters
i = 0 #Sequence
j = 0 #Sub Layer
filter_size = [(7,7), (5,5)]
num_filter = [32, 64]
num_channels = 3
pooling_size = [(3,3), (2,2)]
conv_step = (1,1)
border_mode = 'valid'


conv_layers1 = []
conv_layers1.append(SpatialBatchNormalization(name='spatialBN_{}'.format(i)))
conv_layers1.append(
    Convolutional(
        filter_size=filter_size[j],
        num_filters=num_filter[j],
        step=conv_step,
        border_mode=border_mode,
        name='conv_{}'.format(i)))
conv_layers1.append(BatchNormalization(name='BNconv_{}'.format(i)))
conv_layers1.append(conv_activation[j])
conv_layers1.append(MaxPooling(pooling_size[j], name='pool_{}'.format(i)))

i = i + 1 #Sequence
conv_layers1.append(
    Convolutional(
        filter_size=filter_size[j+1],
        num_filters=num_filter[j+1],
        step=conv_step,
        border_mode=border_mode,
        name='conv_{}'.format(i)))
conv_layers1.append(BatchNormalization(name='BNconv_{}'.format(i)))
conv_layers1.append(conv_activation[0])
conv_layers1.append(MaxPooling(pooling_size[j+1], name='pool_{}'.format(i)))

conv_sequence = ConvolutionalSequence(conv_layers1, num_channels=num_channels, image_size=image_size,
weights_init=Uniform(width=0.2), biases_init=Constant(0.), name='ConvSeq_{}'.format(i))
conv_sequence.initialize()
out = conv_sequence.apply(x)
#out = Flattener().apply(conv_sequence.apply(x))


################# Convolutional Sequence 2A and 2B #################
# conv_layers2 parameters
i = i + 1 #Sequence
j = 0 #Sub Layer
# 1st and 2nd are sequential; 3rd and 4th are sequential; 2 sequences are concatenated
filter_size = [(1,1),(5,5),(1,1),(3,3)]
num_filter = [32, 64, 64, 96]
num_channels = 3
pooling_size = None
conv_step = (1,1)
border_mode = 'half'

out = intranet(i, j, out, image_size, filter_size, num_filter, num_channels, pooling_size, conv_step, border_mode, conv_activation)
out_intra2 = MaxPooling((2,2), name='poolLow').apply(out)


################# Convolutional Sequence 3A and 3B #################
# conv_layers2 parameters
i = i + 1 #Sequence
j = 0 #Sub Layer
# 1st and 2nd are sequential; 3rd and 4th are sequential; 2 sequences are concatenated
filter_size = [(1,1),(3,3),(1,1),(2,2)]
num_filter = [16, 32, 32, 64]
num_channels = 1
pooling_size = None
conv_step = (1,1)
border_mode = 'half'

out_intra3 = intranet(i, j, out, image_size, filter_size, num_filter, num_channels, pooling_size, conv_step, border_mode, conv_activation)


################# concatenate intranet 2 and 3 #################
out=tensor.concatenate([out_intra2, out_intra3], axis=1)


################# Convolutional Sequence 4 #################
# conv_layers4 parameters
i = i + 1 #Sequence
j = 0 #Sub Layer
filter_size = [(1,1), (1,1)]
num_filter = [64, 128]
num_channels = 3
pooling_size = [(1,1), (1,1)]
conv_step = (1,1)
border_mode = 'half'


conv_layers4 = []
conv_layers4.append(SpatialBatchNormalization(name='spatialBN_{}'.format(i)))
conv_layers4.append(
    Convolutional(
        filter_size=filter_size[j],
        num_filters=num_filter[j],
        step=conv_step,
        border_mode=border_mode,
        name='conv_{}'.format(i)))
conv_layers4.append(BatchNormalization(name='BNconv_{}'.format(i)))
conv_layers4.append(conv_activation[j])
conv_layers4.append(MaxPooling(pooling_size[j], name='pool_{}'.format(i)))

i = i + 1 #Sequence
conv_layers4.append(
    Convolutional(
        filter_size=filter_size[j+1],
        num_filters=num_filter[j+1],
        step=conv_step,
        border_mode=border_mode,
        name='conv_{}'.format(i)))
conv_layers4.append(BatchNormalization(name='BNconv_{}'.format(i)))
conv_layers4.append(conv_activation[0])
conv_layers4.append(MaxPooling(pooling_size[j+1], name='pool_{}'.format(i)))

conv_sequence4 = ConvolutionalSequence(conv_layers1, num_channels=num_channels, image_size=image_size,
weights_init=Uniform(width=0.2), biases_init=Constant(0.), name='ConvSeq_{}'.format(i))
conv_sequence4.initialize()
out = Flattener().apply(conv_sequence4.apply(out))


################# Final MLP layers #################
# MLP parameters
mlp_hiddens = 1000

top_mlp_dims = [numpy.prod(conv_sequence4.get_dim('output'))] + [mlp_hiddens] + [output_size]
top_mlp = MLP(mlp_activation, top_mlp_dims,weights_init=Uniform(width=0.2),biases_init=Constant(0.))
top_mlp.initialize()

probs = top_mlp.apply(out)
cost = CategoricalCrossEntropy(name='Cross1').apply(y.flatten(), probs).copy(name='cost1')

error_rate = (MisclassificationRate().apply(y.flatten(), probs).copy(name='error_rate'))
error_rate2 = error_rate.copy(name='error_rate2')
cg = ComputationGraph([cost, error_rate])
weights = VariableFilter(roles=[FILTER, WEIGHT])(cg.variables)


########### Loading images #####################

from fuel.datasets.dogs_vs_cats import DogsVsCats
from fuel.streams import DataStream, ServerDataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers.image import RandomFixedSizeCrop, MinimumImageDimensions, Random2DRotation
from fuel.transformers import Flatten, Cast, ScaleAndShift

def create_data(data):

    stream = DataStream(data, iteration_scheme=ShuffledScheme(data.num_examples, batch_size))

    # Data Augmentation
    stream = MinimumImageDimensions(stream, image_size, which_sources=('image_features',))
    stream = MaximumImageDimensions(stream, image_size, which_sources=('image_features',))
    stream = RandomHorizontalSwap(stream, which_sources=('image_features',))
    stream = Random2DRotation(stream, which_sources=('image_features',))
    #stream = ScikitResize(stream, image_size, which_sources=('image_features',))

    # Data Preprocessing

    # Data Transformation
    stream = ScaleAndShift(stream, 1./255, 0, which_sources=('image_features',))
    stream = Cast(stream, dtype='float32', which_sources=('image_features',))
    return stream


if mode is "CPU_test":
    stream_data_train = create_data(DogsVsCats(('train',), subset=slice(0, 100)))
    stream_data_valid = create_data(DogsVsCats(('train',), subset=slice(100, 110)))
if mode is  "GPU_run":
    stream_data_train = create_data(DogsVsCats(('train',), subset=slice(0, 22500)))
    stream_data_valid = create_data(DogsVsCats(('train',), subset=slice(22500, 25000)))
if mode is "data_server":
    stream_data_train = ServerDataStream(('image_features','targets'), False, port=5560)
    stream_data_valid = ServerDataStream(('image_features','targets'), False, port=5561)

########### Running model #####################

algorithm = GradientDescent(cost=cost, parameters=cg.parameters,step_rule=Momentum(learning_rate=learning_rate, momentum=0.995))

extensions = []
extensions.append(Timing())
extensions.append(FinishAfter(after_n_epochs=num_epochs,after_n_batches=num_batches))
extensions.append(DataStreamMonitoring([cost, error_rate],stream_data_valid,prefix="valid"))
extensions.append(TrainingDataMonitoring([cost, error_rate,aggregation.mean(algorithm.total_gradient_norm)],prefix="train",after_epoch=True))
extensions.append(ProgressBar())
extensions.append(Printing())

if mode == ("GPU_run" or "data_server"):
    try:
        from plot import Plot
        extensions.append(Plot('%s %s @ %s' % (graph_name, datetime.datetime.now(), socket.gethostname()),
                            channels=[['train_error_rate', 'valid_error_rate'],
                             ['train_total_gradient_norm']], after_epoch=True, server_url=host_plot))
        PLOT_AVAILABLE = True
    except ImportError:
        PLOT_AVAILABLE = False
    extensions.append(Checkpoint(save_to, after_epoch=True, after_training=True, save_separately=['log']))



logger.info("Building the model")

model = Model(cost)

main_loop = MainLoop(
    algorithm,
    stream_data_train,
    model=model,
    extensions=extensions)

main_loop.run()
