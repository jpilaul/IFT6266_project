"""
Convolutional network example using adversial Networks
Inspired from GoogleNet inception and based on the following paper
Ian J. Goodfellow, Jonathon Shlens & Christian Szegedy, "Explaining and harnessing adversarial examples", March 2015, ICLR 2015
and
S. Dieleman, K. W. Willett, J. Dambr, "Rotation-invariant convolutional neural networks for galaxy morphology prediction", March 2015

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

from data_preprocessing import data_preprocessing1, data_preprocessing2

logger = logging.getLogger(__name__)

# Global parameters
num_epochs = 500
learning_rate = 0.0005
batch_size = 64
num_batches = None
image_size = (128,128)
output_size = 2
mode = "CPU_test" # "CPU_test" or "GPU_run"
host_plot='http://hades:5090'
save_to = "Model13_Adversarial.pkl"
graph_name = 'Model13_Adversarial'

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
filter_size = [(7,7), (5,5), (2,2), (5,5)]
num_filter = [16, 32, 48, 64]
num_channels = 3
pooling_size = [(3,3), (2,2), (2,2)]
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
conv_layers1.append(conv_activation[0])
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

i = i + 1 #Sequence with no MaxPooling
conv_layers1.append(
    Convolutional(
        filter_size=filter_size[j+2],
        num_filters=num_filter[j+2],
        step=conv_step,
        border_mode=border_mode,
        name='conv_{}'.format(i)))
conv_layers1.append(BatchNormalization(name='BNconv_{}'.format(i)))
conv_layers1.append(conv_activation[0])

i = i + 1 #Sequence
conv_layers1.append(
    Convolutional(
        filter_size=filter_size[j+3],
        num_filters=num_filter[j+3],
        step=conv_step,
        border_mode=border_mode,
        name='conv_{}'.format(i)))
conv_layers1.append(BatchNormalization(name='BNconv_{}'.format(i)))
conv_layers1.append(conv_activation[0])
conv_layers1.append(MaxPooling(pooling_size[j+2], name='pool_{}'.format(i)))

conv_sequence1 = ConvolutionalSequence(conv_layers1, num_channels=num_channels, image_size=image_size,
weights_init=Uniform(width=0.2), biases_init=Constant(0.), name='ConvSeq1_{}'.format(i))
conv_sequence1.initialize()
out1 = conv_sequence1.apply(x)


################# Convolutional Sequence 2 #################
# conv_layers2 parameters
i = i+1 #Sequence
j = 0 #Sub Layer
filter_size = [(7,7), (5,5), (2,2), (5,5)]
num_filter = [16, 32, 48, 64]
num_channels = 3
pooling_size = [(3,3), (2,2), (2,2)]
conv_step = (1,1)
border_mode = 'valid'

conv_layers2 = []
conv_layers2.append(SpatialBatchNormalization(name='spatialBN_{}'.format(i)))
conv_layers2.append(
    Convolutional(
        filter_size=filter_size[j],
        num_filters=num_filter[j],
        step=conv_step,
        border_mode=border_mode,
        name='conv_{}'.format(i)))
conv_layers2.append(BatchNormalization(name='BNconv_{}'.format(i)))
conv_layers2.append(conv_activation[0])
conv_layers2.append(MaxPooling(pooling_size[j], name='pool_{}'.format(i)))

i = i + 1 #Sequence
conv_layers2.append(
    Convolutional(
        filter_size=filter_size[j+1],
        num_filters=num_filter[j+1],
        step=conv_step,
        border_mode=border_mode,
        name='conv_{}'.format(i)))
conv_layers2.append(BatchNormalization(name='BNconv_{}'.format(i)))
conv_layers2.append(conv_activation[0])
conv_layers2.append(MaxPooling(pooling_size[j+1], name='pool_{}'.format(i)))

i = i + 1 #Sequence with no MaxPooling
conv_layers2.append(
    Convolutional(
        filter_size=filter_size[j+2],
        num_filters=num_filter[j+2],
        step=conv_step,
        border_mode=border_mode,
        name='conv_{}'.format(i)))
conv_layers2.append(BatchNormalization(name='BNconv_{}'.format(i)))
conv_layers2.append(conv_activation[0])

i = i + 1 #Sequence
conv_layers2.append(
    Convolutional(
        filter_size=filter_size[j+3],
        num_filters=num_filter[j+3],
        step=conv_step,
        border_mode=border_mode,
        name='conv_{}'.format(i)))
conv_layers2.append(BatchNormalization(name='BNconv_{}'.format(i)))
conv_layers2.append(conv_activation[0])
conv_layers2.append(MaxPooling(pooling_size[j+2], name='pool_{}'.format(i)))

conv_sequence2 = ConvolutionalSequence(conv_layers2, num_channels=num_channels, image_size=image_size,
weights_init=Uniform(width=0.2), biases_init=Constant(0.), name='ConvSeq2_{}'.format(i))
conv_sequence2.initialize()
out2 = conv_sequence2.apply(x)


# Theano variables
x = tensor.tensor4('image_features')
y = tensor.lmatrix('targets')


### setting up "clean" and "dirty" images with data_preprocessing
if mode == ("GPU_run"):
    try:
        x1 = data_preprocessing1(x).copy(name='x_clean'))
        x2 = data_preprocessing2(x).copy(name='x_dirty'))
        out1 = conv_sequence2.apply(x1)
        out2 = conv_sequence2.apply(x2)


### Flattening data
conv_out1 = Flattener(name='flattener1').apply(out1)
conv_out2 = Flattener(name='flattener2').apply(out2)
conv_out = tensor.concatenate([conv_out1,conv_out2],axis=1)

### MLP
mlp_hiddens = 1000
top_mlp_dims = [numpy.prod(conv_sequence1.get_dim('output'))+numpy.prod(conv_sequence1.get_dim('output'))] + [mlp_hiddens] + [output_size]
top_mlp = MLP(mlp_activation, top_mlp_dims,weights_init=Uniform(width=0.2),biases_init=Constant(0.))
top_mlp.initialize()



### Getting the data

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

    # Data Transformation
    stream = ScaleAndShift(stream, 1./255, 0, which_sources=('image_features',))
    stream = Cast(stream, dtype='float32', which_sources=('image_features',))
    return stream


if mode is "CPU_test":
    data_train_stream = create_data(DogsVsCats(('train',), subset=slice(0, 100)))
    data_valid_stream = create_data(DogsVsCats(('train',), subset=slice(100, 110)))
if mode is  "GPU_run":
    data_train_stream = create_data(DogsVsCats(('train',), subset=slice(0, 22500)))
    data_valid_stream = create_data(DogsVsCats(('train',), subset=slice(22500, 25000)))
if mode is "data_server":
    data_train_stream = ServerDataStream(('image_features','targets'), False, port=5560)
    data_valid_stream = ServerDataStream(('image_features','targets'), False, port=5561)


### Setting up the model
probs = top_mlp.apply(conv_out)

cost = CategoricalCrossEntropy().apply(y.flatten(), probs).copy(name='cost')
error = MisclassificationRate().apply(y.flatten(), probs)
error_rate = error.copy(name='error_rate')
error_rate2 = error.copy(name='error_rate2')
cg = ComputationGraph([cost, error_rate])

### Gradient Descent
algorithm = GradientDescent(cost=cost, parameters=cg.parameters, step_rule=Scale(learning_rate=learning_rate))

extensions = [Timing(),
              FinishAfter(after_n_epochs=num_epochs),
              DataStreamMonitoring(
                  [cost, error_rate, error_rate2],
                  data_valid_stream,
                  prefix="valid"),
              TrainingDataMonitoring(
                  [cost, error_rate,
                   aggregation.mean(algorithm.total_gradient_norm)],
                  prefix="train",
                  after_epoch=True),
             Checkpoint(save_to),
              ProgressBar(),
              Printing()]

### Plotting extensions
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

model = Model(cost)
main_loop = MainLoop(algorithm=algorithm,data_stream=data_train_stream,model=model,extensions=extensions)

main_loop.run()
