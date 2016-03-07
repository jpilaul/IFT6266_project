from theano.tensor.nnet.conv import conv2d, ConvOp
from theano.tensor.signal.downsample import max_pool_2d, DownsampleFactorMax

from blocks.bricks import Initializable, Feedforward, Sequence
from blocks.bricks.base import application, Brick, lazy
from blocks.roles import add_role, FILTER, BIAS
from blocks.utils import shared_floatx_nans
from blocks.bricks.conv import (Convolutional, MaxPooling)

class ConvolutionalActivation(Sequence, Initializable):
    """A convolution followed by an activation function.
    Parameters
    ----------
    activation : :class:`.BoundApplication`
        The application method to apply after convolution (i.e.
        the nonlinear activation function)
    See Also
    --------
    :class:`Convolutional` for the other parameters.
    """

    def __init__(self, activation, filter_size, num_filters, num_channels,
                 batch_size=None, image_size=None, step=(1, 1),
                 border_mode='valid', **kwargs):
        self.convolution = Convolutional()

        self.filter_size = filter_size
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.image_size = image_size
        self.step = step
        self.border_mode = border_mode

        super(ConvolutionalActivation, self).__init__(
            application_methods=[self.convolution.apply, activation],
            **kwargs)

    def _push_allocation_config(self):
        for attr in ['filter_size', 'num_filters', 'step', 'border_mode',
                     'batch_size', 'num_channels', 'image_size']:
            setattr(self.convolution, attr, getattr(self, attr))

    def get_dim(self, name):
        # TODO The name of the activation output doesn't need to be `output`
        return self.convolution.get_dim(name)



class ConvolutionalLayer(Sequence, Initializable):
    """A complete convolutional layer: Convolution, nonlinearity, pooling.
    .. todo::
       Mean pooling.
    Parameters
    ----------
    activation : :class:`.BoundApplication`
        The application method to apply in the detector stage (i.e. the
        nonlinearity before pooling. Needed for ``__init__``.
    See Also
    --------
    :class:`Convolutional` and :class:`MaxPooling` for the other
    parameters.
    Notes
    -----
    Uses max pooling.
    """

    def __init__(self, activation, filter_size, num_filters, pooling_size,
                 num_channels, conv_step=(1, 1), pooling_step=None,
                 batch_size=None, image_size=None, border_mode='valid',
                 **kwargs):
        self.convolution = ConvolutionalActivation(activation, filter_size, num_filters, num_channels)
        self.pooling = MaxPooling()
        super(ConvolutionalLayer, self).__init__(
            application_methods=[self.convolution.apply,
                                 self.pooling.apply], **kwargs)
        self.convolution.name = self.name + '_convolution'
        self.pooling.name = self.name + '_pooling'

        self.filter_size = filter_size
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.pooling_size = pooling_size
        self.conv_step = conv_step
        self.pooling_step = pooling_step
        self.batch_size = batch_size
        self.border_mode = border_mode
        self.image_size = image_size

    def _push_allocation_config(self):
        for attr in ['filter_size', 'num_filters', 'num_channels',
                     'batch_size', 'border_mode', 'image_size']:
            setattr(self.convolution, attr, getattr(self, attr))
        self.convolution.step = self.conv_step
        self.convolution._push_allocation_config()
        if self.image_size is not None:
            pooling_input_dim = self.convolution.get_dim('output')
        else:
            pooling_input_dim = None
        self.pooling.input_dim = pooling_input_dim
        self.pooling.pooling_size = self.pooling_size
        self.pooling.step = self.pooling_step
        self.pooling.batch_size = self.batch_size

    def get_dim(self, name):
        if name == 'input_':
            return self.convolution.get_dim('input_')
        if name == 'output':
            return self.pooling.get_dim('output')
        return super(ConvolutionalLayer, self).get_dim(name)

    @property
    def num_output_channels(self):
        return self.num_filters
