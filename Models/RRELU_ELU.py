"""Randomized Leaky Relu (RReLU)"""
import logging

from theano import tensor
import numbers
import numpy

from blocks.bricks.base import application, Brick, lazy
from blocks.bricks.interfaces import Activation, Feedforward, Initializable
from blocks.bricks.interfaces import LinearLike, Random  # noqa

from blocks.bricks.wrappers import WithExtraDims
from blocks.roles import add_role, WEIGHT, BIAS
from blocks.utils import shared_floatx_nans

logger = logging.getLogger(__name__)



class Rand_Leaky_Rectifier(Activation):
    r""" Random Leaky Rectifier brick.

    The softplus is defined as :math:
    `y=\left\{\begin{matrix}x_{i,j}\ if\ x_{i,j}\geq 0
    \\
    a_{i,j}x_{i,j} \if\ x_{i,j}\leq 0
    \end{matrix}\right.`.

    .. Xu, B., Wang, N., Tianqui, C., Mu, L. (May 2015). Empirical Evaluation of Rectified Activations in Convolution
    Network. ICML Deep Learning Workshop 2015.

    Parameters
    ----------
    rng : :class:`numpy.random.RandomState`
    """
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_, lower = 3, upper = 8):
        #    Suggested by the NDSB competition winner, 'a' is sampled from U(3, 8).
        a = numpy.random.uniform(lower, upper, size=None) # broadcastable
        return tensor.switch(input_ > 0, input_, input_/a)

# Code from Julien St-Pierre-Fortin
class ELU(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
    	self.alpha = 1.0
        return tensor.switch(input_ >= 0, input_, self.alpha*(tensor.exp(input_) - 1))
