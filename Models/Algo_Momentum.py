__author__ = 'Jonathan Pilault'
"""Training algorithms."""
import logging
import itertools
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from six.moves import reduce

from picklable_itertools.extras import equizip

import theano
from six import add_metaclass
from theano import tensor

from blocks.graph import ComputationGraph
from blocks.roles import add_role, ALGORITHM_HYPERPARAMETER, ALGORITHM_BUFFER
from blocks.theano_expressions import l2_norm


from blocks.algorithms import CompositeRule, Scale, StepRule
from blocks.utils import (dict_subset, pack, shared_floatx,
                          shared_floatx_zeros_matching)

logger = logging.getLogger(__name__)


def _create_algorithm_buffer_for(param, *args, **kwargs):
    buf = shared_floatx_zeros_matching(param, *args, **kwargs)
    buf.tag.for_parameter = param
    add_role(buf, ALGORITHM_BUFFER)
    return buf



class BasicNesterovAdam(StepRule):
    """ Idea came from the paper below
    .. Timothy Dozat May 2016
       *INCORPORATING NESTEROV MOMENTUM INTO ADAM*,
       ICLR 2016
       http://beta.openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ

    Parameters
    ----------
    learning_rate : float, optional
        Step size.
        Default value is set to 0.002.
    mu1 : float, optional and same as momentum coefficient
        The momentum coefficient. Defaults to 0.99
    epsilon : float, optional
        Default value is set to 1e-8.
    decay_factor : float, optional
        Default value is set to 1 - 1e-8.

    """
    def __init__(self, learning_rate=0.002,
                 mu1=0.99, nu2=0.999, epsilon=1e-8,
                 decay_prod=(1.)):
        self.learning_rate = shared_floatx(learning_rate, "learning_rate")
        self.mu1 = shared_floatx(mu1, "mu1")
        self.nu2 = shared_floatx(nu2, "nu2")
        self.epsilon = shared_floatx(epsilon, "epsilon")
        self.decay_prod = shared_floatx(decay_prod, "decay_prod")
        for param in [self.learning_rate, self.mu1, self.nu2, self.epsilon,
                      self.decay_prod]:
            add_role(param, ALGORITHM_HYPERPARAMETER)

    def compute_step(self, parameter, previous_step):
        BNA_velocity = _create_algorithm_buffer_for(parameter, 'BNA_velocity')
        norm = _create_algorithm_buffer_for(parameter, 'norm')
        time = shared_floatx(0., 'time')
        add_role(time, ALGORITHM_BUFFER)


        ############## Set parameter ##############
        # Eta
        eta = self.learning_rate

        # Momentum decay schedule u_t as defined in the paper (based on u)
        u_t = self.mu1*(1-0.5*0.96**time)
        u_t_plus_1 = self.mu1*(1-0.5*0.96**(time+1))

        # Decay factorizes over time product(u_i)
        self.decay_prod = self.decay_prod*(u_t)

        ############## Nesterov-accelerated adaptive moment algorithm ##############

        # Step 1 get gradient: g_t <- delta(f(theta_(t-1)))
        g_t = previous_step

        # Step 2 decay gradient: g_hat <- g_t/(1-product(u_i)
        g_hat = g_t/(1-self.decay_prod)

        # Step 3 get momentum vector: m_t <- u*m_(t-1) + (1-u)*g_t
        m_t = self.mu1*BNA_velocity + (1-self.mu1)*g_t

        # Step 4 decay momentum: m_hat <- m_t/(1-product(u_i)
        m_hat = m_t/(1-self.decay_prod)

        # Step 5 get the norm vector: n_t <-nu*n_(t-1) + (1+nu)*(g_t)**2
        n_t = self.nu2*norm + (1-self.nu2)*g_t**2

        # Step 6 decay the norm vector: n_hat <- n_t/(1-nu**t)
        n_hat = n_t/(1-self.nu2**time)

        # Step 7 recompute momentum using m_hat and g_hat: m_last <- (1-u_t)g_hat + u_(t+1)m_hat
        m_last = (1-u_t)*g_hat + u_t_plus_1*m_hat

        # Step 8 compute step for update: rho*(m_last)/((n_hat)**(1/2)+epsilon)

        step = (eta * m_last / (tensor.sqrt(n_hat) + self.epsilon))

        updates = [(BNA_velocity, m_t),
                   (norm, n_t),
                   (time, time+1)]

        return step, updates


class NesterovAdam(CompositeRule):
    """Accumulates step with exponential discount.
    Combines :class:`BasicNesterovMomentum` and :class:`Scale` to form the
    Nesterov-Adam momentum step rule.
    Parameters
    ----------
    learning_rate : float, optional
        The learning rate by which the previous step scaled. Defaults to 1.
    momentum : float, optional
        The momentum coefficient. Defaults to 0.99
    Attributes
    ----------
    learning_rate : :class:`~tensor.SharedVariable`
        A variable for learning rate.
    momentum : :class:`~tensor.SharedVariable`
        A variable for momentum.
    See Also
    --------
    :class:`SharedVariableModifier`
    """
    def __init__(self, learning_rate=1.0, momentum=0.99):
        scale = Scale(learning_rate=learning_rate)
        basic_nesterov_adam = BasicNesterovAdam(mu1=momentum)
        self.learning_rate = scale.learning_rate
        self.momentum = basic_nesterov_adam.mu1
        self.components = [scale, basic_nesterov_adam]


#### Below code not from me
#### Code from blocks-extras https://github.com/mila-udem/blocks-extras/blob/master/blocks_extras/algorithms/__init__.py

class BasicNesterovMomentum(StepRule):
    u"""Accumulates step with exponential discount.

    Parameters
    ----------
    momentum : float, optional
        The momentum coefficient. Defaults to 0.

    Notes
    -----
    This step rule is intended to be used in conjunction with another
    step rule, _e.g._ :class:`Scale`. For an all-batteries-included
    experience, look at :class:`NesterovMomentum`.

    The Nesterov momentum comes from [N83] and it is implemented as in
    [BBR13].

    .. [N83] Yurii Nesterov, *A method of solving a convex programming
        problem with convergence rate O(1/sqr(k))*, Soviet Mathematics
        Doklady (1983), Vol. 27, No. 2, pp. 372-376.

    .. [BBR13] Yoshua Bengio, Nicolas Boulanger-Lewandowski, and Razvan
        Pascanu, *Advances in optimizing recurrent networks*,
        ICASSP (2013), pp 8624-8628.

    """
    def __init__(self, momentum=0.):
        self.momentum = shared_floatx(momentum)

    def compute_step(self, parameter, previous_step):
        velocity = shared_floatx(parameter.get_value() * 0.)
        velocity_update = self.momentum*velocity + previous_step
        step = (self.momentum**2 * velocity + previous_step *
                (1 + self.momentum))
        updates = [(velocity, velocity_update)]
        return step, updates


class NesterovMomentum(CompositeRule):
    """Accumulates step with exponential discount.

    Combines :class:`BasicNesterovMomentum` and :class:`Scale` to form the
    usual Nesterov momentum step rule.

    Parameters
    ----------
    learning_rate : float, optional
        The learning rate by which the previous step scaled. Defaults to 1.
    momentum : float, optional
        The momentum coefficient. Defaults to 0.

    Attributes
    ----------
    learning_rate : :class:`~tensor.SharedVariable`
        A variable for learning rate.
    momentum : :class:`~tensor.SharedVariable`
        A variable for momentum.

    See Also
    --------
    :class:`SharedVariableModifier`

    """
    def __init__(self, learning_rate=1.0, momentum=0.):
        scale = Scale(learning_rate=learning_rate)
        basic_nesterov_momentum = BasicNesterovMomentum(momentum=momentum)
        self.learning_rate = scale.learning_rate
        self.momentum = basic_nesterov_momentum.momentum
        self.components = [scale, basic_nesterov_momentum]
