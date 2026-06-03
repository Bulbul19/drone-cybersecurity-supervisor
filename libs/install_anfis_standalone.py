#!/usr/bin/env python3
"""
Standalone installation script for PyTorch-compatible ANFIS library
This script creates the package files directly without needing GitHub
Run this in your virtual environment: python install_anfis_standalone.py
"""

import os
import sys
import shutil
from pathlib import Path

# ANFIS source code
ANFIS_SOURCE = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    ANFIS in torch: the ANFIS layers
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
    Acknowledgement: twmeggs' implementation of ANFIS in Python was very
    useful in understanding how the ANFIS structures could be interpreted:
        https://github.com/twmeggs/anfis
"""

import itertools
from collections import OrderedDict

import numpy as np

import torch
import torch.nn.functional as F


dtype = torch.float


class FuzzifyVariable(torch.nn.Module):
    """
        Represents a single fuzzy variable, holds a list of its MFs.
        Forward pass will then fuzzify the input (value for each MF).
    """
    def __init__(self, mfdefs):
        super(FuzzifyVariable, self).__init__()
        if isinstance(mfdefs, list):  # No MF names supplied
            mfnames = ['mf{}'.format(i) for i in range(len(mfdefs))]
            mfdefs = OrderedDict(zip(mfnames, mfdefs))
        self.mfdefs = torch.nn.ModuleDict(mfdefs)
        self.padding = 0

    @property
    def num_mfs(self):
        """Return the actual number of MFs (ignoring any padding)"""
        return len(self.mfdefs)

    def members(self):
        """
            Return an iterator over this variables's membership functions.
            Yields tuples of the form (mf-name, MembFunc-object)
        """
        return self.mfdefs.items()

    def pad_to(self, new_size):
        """
            Will pad result of forward-pass (with zeros) so it has new_size,
            i.e. as if it had new_size MFs.
        """
        self.padding = new_size - len(self.mfdefs)

    def fuzzify(self, x):
        """
            Yield a list of (mf-name, fuzzy values) for these input values.
        """
        for mfname, mfdef in self.mfdefs.items():
            yvals = mfdef(x)
            yield(mfname, yvals)

    def forward(self, x):
        """
            Return a tensor giving the membership value for each MF.
            x.shape: n_cases
            y.shape: n_cases * n_mfs
        """
        y_pred = torch.cat([mf(x) for mf in self.mfdefs.values()], dim=1)
        if self.padding > 0:
            y_pred = torch.cat([y_pred,
                                torch.zeros(x.shape[0], self.padding)], dim=1)
        return y_pred


class FuzzifyLayer(torch.nn.Module):
    """
        A list of fuzzy variables, representing the inputs to the FIS.
        Forward pass will fuzzify each variable individually.
        We pad the variables so they all seem to have the same number of MFs,
        as this allows us to put all results in the same tensor.
    """
    def __init__(self, varmfs, varnames=None):
        super(FuzzifyLayer, self).__init__()
        if not varnames:
            self.varnames = ['x{}'.format(i) for i in range(len(varmfs))]
        else:
            self.varnames = list(varnames)
        maxmfs = max([var.num_mfs for var in varmfs])
        for var in varmfs:
            var.pad_to(maxmfs)
        self.varmfs = torch.nn.ModuleDict(zip(self.varnames, varmfs))

    @property
    def num_in(self):
        """Return the number of input variables"""
        return len(self.varmfs)

    @property
    def max_mfs(self):
        """ Return the max number of MFs in any variable"""
        return max([var.num_mfs for var in self.varmfs.values()])

    def __repr__(self):
        """
            Print the variables, MFS and their parameters (for info only)
        """
        r = ['Input variables']
        for varname, members in self.varmfs.items():
            r.append('Variable {}'.format(varname))
            for mfname, mfdef in members.mfdefs.items():
                r.append('- {}: {}({})'.format(mfname,
                         mfdef.__class__.__name__,
                         ', '.join(['{}={}'.format(n, p.item())
                                   for n, p in mfdef.named_parameters()])))
        return '\\n'.join(r)

    def forward(self, x):
        """ Fuzzyify each variable's value using each of its corresponding mfs.
            x.shape = n_cases * n_in
            y.shape = n_cases * n_in * n_mfs
        """
        assert x.shape[1] == self.num_in,\\
            '{} is wrong no. of input values'.format(self.num_in)
        y_pred = torch.stack([var(x[:, i:i+1])
                              for i, var in enumerate(self.varmfs.values())],
                             dim=1)
        return y_pred


class AntecedentLayer(torch.nn.Module):
    """
        Form the 'rules' by taking all possible combinations of the MFs
        for each variable. Forward pass then calculates the fire-strengths.
    """
    def __init__(self, varlist):
        super(AntecedentLayer, self).__init__()
        # Count the (actual) mfs for each variable:
        mf_count = [var.num_mfs for var in varlist]
        # Now make the MF indices for each rule:
        mf_indices = itertools.product(*[range(n) for n in mf_count])
        self.mf_indices = torch.tensor(list(mf_indices))
        # mf_indices.shape is n_rules * n_in

    def num_rules(self):
        return len(self.mf_indices)

    def extra_repr(self, varlist=None):
        if not varlist:
            return None
        row_ants = []
        mf_count = [len(fv.mfdefs) for fv in varlist.values()]
        for rule_idx in itertools.product(*[range(n) for n in mf_count]):
            thisrule = []
            for (varname, fv), i in zip(varlist.items(), rule_idx):
                thisrule.append('{} is {}'
                                .format(varname, list(fv.mfdefs.keys())[i]))
            row_ants.append(' and '.join(thisrule))
        return '\\n'.join(row_ants)

    def forward(self, x):
        """ Calculate the fire-strength for (the antecedent of) each rule
            x.shape = n_cases * n_in * n_mfs
            y.shape = n_cases * n_rules
        """
        # Expand (repeat) the rule indices to equal the batch size:
        batch_indices = self.mf_indices.expand((x.shape[0], -1, -1))
        # Then use these indices to populate the rule-antecedents
        ants = torch.gather(x.transpose(1, 2), 1, batch_indices)
        # ants.shape is n_cases * n_rules * n_in
        # Last, take the AND (= product) for each rule-antecedent
        rules = torch.prod(ants, dim=2)
        return rules


class ConsequentLayer(torch.nn.Module):
    """
        A simple linear layer to represent the TSK consequents.
        Hybrid learning, so use MSE (not BP) to adjust coefficients.
        Hence, coeffs are no longer parameters for backprop.
    """
    def __init__(self, d_in, d_rule, d_out):
        super(ConsequentLayer, self).__init__()
        c_shape = torch.Size([d_rule, d_out, d_in+1])
        self._coeff = torch.zeros(c_shape, dtype=dtype, requires_grad=True)

    @property
    def coeff(self):
        """
            Record the (current) coefficients for all the rules
            coeff.shape: n_rules * n_out * (n_in+1)
        """
        return self._coeff

    @coeff.setter
    def coeff(self, new_coeff):
        """
            Record new coefficients for all the rules
            coeff: for each rule, for each output variable:
                   a coefficient for each input variable, plus a constant
        """
        assert new_coeff.shape == self.coeff.shape, \\
            'Coeff shape should be {}, but is actually {}'\\
            .format(self.coeff.shape, new_coeff.shape)
        self._coeff = new_coeff

    def fit_coeff(self, x, weights, y_actual):
        """
            Use LSE to solve for coeff: y_actual = coeff * (weighted)x
                  x.shape: n_cases * n_in
            weights.shape: n_cases * n_rules
            [ coeff.shape: n_rules * n_out * (n_in+1) ]
                  y.shape: n_cases * n_out
        """
        # Append 1 to each list of input vals, for the constant term:
        x_plus = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        # Shape of weighted_x is n_cases * n_rules * (n_in+1)
        weighted_x = torch.einsum('bp, bq -> bpq', weights, x_plus)
        # Can't have value 0 for weights, or LSE won't work:
        weighted_x[weighted_x == 0] = 1e-12
        # Squash x and y down to 2D matrices for lstsq:
        weighted_x_2d = weighted_x.view(weighted_x.shape[0], -1)
        y_actual_2d = y_actual.view(y_actual.shape[0], -1)
        # Use lstsq to do LSE, then pick out the solution rows:
        try:
            coeff_2d, _ = torch.lstsq(y_actual_2d, weighted_x_2d)
        except RuntimeError as e:
            print('Internal error in lstsq', e)
            print('Weights are:', weighted_x)
            raise e
        coeff_2d = coeff_2d[0:weighted_x_2d.shape[1]]
        # Reshape to 3D tensor: divide by rules, n_in+1, then swap last 2 dims
        self.coeff = coeff_2d.view(weights.shape[1], x.shape[1]+1, -1)\\
            .transpose(1, 2)
        # coeff dim is thus: n_rules * n_out * (n_in+1)

    def forward(self, x):
        """
            Calculate: y = coeff * x + const   [NB: no weights yet]
                  x.shape: n_cases * n_in
              coeff.shape: n_rules * n_out * (n_in+1)
                  y.shape: n_cases * n_out * n_rules
        """
        # Append 1 to each list of input vals, for the constant term:
        x_plus = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        # Need to switch dimansion for the multipy, then switch back:
        y_pred = torch.matmul(self.coeff, x_plus.t())
        return y_pred.transpose(0, 2)  # swaps cases and rules


class PlainConsequentLayer(ConsequentLayer):
    """
        A linear layer to represent the TSK consequents.
        Not hybrid learning, so coefficients are backprop-learnable parameters.
    """
    def __init__(self, *params):
        super(PlainConsequentLayer, self).__init__(*params)
        self.register_parameter('coefficients',
                                torch.nn.Parameter(self._coeff))

    @property
    def coeff(self):
        """
            Record the (current) coefficients for all the rules
            coeff.shape: n_rules * n_out * (n_in+1)
        """
        return self.coefficients

    def fit_coeff(self, x, weights, y_actual):
        """Not hybrid learning: using BP to learn coefficients"""
        assert False,\\
            'Not hybrid learning: I\\'m using BP to learn coefficients'


class WeightedSumLayer(torch.nn.Module):
    """
        Sum the TSK for each outvar over rules, weighted by fire strengths.
        This could/should be layer 5 of the Anfis net.
        I don't actually use this class, since it's just one line of code.
    """
    def __init__(self):
        super(WeightedSumLayer, self).__init__()

    def forward(self, weights, tsk):
        """
            weights.shape: n_cases * n_rules
                tsk.shape: n_cases * n_out * n_rules
             y_pred.shape: n_cases * n_out
        """
        # Add a dimension to weights to get the bmm to work:
        y_pred = torch.bmm(tsk, weights.unsqueeze(2))
        return y_pred.squeeze(2)


class AnfisNet(torch.nn.Module):
    """
        This is a container for the 5 layers of the ANFIS net.
        The forward pass maps inputs to outputs based on current settings,
        and then fit_coeff will adjust the TSK coeff using LSE.
    """
    def __init__(self, description, invardefs, outvarnames, hybrid=True):
        super(AnfisNet, self).__init__()
        self.description = description
        self.outvarnames = outvarnames
        self.hybrid = hybrid
        varnames = [v for v, _ in invardefs]
        mfdefs = [FuzzifyVariable(mfs) for _, mfs in invardefs]
        self.num_in = len(invardefs)
        self.num_rules = np.prod([len(mfs) for _, mfs in invardefs])
        if self.hybrid:
            cl = ConsequentLayer(self.num_in, self.num_rules, self.num_out)
        else:
            cl = PlainConsequentLayer(self.num_in, self.num_rules, self.num_out)
        self.layer = torch.nn.ModuleDict(OrderedDict([
            ('fuzzify', FuzzifyLayer(mfdefs, varnames)),
            ('rules', AntecedentLayer(mfdefs)),
            # normalisation layer is just implemented as a function.
            ('consequent', cl),
            # weighted-sum layer is just implemented as a function.
            ]))

    @property
    def num_out(self):
        return len(self.outvarnames)

    @property
    def coeff(self):
        return self.layer['consequent'].coeff

    @coeff.setter
    def coeff(self, new_coeff):
        self.layer['consequent'].coeff = new_coeff

    def fit_coeff(self, x, y_actual):
        """
            Do a forward pass (to get weights), then fit to y_actual.
            Does nothing for a non-hybrid ANFIS, so we have same interface.
        """
        if self.hybrid:
            self(x)
            self.layer['consequent'].fit_coeff(x, self.weights, y_actual)

    def input_variables(self):
        """
            Return an iterator over this system's input variables.
            Yields tuples of the form (var-name, FuzzifyVariable-object)
        """
        return self.layer['fuzzify'].varmfs.items()

    def output_variables(self):
        """
            Return an list of the names of the system's output variables.
        """
        return self.outvarnames

    def extra_repr(self):
        rstr = []
        vardefs = self.layer['fuzzify'].varmfs
        rule_ants = self.layer['rules'].extra_repr(vardefs).split('\\n')
        for i, crow in enumerate(self.layer['consequent'].coeff):
            rstr.append('Rule {:2d}: IF {}'.format(i, rule_ants[i]))
            rstr.append(' '*9+'THEN {}'.format(crow.tolist()))
        return '\\n'.join(rstr)

    def forward(self, x):
        """
            Forward pass: run x thru the five layers and return the y values.
            I save the outputs from each layer to an instance variable,
            as this might be useful for comprehension/debugging.
        """
        self.fuzzified = self.layer['fuzzify'](x)
        self.raw_weights = self.layer['rules'](self.fuzzified)
        self.weights = F.normalize(self.raw_weights, p=1, dim=1)
        self.rule_tsk = self.layer['consequent'](x)
        # y_pred = self.layer['weighted_sum'](self.weights, self.rule_tsk)
        y_pred = torch.bmm(self.rule_tsk, self.weights.unsqueeze(2))
        self.y_pred = y_pred.squeeze(2)
        return self.y_pred


# These hooks are handy for debugging:

def module_hook(label):
    """ Use this module hook like this:
        m = AnfisNet()
        m.layer.fuzzify.register_backward_hook(module_hook('fuzzify'))
        m.layer.consequent.register_backward_hook(modul_hook('consequent'))
    """
    return (lambda module, grad_input, grad_output:
            print('BP for module', label,
                  'with out grad:', grad_output,
                  'and in grad:', grad_input))


def tensor_hook(label):
    """
        If you want something more fine-graned, attach this to a tensor.
    """
    return (lambda grad:
            print('BP for', label, 'with grad:', grad))
'''

# Membership functions source code
MEMBERSHIP_SOURCE = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    ANFIS in torch: some fuzzy membership functions.
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
"""

import torch


def _mk_param(val):
    """Make a torch parameter from a scalar value"""
    if isinstance(val, torch.Tensor):
        val = val.item()
    return torch.nn.Parameter(torch.tensor(val, dtype=torch.float))


class GaussMembFunc(torch.nn.Module):
    """
        Gaussian membership functions, defined by two parameters:
            mu, the mean (center)
            sigma, the standard deviation.
    """
    def __init__(self, mu, sigma):
        super(GaussMembFunc, self).__init__()
        self.register_parameter('mu', _mk_param(mu))
        self.register_parameter('sigma', _mk_param(sigma))

    def forward(self, x):
        val = torch.exp(-torch.pow(x - self.mu, 2) / (2 * self.sigma**2))
        return val.unsqueeze(1)

    def pretty(self):
        return 'GaussMembFunc {} {}'.format(self.mu, self.sigma)


def make_gauss_mfs(sigma, mu_list):
    """Return a list of gaussian mfs, same sigma, list of means"""
    return [GaussMembFunc(mu, sigma) for mu in mu_list]


class BellMembFunc(torch.nn.Module):
    """
        Generalised Bell membership function; defined by three parameters:
            a, the half-width (at the crossover point)
            b, controls the slope at the crossover point (which is -b/2a)
            c, the center point
    """
    def __init__(self, a, b, c):
        super(BellMembFunc, self).__init__()
        self.register_parameter('a', _mk_param(a))
        self.register_parameter('b', _mk_param(b))
        self.register_parameter('c', _mk_param(c))
        self.b.register_hook(BellMembFunc.b_log_hook)

    @staticmethod
    def b_log_hook(grad):
        """
            Possibility of a log(0) in the grad for b, giving a nan.
            Fix this by replacing any nan in the grad with ~0.
        """
        grad[torch.isnan(grad)] = 1e-9
        return grad

    def forward(self, x):
        dist = torch.pow((x - self.c)/self.a, 2)
        return torch.reciprocal(1 + torch.pow(dist, self.b)).unsqueeze(1)

    def pretty(self):
        return 'BellMembFunc {} {} {}'.format(self.a, self.b, self.c)


def make_bell_mfs(a, b, clist):
    """Return a list of bell mfs, same (a,b), list of centers"""
    return [BellMembFunc(a, b, c) for c in clist]


class TriangularMembFunc(torch.nn.Module):
    """
        Triangular membership function; defined by three parameters:
            a, left foot, mu(x) = 0
            b, midpoint, mu(x) = 1
            c, right foot, mu(x) = 0
    """
    def __init__(self, a, b, c):
        super(TriangularMembFunc, self).__init__()
        assert a <= b and b <= c,\\
            'Triangular parameters: must have a <= b <= c.'
        self.register_parameter('a', _mk_param(a))
        self.register_parameter('b', _mk_param(b))
        self.register_parameter('c', _mk_param(c))

    @staticmethod
    def isosceles(width, center):
        """
            Construct a triangle MF with given width-of-base and center
        """
        return TriangularMembFunc(center-width, center, center+width)

    def forward(self, x):
        result = torch.where(
            (self.a < x) & (x <= self.b),
            (x - self.a) / (self.b - self.a),
            # else
            torch.where(
                (self.b < x) & (x <= self.c),
                (self.c - x) / (self.c - self.b),
                torch.zeros_like(x)))
        return result.unsqueeze(1)

    def pretty(self):
        return 'TriangularMembFunc {} {} {}'.format(self.a, self.b, self.c)


def make_tri_mfs(width, clist):
    """Return a list of triangular mfs, same width, list of centers"""
    return [TriangularMembFunc(c-width/2, c, c+width/2) for c in clist]


class TrapezoidalMembFunc(torch.nn.Module):
    """
        Trapezoidal membership function; defined by four parameters.
        Membership is defined as:
            to the left of a: always 0
            from a to b: slopes from 0 up to 1
            from b to c: always 1
            from c to d: slopes from 1 down to 0
            to the right of d: always 0
    """
    def __init__(self, a, b, c, d):
        super(TrapezoidalMembFunc, self).__init__()
        assert a <= b and b <= c and c <= d,\\
            'Trapezoidal parameters: must have a <= b <= c <= d.'
        self.register_parameter('a', _mk_param(a))
        self.register_parameter('b', _mk_param(b))
        self.register_parameter('c', _mk_param(c))
        self.register_parameter('d', _mk_param(d))

    @staticmethod
    def symmetric(topwidth, slope, midpt):
        """
            Make a (symmetric) trapezoid mf, given
                topwidth: length of top (when mu == 1)
                slope: extra length at either side for bottom
                midpt: center point of trapezoid
        """
        b = midpt - topwidth / 2
        c = midpt + topwidth / 2
        return TrapezoidalMembFunc(b - slope, b, c, c + slope)

    @staticmethod
    def rectangle(left, right):
        """
            Make a Trapezoidal MF with vertical sides (so a==b and c==d)
        """
        return TrapezoidalMembFunc(left, left, right, right)

    @staticmethod
    def triangle(left, midpt, right):
        """
            Make a triangle-shaped MF as a special case of a Trapezoidal MF.
            Note: this may revert to general trapezoid under learning.
        """
        return TrapezoidalMembFunc(left, midpt, midpt, right)

    def forward(self, x):
        yvals = torch.zeros_like(x)
        if self.a < self.b:
            incr = (self.a < x) & (x <= self.b)
            yvals[incr] = (x[incr] - self.a) / (self.b - self.a)
        if self.b < self.c:
            decr = (self.b < x) & (x < self.c)
            yvals[decr] = 1
        if self.c < self.d:
            decr = (self.c <= x) & (x < self.d)
            yvals[decr] = (self.d - x[decr]) / (self.d - self.c)
        return yvals.unsqueeze(1)

    def pretty(self):
        return 'TrapezoidalMembFunc a={} b={} c={} d={}'.\\
            format(self.a, self.b, self.c, self.d)


def make_trap_mfs(width, slope, clist):
    """Return a list of symmetric Trap mfs, same (w,s), list of centers"""
    return [TrapezoidalMembFunc.symmetric(width, slope, c) for c in clist]


# Make the classes available via (controlled) reflection:
get_class_for = {n: globals()[n]
                 for n in ['BellMembFunc',
                           'GaussMembFunc',
                           'TriangularMembFunc',
                           'TrapezoidalMembFunc',
                           ]}


def make_anfis(x, num_mfs=5, num_out=1, hybrid=True):
    """
        Make an ANFIS model, auto-calculating the (Gaussian) MFs.
        I need the x-vals to calculate a range and spread for the MFs.
        Variables get named x0, x1, x2,... and y0, y1, y2 etc.
    """
    num_invars = x.shape[1]
    minvals, _ = torch.min(x, dim=0)
    maxvals, _ = torch.max(x, dim=0)
    ranges = maxvals-minvals
    invars = []
    for i in range(num_invars):
        sigma = ranges[i] / num_mfs
        mulist = torch.linspace(minvals[i], maxvals[i], num_mfs).tolist()
        invars.append(('x{}'.format(i), make_gauss_mfs(sigma, mulist)))
    outvars = ['y{}'.format(i) for i in range(num_out)]
    from .anfis import AnfisNet
    model = AnfisNet('Simple classifier', invars, outvars, hybrid=hybrid)
    return model
'''

# Package __init__.py
ANFIS_INIT = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANFIS PyTorch Package
"""

from .anfis import AnfisNet as _AnfisNet

class AnfisNet(_AnfisNet):
    """
    Wrapper for AnfisNet that supports simplified initialization.
    Can be called with just membership_functions for single-output models.
    """
    def __init__(self, membership_functions=None, description='ANFIS Model', 
                 outvarnames=None, hybrid=True, **kwargs):
        # Support backward compatibility: if first arg is membership_functions list
        if membership_functions is not None and isinstance(membership_functions, list):
            if outvarnames is None:
                outvarnames = ['y0']  # Default single output
            super().__init__(description, membership_functions, outvarnames, hybrid)
        else:
            # Original API: description, invardefs, outvarnames, hybrid
            super().__init__(membership_functions, description, outvarnames, hybrid)

__all__ = ['AnfisNet']
'''

# Membership __init__.py
MEMBERSHIP_INIT = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANFIS Membership Functions
"""

from ..membership_funcs import (
    GaussMembFunc,
    BellMembFunc,
    TriangularMembFunc,
    TrapezoidalMembFunc,
    make_gauss_mfs,
    make_bell_mfs,
    make_tri_mfs,
    make_trap_mfs,
)

# Create wrapper functions that accept arrays and return lists
def GaussMf(mu_array, sigma_array):
    """
    Create a list of Gaussian membership functions from arrays.
    mu_array: array of mean values
    sigma_array: array of sigma values
    Returns: list of GaussMembFunc objects
    """
    import numpy as np
    if isinstance(mu_array, np.ndarray):
        mu_array = mu_array.tolist()
    if isinstance(sigma_array, np.ndarray):
        sigma_array = sigma_array.tolist()
    if len(mu_array) != len(sigma_array):
        raise ValueError("mu_array and sigma_array must have the same length")
    return [GaussMembFunc(mu, sigma) for mu, sigma in zip(mu_array, sigma_array)]

def BellMf(a_array, b_array, c_array):
    """
    Create a list of Bell membership functions from arrays.
    Returns: list of BellMembFunc objects
    """
    import numpy as np
    if isinstance(a_array, np.ndarray):
        a_array = a_array.tolist()
    if isinstance(b_array, np.ndarray):
        b_array = b_array.tolist()
    if isinstance(c_array, np.ndarray):
        c_array = c_array.tolist()
    if not (len(a_array) == len(b_array) == len(c_array)):
        raise ValueError("a_array, b_array, and c_array must have the same length")
    return [BellMembFunc(a, b, c) for a, b, c in zip(a_array, b_array, c_array)]

def SigmoidMf(a_array, b_array, c_array):
    """
    Create a list of Sigmoid membership functions from arrays.
    Note: Using Bell function as Sigmoid implementation.
    Returns: list of BellMembFunc objects
    """
    return BellMf(a_array, b_array, c_array)

__all__ = [
    'GaussMembFunc',
    'BellMembFunc',
    'TriangularMembFunc',
    'TrapezoidalMembFunc',
    'GaussMf',
    'BellMf',
    'SigmoidMf',
    'make_gauss_mfs',
    'make_bell_mfs',
    'make_tri_mfs',
    'make_trap_mfs',
]
'''

# Setup.py
SETUP_PY = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for anfis-pytorch package
"""

from setuptools import setup, find_packages

setup(
    name='anfis-pytorch',
    version='0.1.0',
    description='ANFIS implementation in PyTorch',
    author='James Power',
    author_email='james.power@mu.ie',
    url='https://github.com/jfpower/anfis-pytorch',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
    ],
    python_requires='>=3.6',
)
'''

def create_package():
    """Create the anfis package structure directly"""
    
    # Get the project directory
    project_dir = Path(__file__).parent
    package_dir = project_dir / "anfis_package" / "anfis" / "membership"
    package_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating ANFIS package structure...")
    
    # Create anfis.py
    anfis_file = project_dir / "anfis_package" / "anfis" / "anfis.py"
    anfis_file.write_text(ANFIS_SOURCE)
    
    # Create membership_funcs.py
    membership_file = project_dir / "anfis_package" / "anfis" / "membership_funcs.py"
    membership_file.write_text(MEMBERSHIP_SOURCE)
    
    # Create __init__.py for anfis package
    anfis_init = project_dir / "anfis_package" / "anfis" / "__init__.py"
    anfis_init.write_text(ANFIS_INIT)
    
    # Create membership/__init__.py
    membership_init = project_dir / "anfis_package" / "anfis" / "membership" / "__init__.py"
    membership_init.write_text(MEMBERSHIP_INIT)
    
    # Create setup.py
    setup_file = project_dir / "anfis_package" / "setup.py"
    setup_file.write_text(SETUP_PY)
    
    print("Package structure created successfully!")
    return project_dir / "anfis_package"

def main():
    print("="*60)
    print("Standalone ANFIS Installation")
    print("="*60)
    
    try:
        package_dir = create_package()
        
        # Install the package
        print("\nInstalling package...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(package_dir)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error during installation: {result.stderr}")
            sys.exit(1)
        
        print("\n" + "="*60)
        print("Installation complete!")
        print("="*60)
        print("You can now use:")
        print("  import anfis")
        print("  from anfis.membership import GaussMf, BellMf, SigmoidMf")
        print("="*60)
        
        # Test the installation
        print("\nTesting installation...")
        try:
            import anfis
            from anfis.membership import GaussMf
            print("✓ Import test successful!")
        except Exception as e:
            print(f"✗ Import test failed: {e}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
