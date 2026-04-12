#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    ANFIS in torch: some fuzzy membership functions.
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch

def _mk_param(val):
    if isinstance(val, torch.Tensor):
        val = val.item()
    return torch.nn.Parameter(torch.tensor(val, dtype=torch.float))

class GaussMembFunc(torch.nn.Module):
    def __init__(self, mu, sigma):
        super(GaussMembFunc, self).__init__()
        self.register_parameter('mu', _mk_param(mu))
        self.register_parameter('sigma', _mk_param(sigma))

    def forward(self, x):
        val = torch.exp(-torch.pow(x - self.mu, 2) / (2 * self.sigma**2))
        return val

class BellMembFunc(torch.nn.Module):
    def __init__(self, a, b, c):
        super(BellMembFunc, self).__init__()
        self.register_parameter('a', _mk_param(a))
        self.register_parameter('b', _mk_param(b))
        self.register_parameter('c', _mk_param(c))
        self.b.register_hook(BellMembFunc.b_log_hook)

    @staticmethod
    def b_log_hook(grad):
        grad[torch.isnan(grad)] = 1e-9
        return grad

    def forward(self, x):
        dist = torch.pow((x - self.c)/self.a, 2)
        return torch.reciprocal(1 + torch.pow(dist, self.b))

class TriangularMembFunc(torch.nn.Module):
    def __init__(self, a, b, c):
        super(TriangularMembFunc, self).__init__()
        assert a <= b <= c, 'Triangular parameters must satisfy a <= b <= c'
        self.register_parameter('a', _mk_param(a))
        self.register_parameter('b', _mk_param(b))
        self.register_parameter('c', _mk_param(c))

    def forward(self, x):
        ascending = (self.a < x) & (x <= self.b)
        descending = (self.b < x) & (x <= self.c)
        result = torch.zeros_like(x)
        result[ascending] = (x[ascending] - self.a) / (self.b - self.a)
        result[descending] = (self.c - x[descending]) / (self.c - self.b)
        return result

class TrapezoidalMembFunc(torch.nn.Module):
    def __init__(self, a, b, c, d):
        super(TrapezoidalMembFunc, self).__init__()
        assert a <= b <= c <= d, 'Trapezoid parameters must satisfy a <= b <= c <= d'
        self.register_parameter('a', _mk_param(a))
        self.register_parameter('b', _mk_param(b))
        self.register_parameter('c', _mk_param(c))
        self.register_parameter('d', _mk_param(d))

    def forward(self, x):
        yvals = torch.zeros_like(x)
        rising = (self.a < x) & (x <= self.b)
        plateau = (self.b < x) & (x < self.c)
        falling = (self.c <= x) & (x < self.d)
        yvals[rising] = (x[rising] - self.a) / (self.b - self.a)
        yvals[plateau] = 1
        yvals[falling] = (self.d - x[falling]) / (self.d - self.c)
        return yvals

# --- Compatibility helpers (return lists of membership functions) ---
def _ensure_iterable(arr):
    if hasattr(arr, 'tolist'):
        return arr.tolist()
    if np.isscalar(arr):
        return [float(arr)]
    return list(arr)

def GaussMf(mu_array, sigma_array):
    mus = _ensure_iterable(mu_array)
    sigmas = _ensure_iterable(sigma_array)
    if len(mus) != len(sigmas):
        raise ValueError("mu_array and sigma_array must have the same length")
    return [GaussMembFunc(float(mu), float(sig)) for mu, sig in zip(mus, sigmas)]

def BellMf(a_array, b_array, c_array):
    a_vals = _ensure_iterable(a_array)
    b_vals = _ensure_iterable(b_array)
    c_vals = _ensure_iterable(c_array)
    if not (len(a_vals) == len(b_vals) == len(c_vals)):
        raise ValueError("BellMf inputs must have the same length")
    return [BellMembFunc(float(a), float(b), float(c))
            for a, b, c in zip(a_vals, b_vals, c_vals)]

def SigmoidMf(*args, **kwargs):
    # No explicit sigmoid class in this repo; reuse Bell definition
    return BellMf(*args, **kwargs)
