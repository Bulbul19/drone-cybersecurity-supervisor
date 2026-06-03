#!/usr/bin/env python3
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
