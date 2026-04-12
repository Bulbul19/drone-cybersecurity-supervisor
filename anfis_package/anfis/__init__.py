#!/usr/bin/env python3
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
