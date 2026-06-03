#!/usr/bin/env python3
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
