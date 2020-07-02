#!/usr/bin/env python
# coding: utf-8

"""
@File      : __init__.py
@author    : alex
@Date      : 2020/6/28
@Desc      : 
"""

from .lenet import LeNet
from .alexnet import AlexNet
from .vggnet import *

__all__ = [k for k in list(globals().keys()) if not k.startswith("_")]
