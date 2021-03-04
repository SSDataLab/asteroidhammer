#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
from .version import __version__
from .main import Data
from .matrices import *
from .plotting import *
from .background import *
from .rip import *
