from __future__ import print_function
from __future__ import division
import numpy as np
import tensorflow as tf
from datetime import datetime


def print_time(s, **kwargs):
    now = datetime.now()
    prt = (now.year, now.month, now.day, now.hour, now.minute, now.second,
           str(s))
    print('[%4d-%02d-%02d %02d:%02d:%02d] %s' % prt, **kwargs)
