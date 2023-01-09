import numpy as np
import math
from collections import namedtuple
from scipy.sparse import diags, eye

def spin_operators(S, *, to_dense_array=False, format=None, dtype=np.float_):
    Sz = diags([m for m in np.arange(-S, S + 1)], format=format, dtype=dtype)
    Sp = diags([math.sqrt(S * (S + 1) - m * (m + 1)) for m in np.arange(-S, S)],
               offsets=-1,
               format=format,
               dtype=dtype
               )
    Sm = Sp.T
    Seye = eye(2 * S + 1, format=format, dtype=dtype)

    Spin_operators = namedtuple('Spin_operators', 'Sz Sp Sm Seye')
    ops = Spin_operators(Sz, Sp, Sm, Seye)
    if to_dense_array:
        ops = Spin_operators(*[o.toarray() for o in ops])

    return ops