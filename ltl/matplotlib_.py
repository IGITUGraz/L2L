"""
This is a utility class that imports matplotlib and does the right things e.g. supress warnings and setting the right
matplotlib backend. In the rest of the code always use ``from ltl.matplotlib_ import plt`` to get the equivalent of
```import matplotlib.pyplot as plt```
"""

import warnings
import socket

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib

    # matplotlib.rcParams.update({'text.usetex': False})
    if socket.gethostname() == 'figipc63':
        matplotlib.use('gtk3agg')
        # matplotlib.use('svg')
        # matplotlib.use('agg')
    else:
        # matplotlib.use('qt4agg')
        # matplotlib.use('svg')
        matplotlib.use('agg')
    import matplotlib.pyplot as plt_

    plt = plt_
