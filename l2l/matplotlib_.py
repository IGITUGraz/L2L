"""
This is a utility module that imports matplotlib and does the right things e.g. supress warnings and set the right
matplotlib backend. In the rest of the code always use ``from l2l.matplotlib_ import plt`` to get the equivalent of
```import matplotlib.pyplot as plt```
"""

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import matplotlib

    # matplotlib.rcParams.update({'text.usetex': False})
    # matplotlib.use('gtk3agg')
    # matplotlib.use('qt4agg')
    # matplotlib.use('tkagg')
    # matplotlib.use('svg')
    matplotlib.use('agg')
    import matplotlib.pyplot as plt_

    plt = plt_
