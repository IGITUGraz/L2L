u"""
This is a utility module that imports matplotlib and does the right things e.g. supress warnings and set the right
matplotlib backend. In the rest of the code always use ``from ltl.matplotlib_ import plt`` to get the equivalent of
```import matplotlib.pyplot as plt```
"""

from __future__ import with_statement
from __future__ import absolute_import
import warnings

with warnings.catch_warnings():
    warnings.simplefilter(u"ignore")
    import matplotlib

    # matplotlib.rcParams.update({'text.usetex': False})
    # matplotlib.use('gtk3agg')
    # matplotlib.use('qt4agg')
    # matplotlib.use('tkagg')
    # matplotlib.use('svg')
    matplotlib.use(u'agg')
    import matplotlib.pyplot as plt_

    plt = plt_
