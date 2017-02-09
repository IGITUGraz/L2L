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
