import os
import logging
import logging.config
import socket
import copy


def create_shared_logger_data(logger_names, log_levels, log_to_consoles,
                              sim_name, log_directory):
    """
    This function must be called to create a shared copy of information that will be
    required to setup logging across processes. This must be run exactly once in the
    root process.

    :param logger_names: This is a list of names of the loggers whose output you're
        interested in.

    :param log_levels: This is the list of the same size of `logger_names` containing
        the log levels specified as strings (e.g. 'INFO', 'DEBUG', 'WARNING', 'ERROR').

    :param log_to_consoles: This is a list of the same size of `logger_names` containing
        boolean values which indicate whether or not to redirect the output of the said
        logger to stdout or not. Note that with scoop, and output to stdout on any
        worker gets directed to the console of the main process.

    :param sim_name: This is a string that is used when creating the log files.
        Short for simulation name.

    :param log_directory: This is the path of the directory in which the log files will
        be stored. This directory must be an existing directory.
    """

    # process / validate input
    assert len(logger_names) == len(log_levels) == len(log_to_consoles), \
        "The sizes of logger_names, log_levels, log_to_consoles are inconsistent"
    assert all(isinstance(x, str) for x in logger_names + log_levels), \
        "'logger_names' and 'log_levels' must be lists of strings"
    assert os.path.isdir(log_directory), "The log_directory {} is not a vlid log directory".format(log_directory)

    log_to_consoles = [bool(x) for x in log_to_consoles]

    global logger_names_global, log_levels_global, log_to_consoles_global
    global sim_name_global, log_directory_global
    logger_names_global = logger_names
    log_levels_global = log_levels
    log_to_consoles_global = log_to_consoles
    sim_name_global = sim_name
    log_directory_global = log_directory


def configure_loggers(exactly_once=False):
    """
    This function configures the loggers using the shared information that was set by
    :func:`.create_shared_logger_data`. This function must be run at the beginning
    of every function that is parallelized in order to be able to reliably
    configure the loggers. As an example look at its usage in the method
    :meth:`~.FunctionGeneratorOptimizee.simulate()` from the
    class :class:`~.FunctionGeneratorOptimizee`

    You may also wish to call this function in your main simulation (after calling
    :func:`.create_shared_logger_data`) to configure the logging for the root process
    before any of the parallelized functions are run.

    :param exactly_once: If the configuration of logging is causing a significant
        overhead per parallelized run (This is a rather unlikely scenario), then this
        value may be set to `True`. When True, the function will configure the loggers
        exactly once per scoop worker.
    """

    if exactly_once and configure_loggers._already_configured:
        return

    # Scoop logging has been removed, as JUBE takes care of the logging of each iteration
    # Get logger data from global variables and perform the relevant thing
    logger_names = logger_names_global
    log_levels = log_levels_global
    log_to_consoles = log_to_consoles_global
    sim_name = sim_name_global
    log_directory = log_directory_global

    file_name_prefix = '%s_' % (sim_name,)

    config_dict_copy = copy.deepcopy(configure_loggers.basic_config_dict)

    config_dict_copy['loggers'] = {}

    # Configuring the output files
    log_fname = os.path.join(log_directory,
                             file_name_prefix + config_dict_copy['handlers']['file_log']['filename'])
    error_fname = os.path.join(log_directory,
                               file_name_prefix + config_dict_copy['handlers']['file_error']['filename'])
    config_dict_copy['handlers']['file_log']['filename'] = log_fname
    config_dict_copy['handlers']['file_error']['filename'] = error_fname

    # Creating logger entries
    for logger_name, log_level, log_to_console in zip(logger_names, log_levels, log_to_consoles):
        config_dict_copy['loggers'][logger_name] = {}
        logger_dict = config_dict_copy['loggers'][logger_name]
        logger_dict['level'] = log_level
        if log_to_console:
            logger_dict['handlers'] = ['console', 'file_log', 'file_error']
        else:
            logger_dict['handlers'] = ['file_log', 'file_error']

    logging.config.dictConfig(config_dict_copy)
    configure_loggers._already_configured = True


configure_loggers._already_configured = False
configure_loggers.basic_config_dict = {
    'version': 1,
    'formatters': {
        'file': {
            'format': '%(asctime)s %(name)s {} %(process)d %(levelname)-8s: %(message)s'.format(socket.gethostname())
        },
        'stream': {
            'format': '%(processName)-10s %(name)s {} %(process)d %(levelname)-8s: %(message)s'.format(
                socket.gethostname())
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
            'formatter': 'stream',
        },
        'file_log': {
            'class': 'logging.FileHandler',
            'formatter': 'file',
            'filename': 'LOG.txt',
        },
        'file_error': {
            'class': 'logging.FileHandler',
            'formatter': 'file',
            'filename': 'ERROR.txt',
            'level': 'ERROR',
        },
    },
    'loggers': {},
    'root': {
        # 'level': 'INFO',
        'handlers': []
    }
}
