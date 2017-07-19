import os
import logging
import logging.config
import scoop
import scoop.shared
import socket
import copy


def create_shared_logger_data(logger_names, log_levels, log_to_consoles,
                              sim_name, log_directory, multiproc=True):
    if multiproc:
        assert scoop.IS_RUNNING, \
            "multi_proc=True cannot be called without having scoop running"
        assert scoop.IS_ORIGIN, \
            "create_shared_logger_data must be called only on the origin worker"
        scoop.shared.setConst(logger_names=logger_names, log_levels=log_levels,
                              sim_name=sim_name, log_directory=log_directory, log_to_consoles=log_to_consoles)
    else:
        global logger_names_global, log_levels_global, log_to_consoles_global
        global sim_name_global, log_directory_global
        logger_names_global = logger_names
        log_levels_global = log_levels
        log_to_consoles_global = log_to_consoles
        sim_name_global = sim_name
        log_directory_global = log_directory


def configure_loggers(multiproc=False):
    if multiproc:
        assert scoop.IS_RUNNING, \
            "configure_loggers(True) cannot be called without having scoop running"

        # Get shared data from scoop and perform the relevant configuration
        logger_names = scoop.shared.getConst('logger_names', timeout=1.0)
        log_levels = scoop.shared.getConst('log_levels', timeout=1.0)
        log_to_consoles = scoop.shared.getConst('log_to_consoles', timeout=1.0)
        sim_name = scoop.shared.getConst('sim_name', timeout=1.0)
        log_directory = scoop.shared.getConst('log_directory', timeout=1.0)
        if logger_names is None:
            return
    else:
        # Get logger data from global variables and perform the relevant thing
        logger_names = logger_names_global
        log_levels = log_levels_global
        log_to_consoles = log_to_consoles_global
        sim_name = sim_name_global
        log_directory = log_directory_global

    if multiproc and not scoop.IS_ORIGIN:
        file_name_prefix = '%s_%s_%s_' % (sim_name, socket.gethostname(), os.getpid())
    else:
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
