from pypet import Trajectory, pypetconstants, ParameterGroup

from . import timed
import logging

logger = logging.getLogger('ltl.dataprocessing')


def get_var_from_generations(traj: Trajectory, variable_name_full: str, status_interval=None):
    """
    This function extracts the contents of the variable named as specified in
    `variable_name_full`, from each generation and returns a list containing them.

    NOTE: Unlike get_var_from_runs, we do not allow fetching partially from selected
    generations and assume that the specified variable is stored for all
    generations.

    :param traj: This is the trajectory that contains the relevant information. note
        that the trajectory need not have been fully loaded beforehand as the
        loading can happen dynamically while loading specific variables

    :variable_name_full: This is the full variable name WITHOUT the prefix
        'results.generation_params', whose value you want to extract. This variable
        must be a variable stored in a generation::

            parameters.individual.coords
            results.fitness

        The first group in the name must be one of `{'results', 'parameters',
        'derived_parameters'}`. Note that whatever variable is extracted, must have
        a value corresponding to each run therefore, variables stored per generation
        cannot be accessed using this function

    :param status_interval: Sometimes the loading takes too long and you'd like a
        status message, specify this value (an integer) to have the status displayed
        every `status_interval` generations extracted

    :return: A list with each entry containing the variable value. The index in the
        list gives the generation value

    Example Usage:
    
    To get the distribution parameters from each generation of a run of cross-
    entropy, the following works.

        get_var_from_generations(traj, 'distribution_params')

    Here, 'distribution_params' is a variable stored once per generation in the
    trajectory (see the `post_process` function of the
    :class:`~ltl.optimizers.crossentropy.optimizer.CrossEntropyOptimizer` for
    further details). 
    """
    status_interval = int(status_interval) if status_interval else None

    top_group = traj.f_load_child('results.generation_params', recursive=True, load_data=pypetconstants.LOAD_SKELETON)

    def get_generation_id(gen_name):
        return int(gen_name.rsplit('_', maxsplit=1)[-1])  # generation_XXXX

    result_list = []
    gen_counter = 0
    for gen_group in top_group.f_iter_nodes(recursive=False):
        gen_id = get_generation_id(gen_group.v_name)
        result_list.append((gen_id, gen_group.f_get(variable_name_full, auto_load=True, fast_access=True)))
        if status_interval and gen_counter % status_interval == 0:
            logger.info("Completed variable extraction of {} for {} generations".format(variable_name_full, gen_counter))
        gen_counter += 1
    result_list = sorted(result_list)
    result_list = [x[1] for x in result_list]
    return result_list


def get_var_from_runs(traj: Trajectory, variable_name_full: str, run_ids=None, with_ids=False, status_interval=None):
    """
    This function retrieves a variable with name specified by `variable_name_full`
    from the trajectory `traj` from the specified runs and returns a dict containing
    them

    :param traj: This is the trajectory that contains the relevant information. note
        that the trajectory need not have been fully loaded beforehand as the
        loading can happen dynamically while loading specific variables

    :variable_name_full: This is the full variable name whose value you want to
        extract. This variable must either be a parameter, result or
        derived_parameter of a run. For example the following names are acceptable::

            parameters.individual.coords
            results.fitness

        The first group in the name must be one of `{'results', 'parameters',
        'derived_parameters'}`. Note that whatever variable is extracted, must have
        a value corresponding to each run therefore, variables stored per generation
        cannot be accessed using this function

    :param run_ids: This should be an iterator returning the run_ids from which you
        want the variables. If `None` then the variable is extracted from all runs.
        Note that if the variable is not found in a particular run, an exception is
        raised.

    :param with_ids: Boolean flag indicating whether the corresponding run indices
        should be returned as well.

    :param status_interval: Sometimes the loading takes too long and you'd like a
        status message, specify this value (an integer) to have the status displayed
        every `status_interval` runs extracted

    :return: A list containing variable value for each run. If run_ids are
        specified, the entries are arranged in the same order as run_ids, else in
        order of ascending run indices

        if ``with_ids`` is True then a tuple is returned with the first element as
        the above list and the second element containing the corresponding list
        indices

    Example Usage:

        get_var_from_runs(traj, 'parameters.individual.coords')
        get_var_from_runs(traj, 'results.fitness')
    """

    status_interval = int(status_interval) if status_interval else None
    top_group_name, variable_name = variable_name_full.split('.', maxsplit=1)
    allowed_top_group_names = {'results', 'parameters', 'derived_parameters'}
    assert top_group_name in allowed_top_group_names, \
        "The name must be of the form <top_level_group>.<variable_name> where "\
        "top_level_group is one of {}".format(allowed_top_group_names)
    top_group = traj.f_get(top_group_name)
    if run_ids is None:
        run_names = traj.f_get_run_names(sort=True)
        run_ids = iter(traj.f_idx_to_run(name) for name in run_names)
    else:
        run_ids = iter(run_ids)

    result_list = []
    run_id_list = []
    old_vidx = traj.v_idx
    try:
        run_counter = 0
        for run_id in run_ids:
            run_id_list.append(run_id)
            traj.v_idx = run_id
            if not isinstance(top_group, ParameterGroup):
                result_list.append(top_group.f_get('$set.$.' + variable_name,
                                                   auto_load=True, fast_access=True))
            else:
                result_list.append(top_group.f_get(variable_name, fast_access=True, auto_load=True))

            if status_interval and run_id % status_interval == 0:
                logger.info("Completed extraction of variable {} from {} runs.".format(variable_name_full, run_counter))
            run_counter += 1
    finally:
        traj.v_idx = old_vidx
    if with_ids:
        return result_list, run_id_list
    else:
        return result_list


def get_skeleton_traj(filename, name_or_index=-1):
    """
    This takes an hdf5 filename (which should have been generated by pypet/LTL in
    the first place) and returns a :class:`~pypet.Trajectory` instance loaded from
    the Trajectory referred to by the `name_or_index` parameter stored in the file.
    In this function, only the tree structure aka skeleton is loaded for the results
    and derived parameters, whereas the parameters are fully loaded

    This is recommended for moderately sized files (< 20 GB)

    :param filename: filename of an HDF file created by LTL
    :param name_or_index: The name or index of the trajectory to load from the file,
        if unspecified, the LAST trajectory is loaded.
    """

    traj = Trajectory(filename=filename)
    load_params_dict = {
        'load_parameters':pypetconstants.LOAD_DATA,
        'load_results':pypetconstants.LOAD_SKELETON,
        'load_derived_parameters':pypetconstants.LOAD_SKELETON,
        'force':True
    }
    if isinstance(name_or_index, str):
        load_params_dict['name'] = name_or_index
    else:
        index = int(name_or_index)
        load_params_dict['index'] = index

    # Loading Trajectory from file.
    with timed(logger, "Primary Loading of The HDF File"):
        traj.f_load(**load_params_dict)
    logger.info("Finished Primary Loading")
    return traj


def get_empty_traj(filename, name_or_index=-1):
    """
    This takes an hdf5 filename (which should have been generated by pypet/LTL in
    the first place) and returns a :class:`~pypet.Trajectory` instance loaded from
    the `index`th Trajectory stored in the file. In this function, nothing is loaded
    for the results and derived parameters, whereas the parameters are fully loaded.

    This is recommended when the file size is REALLY LARGE (e.g. > 20GB)

    :param filename: filename of an HDF file created by LTL
    :param name_or_index: The name or index of the trajectory to load from the file,
        if unspecified, the LAST trajectory is loaded.
    """
    traj = Trajectory(filename=filename)

    load_params_dict = {
        'load_parameters':pypetconstants.LOAD_DATA,
        'load_results':pypetconstants.LOAD_NOTHING,
        'load_derived_parameters':pypetconstants.LOAD_NOTHING,
        'force':True
    }
    if isinstance(name_or_index, str):
        load_params_dict['name'] = name_or_index
    else:
        index = int(name_or_index)
        load_params_dict['index'] = index

    # Loading Trajectory from file.
    with timed(logger, "Primary Loading of The HDF File"):
        traj.f_load(**load_params_dict)
    logger.info("Finished Primary Loading")
    return traj
