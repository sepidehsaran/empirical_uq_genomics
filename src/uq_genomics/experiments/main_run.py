import datetime
import warnings
import configparser
import os
import tensorflow as tf
from training_functions import *

# Neptune is optional in the released version. The experiments log metrics to it if a
# token is available, and run identically without it. Set NEPTUNE_API_TOKEN in the
# environment (never in the config file) to enable logging.
try:
    import neptune
except ImportError:                      # pragma: no cover
    neptune = None


class _NullRun:
    """Stand-in for a neptune run when logging is disabled.

    Accepts any attribute access, item assignment and call, and does nothing. This lets
    the training code keep its `run["metric"] = value` lines unchanged.
    """

    def __getitem__(self, _):
        return self

    def __setitem__(self, _k, _v):
        pass

    def __getattr__(self, _):
        return self

    def __call__(self, *_a, **_k):
        return self

    def log(self, *_a, **_k):
        pass

    def stop(self):
        pass


def init_logging(config):
    """Return a neptune run, or a no-op stand-in when no token is configured."""
    token = os.environ.get("NEPTUNE_API_TOKEN") or config.get(
        "general", "neptune_token", fallback="")
    project = config.get("general", "neptune_project", fallback="")
    if neptune is None or not token or not project:
        print("Neptune logging disabled (no NEPTUNE_API_TOKEN); continuing without it.")
        return _NullRun()
    return neptune.init_run(project=project, api_token=token)

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
tf.config.run_functions_eagerly(True)


def main_run_function(config, run):
    """
    Main function to run the script based on the settings in the config file
    and log the results to neptune.
    :param config:
    :param run:
    :return:
    """
    verbose = config.getboolean('general', 'verbose')
    run_mode = config.get('general', 'run_mode')
    model_type = config.get('model_selection', 'model_type')

    if verbose:
        print(run_mode, ' for ', model_type)

    if run_mode == 'training':
        train_func = TrainingFunctions(config)
        if model_type == 'det':
            train_func.run_train_det(config, run)
        elif model_type == 'ens':
            train_func.run_train_ens(config, run)
        elif model_type == 'bnn':
            train_func.run_train_bnn(config, run)
        elif model_type == 'mcd':
            train_func.run_train_mcd(config, run)

    elif run_mode == 'generate_evaluation':
        if model_type in ['det', 'bnn', 'mcd']:
            generate_iterations(config, run)
        elif model_type == 'ens':
            generate_iterations_for_ens(config, run)

    elif run_mode == 'all':
        train_func = TrainingFunctions(config)
        if model_type == 'det':
            train_func.run_train_det(config, run)
        elif model_type == 'ens':
            train_func.run_train_ens(config, run)
        elif model_type == 'bnn':
            train_func.run_train_bnn(config, run)
        elif model_type == 'mcd':
            train_func.run_train_mcd(config, run)

        if model_type in ['det', 'bnn', 'mcd']:
            generate_iterations(config, run)
        elif model_type == 'ens':
            generate_iterations_for_ens(config, run)

    else:
        print('invalid run mode selected.')

    return


def main():
    # read basic config
    config = configparser.ConfigParser()
    config.read(['../config/config.ini', '../config/local_config.ini'])
    verbose = config.getboolean('general', 'verbose')
    if verbose:
        print("Starting the script at {0}".format(datetime.now()))
        print(30 * "-")

    # optional experiment tracking; a no-op stand-in is used when unconfigured
    run = init_logging(config)
    assign_run_variables(run, config)
    main_run_function(config, run)

    run.stop()

    return


if __name__ == "__main__":
    main()
