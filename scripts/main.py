import logging
import yaml
from argparse import ArgumentParser
from pathlib import Path

from astropy.time import Time

from aas2rto import TargetSelector
from aas2rto.modeling import empty_modeling, SncosmoSaltModeler
from aas2rto.plotting import plot_default_lightcurve, plot_sncosmo_lightcurve
from aas2rto.plotting.salt_param_plotter import SaltParamPlottingWrapper
from aas2rto.scoring import (
    example_functions,
    SupernovaPeakScore,
    KilonovaDiscReject,
)
from aas2rto import paths

logger = logging.getLogger("main")

parser = ArgumentParser()
parser.add_argument("-c", "--config", default=None, required=True)
parser.add_argument("-i", "--iterations", default=None, type=int)
parser.add_argument(
    "-x", "--recovery-file", default=False, const="last", nargs="?", type=str
)
parser.add_argument("--skip-tasks", nargs="*")
parser.add_argument("--debug", default=False, action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.debug:
        debug_logging_config = paths.config_path / "logging/debug_logging.yaml"
        with open(debug_logging_config, "rt") as f:
            log_config = yaml.safe_load(f.read())
        logging.config.dictConfig(log_config)

    config_file = args.config or paths.config_path / "selector_config.yaml"
    config_file = Path(config_file)

    if not config_file.exists():
        raise FileNotFoundError(f"no config file at {config_file}")
    config_file = Path(args.config)
    selector = TargetSelector.from_config(config_file)

    extra_plotting_functions = []
    lc_plotting_function = plot_default_lightcurve
    if "supernovae" in config_file.stem or "sne" in config_file.stem:
        scoring_function = SupernovaPeakScore(
            use_compiled_lightcurve=True
        )  # This is a class - initialise it!

        models_path = None  # selector.path_manager.project_path / "sncosmo_salt_models"
        modeling_function = SncosmoSaltModeler(
            existing_models_path=models_path,
            use_emcee=False,
            show_traceback=True,
        )
        lc_plotting_function = plot_sncosmo_lightcurve

        salt_param_plotter = SaltParamPlottingWrapper()
        extra_plotting_functions.append(salt_param_plotter)

    elif "atlas_test" in config_file.stem:
        scoring_function = example_functions.latest_flux_atlas_requirement
        modeling_function = empty_modeling
    elif "kn" in config_file.stem:
        scoring_function = KilonovaDiscReject()
        modeling_function = empty_modeling
    else:
        scoring_function = example_functions.latest_flux
        modeling_function = empty_modeling

    logger.info(f"use \033[36;1m {scoring_function.__name__}\033[0m scoring")
    logger.info(f"use \033[36;1m {modeling_function.__name__}\033[0m model")

    if not args.recovery_file:
        logger.info(f"NOT attempting to recover existing targets")

    try:
        selector.start(
            scoring_function=scoring_function,
            modeling_function=modeling_function,
            lc_plotting_function=lc_plotting_function,
            extra_plotting_functions=extra_plotting_functions,
            iterations=args.iterations,
            recovery_file=args.recovery_file,
            skip_tasks=args.skip_tasks,
        )
    except Exception as e:
        t_crash = Time.now()
        timestamp = t_crash.strftime("%y-%m-%d %H:%M:%S")
        selector.send_crash_reports(
            text=f"CRASH at {timestamp}!\nexception caught in main try/except"
        )
