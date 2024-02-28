import logging
import yaml
from argparse import ArgumentParser
from pathlib import Path

from astropy.time import Time

from dk154_targets import TargetSelector
from dk154_targets.scoring import (
    example_functions,
    SupernovaPeakScore,
    KilonovaDiscReject,
)
from dk154_targets.modeling import empty_modeling, sncosmo_salt

from dk154_targets import paths

logger = logging.getLogger("main")

parser = ArgumentParser()
parser.add_argument("-c", "--config", default=None, required=True)
parser.add_argument("-i", "--iterations", default=None, type=int)
parser.add_argument(
    "-x", "--existing-targets-file", default=False, const="last", nargs="?", type=str
)
parser.add_argument("--debug", default=False, action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.debug:
        debug_logging_config = paths.config_path / "logging/debug_logging.yaml"
        with open(debug_logging_config, "rt") as f:
            log_config = yaml.safe_load(f.read())
        logging.config.dictConfig(log_config)

    config_file = args.config or paths.config_path / "selector_config.yaml"
    config_file = Path(args.config)

    if "supernovae" in config_file.stem:
        scoring_function = SupernovaPeakScore()  # This is a class - initialise it!
        modeling_function = sncosmo_salt()
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

    if not args.existing_targets_file:
        logger.info(f"NOT attempting to recover existing targets")

    selector = TargetSelector.from_config(config_file)
    try:
        selector.start(
            scoring_function=scoring_function,
            modeling_function=modeling_function,
            iterations=args.iterations,
            existing_targets_file=args.existing_targets_file,
        )
    except Exception as e:
        t_crash = Time.now()
        timestamp = t_crash.strftime("%y-%m-%d %H:%M:%S")
        selector.send_crash_reports(
            text=f"CRASH at {timestamp}!\nexception caught in main try/except"
        )
