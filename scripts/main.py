import logging
import yaml

from argparse import ArgumentParser

from dk154_targets import TargetSelector
from dk154_targets.scoring import latest_flux, supernova_peak_score
from dk154_targets.modeling import empty_modeling, sncosmo_model

from dk154_targets import paths

logger = logging.getLogger("main")

parser = ArgumentParser()
parser.add_argument("-c", "--config", default=None)
parser.add_argument("--iterations", default=None, type=int)
parser.add_argument("--existing", default=False, choices=["read", "clear"])
parser.add_argument("--debug", default=False, action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.debug:
        debug_logging_config = paths.config_path / "logging/debug_logging.yaml"
        with open(debug_logging_config, "rt") as f:
            log_config = yaml.safe_load(f.read())
        logging.config.dictConfig(log_config)

    config_file = args.config or paths.config_path / "selector_config.yaml"
    selector = TargetSelector.from_config(config_file)

    selector.start(
        scoring_function=latest_flux,
        modeling_function=empty_modeling,
        iterations=args.iterations,
        existing_targets=args.existing,
    )
