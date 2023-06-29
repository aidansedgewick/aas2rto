from argparse import ArgumentParser

from dk154_targets import TargetSelector
from dk154_targets.scoring_functions import peak_flux

from dk154_targets import paths

parser = ArgumentParser()
parser.add_argument("-c", "--config", default=None)
parser.add_argument("--iterations", default=None, type=int)
parser.add_argument("--existing", default=False, choices=["read", "clear"])


def empty_scoring(target, obs):
    return 100.0


def empty_modeling(target):
    return None


if __name__ == "__main__":
    args = parser.parse_args()

    config_file = args.config or paths.config_path / "selector_config.yaml"
    selector = TargetSelector.from_config(config_file)

    selector.start(
        scoring_function=peak_flux,
        modeling_function=empty_modeling,
        iterations=args.iterations,
        existing_targets=args.existing,
    )
