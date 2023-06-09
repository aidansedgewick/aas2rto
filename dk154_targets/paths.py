from pathlib import Path

base_path = Path(__file__).parent.parent

config_path = base_path / "config"

default_data_path = base_path / "data"
default_outputs_path = base_path / "outputs"

test_path = base_path / "test_dk154_targets"
test_data_path = test_path / "test_data"

scratch_path = base_path / "scratch"


def build_paths():
    config_path.mkdir(exist_ok=True, parents=True)
    test_data_path.mkdir(exist_ok=True, parents=True)
    scratch_path.mkdir(exist_ok=True, parents=True)
