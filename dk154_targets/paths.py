from pathlib import Path

base_path = Path(__file__).parent.parent

config_path = base_path / "config"

default_data_dir = "data"
default_outputs_dir = "outputs"
default_opp_targets_dir = "opp_targets"
default_scratch_dir = "scratch"

wkdir = Path.cwd()

test_path = base_path / "test_dk154_targets"
