from pathlib import Path


base_path = Path(__file__).parent.parent

config_path = base_path / "config"

wkdir = Path.cwd()

test_path = base_path / "test_aas2rto"
