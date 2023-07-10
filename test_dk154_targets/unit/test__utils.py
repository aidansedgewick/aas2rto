import os
import pytest

import numpy as np

import pandas as pd

from astropy import units as u
from astropy.time import Time

from dk154_targets import utils
from dk154_targets import paths


def test__calc_file_age():
    df = pd.DataFrame([("a", 1, 10), ("b", 2, 20)], columns="name x y".split())
    df_file = paths.test_data_path / "test_file.csv"

    t_ref = Time.now() + 6 * u.hour
    df.to_csv(df_file, index=False)
    file_age = utils.calc_file_age(df_file, t_ref)
    atol = 1e-4
    assert np.isclose(file_age, 0.25, atol=atol)
    os.remove(df_file)

    non_existant_file = paths.test_data_path / "another_test.csv"
    assert not non_existant_file.exists()

    # Default behaviour
    non_existant_file_age = utils.calc_file_age(non_existant_file, t_ref)
    assert not np.isfinite(non_existant_file_age)

    # Raises error
    with pytest.raises(IOError):
        fail_age = utils.calc_file_age(non_existant_file, t_ref, allow_missing=False)


def test__print_header():
    test_string = "blah"
    utils.print_header(test_string)

    test_string2 = "a" * 100
    utils.print_header(test_string2)
