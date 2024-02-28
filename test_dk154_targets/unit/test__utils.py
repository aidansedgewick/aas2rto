import shutil

import pytest

import numpy as np

from astropy.time import Time

from dk154_targets.utils import calc_file_age, print_header


class Test__CalcFileAge:
    def test__existing_file(self, tmp_path):
        test_file = tmp_path / "test_file.txt"
        assert not test_file.exists()

        t_write = Time.now()
        with open(test_file, "w+") as f:
            f.write("some_data")
        assert test_file.exists()

        t_future = Time(t_write.jd + 10.0, format="jd")
        file_age = calc_file_age(test_file, t_future)

        assert np.isclose(file_age, 10.0, atol=3e-5)  # tolerance of 2.5 seconds...

    def test__missing_file(self, tmp_path):
        t_now = Time.now()

        file_path = tmp_path / "missing_file.csv"
        assert not file_path.exists()

        file_age = calc_file_age(file_path, t_now)  # Should return positive infinity.

        assert np.isposinf(file_age)

    def test__fails_missing_file_disallow_missing(self, tmp_path):
        t_now = Time.now()

        file_path = tmp_path / "missing_file.csv"
        assert not file_path.exists()

        with pytest.raises(IOError):
            file_age = calc_file_age(file_path, t_now, allow_missing=False)


class Test__PrintHeader:
    def test__normal_behaviour(self):
        txt = "some header here"

        print_header(txt)  # Should not crash...

    def test__no_shutil(self, monkeypatch):
        assert callable(shutil.get_terminal_size)
        with monkeypatch.context() as m:
            m.setattr("shutil.get_terminal_size", None)
            assert not callable(shutil.get_terminal_size)

        assert callable(shutil.get_terminal_size)
