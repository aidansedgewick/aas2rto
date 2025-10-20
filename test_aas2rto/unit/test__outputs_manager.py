import pytest
from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time

from astroplan import Observer

from aas2rto.exc import (
    MissingEphemInfoWarning,
    UnexpectedKeysWarning,
    UnknownTargetWarning,
)
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup
from aas2rto.observatory_manager import ObservatoryManager
from aas2rto.outputs_manager import OutputsManager
from aas2rto.path_manager import PathManager


@pytest.fixture
def mod_tl(tlookup: TargetLookup, lasilla: Observer, t_fixed: Time):
    tlookup["T00"].update_science_score_history(10.0, t_ref=t_fixed)
    tlookup["T00"].update_obs_score_history(3.0, lasilla, t_ref=t_fixed)
    tlookup["T00"].update_obs_score_history(0.5, "astrolab", t_ref=t_fixed)
    tlookup["T00"].science_comments = ["T00 comment"]
    tlookup["T00"].obs_comments["lasilla"] = ["T00 lasilla comment"]
    tlookup["T00"].obs_comments["astrolab"] = ["T00 astrolab comment"]

    tlookup["T01"].update_science_score_history(5.0, t_ref=t_fixed)
    tlookup["T01"].update_obs_score_history(1.0, lasilla, t_ref=t_fixed)
    tlookup["T01"].update_obs_score_history(2.0, "astrolab", t_ref=t_fixed)
    tlookup["T01"].science_comments = ["T01 comment"]
    tlookup["T01"].obs_comments["lasilla"] = ["T01 lasilla comment"]
    tlookup["T01"].obs_comments["astrolab"] = ["T01 astrolab comment"]
    return tlookup


@pytest.fixture
def outputs_mgr(
    mod_tl: TargetLookup, path_mgr: PathManager, obs_mgr: ObservatoryManager
):
    return OutputsManager({}, mod_tl, path_mgr, obs_mgr)


@pytest.fixture
def outputs_mgr_with_plots(
    mod_tl: TargetLookup, path_mgr: PathManager, obs_mgr: ObservatoryManager
):
    om = OutputsManager({}, mod_tl, path_mgr, obs_mgr)
    for target_id, target in mod_tl.items():
        lc_path = path_mgr.get_lightcurve_plot_path(target_id)
        blank_plot_helper(lc_path)
        target.lc_fig_path = lc_path
        for obs_name in obs_mgr.sites.keys():
            vis_path = path_mgr.get_visibility_plot_path(target_id, obs_name)
            blank_plot_helper(vis_path)
            target.vis_fig_paths[obs_name] = vis_path
    return om


@pytest.fixture
def tl_vis_targets(lasilla: Observer, t_fixed: Time):
    # Reverse engineer some targets which will be visible and not visible at t_fixed
    midnight = lasilla.midnight(t_fixed, which="next", n_grid_points=40)
    t_early = midnight - 3.0 * u.hour  # earlier
    t_later = midnight + 3.0 * u.hour  # later

    lst_e = lasilla.local_sidereal_time(t_early)  # has units deg
    lst_m = lasilla.local_sidereal_time(midnight)
    lst_l = lasilla.local_sidereal_time(t_later)

    o_lat = lasilla.location.lat  # dec = lat +/- ZD, ZD = (90-transit_alt)
    c00 = SkyCoord(ra=lst_m, dec=o_lat - (90.0 - 80.0) * u.deg)  # 00:00 local
    c01 = SkyCoord(ra=lst_l, dec=o_lat - (90.0 - 80.0) * u.deg)  # 03:00 local
    c02 = SkyCoord(ra=lst_m, dec=o_lat + (90.0 - 20.0) * u.deg)  # too low (North)
    c03 = SkyCoord(ra=lst_e, dec=o_lat - (90.0 - 80.0) * u.deg)  # 21:00 local
    c04 = SkyCoord(ra=lst_m, dec=o_lat - (90.0 - 80.0) * u.deg)  # bad score
    c05 = SkyCoord(ra=lst_m, dec=o_lat - (90.0 - 80.0) * u.deg)  # no score

    target_list = [
        Target("Tv00", c00.transform_to("icrs")),
        Target("Tv01", c01.transform_to("icrs")),
        Target("Tv02", c02.transform_to("icrs")),
        Target("Tv03", c03.transform_to("icrs")),
        Target("Tv04", c04.transform_to("icrs")),
        Target("Tv05", c05.transform_to("icrs")),
    ]

    tl = TargetLookup()
    tl.add_target_list(target_list)
    tl["Tv00"].update_science_score_history(1.0, midnight)
    tl["Tv01"].update_science_score_history(1.0, midnight)
    tl["Tv02"].update_science_score_history(1.0, midnight)
    tl["Tv03"].update_science_score_history(1.0, midnight)
    tl["Tv04"].update_science_score_history(-1.0, midnight)
    # Tv05 has no score!
    return tl


@pytest.fixture  #
def om_vis_targets(
    tl_vis_targets: TargetLookup,
    path_mgr: PathManager,
    obs_mgr_config: dict,
    t_fixed: Time,
):
    obs_mgr = ObservatoryManager(obs_mgr_config, tl_vis_targets, path_mgr)
    om = OutputsManager({}, tl_vis_targets, path_mgr, obs_mgr)
    om.observatory_manager.apply_ephem_info(t_ref=t_fixed)
    return om


def blank_plot_helper(filepath):
    fig, ax = plt.subplots(figsize=(1, 1))
    fig.savefig(filepath)
    plt.close(fig)
    return


class Test__BlankPlotHelper:
    def test__blank_plot_helper(self, tmp_path: Path):
        # Arrange
        test_path = tmp_path / "test.png"

        # Act
        blank_plot_helper(test_path)

        # Assert
        test_path.exists()


class Test__OutputMgrInit:
    def test__outmgr_init(
        self, tlookup: TargetLookup, path_mgr: PathManager, obs_mgr: ObservatoryManager
    ):
        # Act
        om = OutputsManager({}, tlookup, path_mgr, obs_mgr)

        # Assert
        assert om.science_ranked_list is None
        assert isinstance(om.obs_ranked_lists, dict)
        assert isinstance(om.obs_visible_lists, dict)

    def test__outmgr_bad_config_warns(
        self, tlookup: TargetLookup, path_mgr: PathManager, obs_mgr: ObservatoryManager
    ):
        # Arrange
        config = {"bad_key": 100.0}

        # Act
        with pytest.warns(UnexpectedKeysWarning):
            om = OutputsManager(config, tlookup, path_mgr, obs_mgr)  # noqa: F841


class Test__WriteComments:
    def test__write_comments(self, outputs_mgr: OutputsManager, t_fixed: Time):
        # Act
        outputs_mgr.write_target_comments(t_ref=t_fixed)

        # Assert
        comms_path = outputs_mgr.path_manager.comments_path
        exp_T00_comms_filepath = comms_path / "T00_comments.txt"
        assert exp_T00_comms_filepath.exists()
        with open(exp_T00_comms_filepath, "r") as f:
            comm_str = "".join(f.readlines())

        print(comm_str)
        assert "T00 comment" in comm_str
        assert "T00 lasilla comment" in comm_str
        assert "T00 astrolab comment" in comm_str

        exp_T01_comms_filepath = comms_path / "T01_comments.txt"
        assert exp_T01_comms_filepath.exists()
        with open(exp_T01_comms_filepath, "r") as f:
            comm_str = "\n".join(f.readlines())

        assert "T01 comment" in comm_str
        assert "T01 lasilla comment" in comm_str
        assert "T01 astrolab comment" in comm_str

    def test__write_comm_subset(self, outputs_mgr: OutputsManager, t_fixed: Time):
        # Act
        outputs_mgr.write_target_comments(target_id_list=["T00"])

        # Assert
        comms_path = outputs_mgr.path_manager.comments_path
        exp_T00_comms_filepath = comms_path / "T00_comments.txt"
        assert exp_T00_comms_filepath.exists()
        exp_T01_comms_filepath = comms_path / "T01_comments.txt"
        assert not exp_T01_comms_filepath.exists()

    def test__missing_target_warns(self, outputs_mgr: OutputsManager, t_fixed: Time):
        # Act
        with pytest.warns(UnknownTargetWarning):
            outputs_mgr.write_target_comments(target_id_list=["Tx"])


class Test__RankObsOnScienceScore:
    def test__all_ranked(self, outputs_mgr: OutputsManager, t_fixed: Time):
        # Act
        sci_list = outputs_mgr.rank_targets_on_science_score(
            plots=False, write_list=True
        )

        # Assert
        assert isinstance(sci_list, pd.DataFrame)
        assert len(sci_list) == 2
        assert sci_list["target_id"].iloc[0] == "T00"
        assert np.isclose(sci_list["score"].iloc[0], 10.0)
        assert sci_list["target_id"].iloc[1] == "T01"
        assert np.isclose(sci_list["score"].iloc[1], 5.0)

        T00 = outputs_mgr.target_lookup["T00"]
        T01 = outputs_mgr.target_lookup["T01"]
        assert T00.get_latest_science_rank() == 1
        assert T01.get_latest_science_rank() == 2

        outputs_path = outputs_mgr.path_manager.outputs_path
        exp_path = outputs_path / "ranked_lists/science_score.csv"
        assert exp_path.exists()

    def test__excluded_not_in_list(self, outputs_mgr: OutputsManager, t_fixed: Time):
        # Arrange
        T00 = outputs_mgr.target_lookup["T00"]
        T01 = outputs_mgr.target_lookup["T01"]
        T01.update_science_score_history(-1.0, t_fixed + 1.0 * u.day)

        # Act
        sci_list = outputs_mgr.rank_targets_on_science_score(
            plots=False, write_list=True, t_ref=t_fixed
        )

        # Assert
        assert isinstance(sci_list, pd.DataFrame)
        assert len(sci_list) == 1
        assert sci_list["target_id"].iloc[0] == "T00"

        assert T00.get_latest_science_rank() == 1
        assert T01.get_latest_science_rank() == 9999

    def test__all_ranked_with_plots(
        self, outputs_mgr_with_plots: OutputsManager, t_fixed: Time
    ):
        # Arrange
        outputs_mgr = outputs_mgr_with_plots  # short name is nice

        # Act
        sci_list = outputs_mgr.rank_targets_on_science_score(
            plots=True, write_list=True, t_ref=t_fixed
        )

        # Assert
        outputs_path = outputs_mgr.path_manager.outputs_path
        T00_new_lc_path = outputs_path / "plots/science_score/001_T00_lc.png"
        assert T00_new_lc_path.exists()
        T01_new_lc_path = outputs_path / "plots/science_score/002_T01_lc.png"
        assert T01_new_lc_path.exists()


class Test__RankedListAtObs:
    def test__rank_at_obs_observer(
        self, outputs_mgr: OutputsManager, lasilla: Observer, t_fixed: Time
    ):
        # Act
        obs_list = outputs_mgr.rank_targets_on_obs_score(
            lasilla, plots=False, write_list=True, t_ref=t_fixed
        )

        # Assert
        assert isinstance(obs_list, pd.DataFrame)
        assert len(obs_list) == 2
        assert obs_list["target_id"].iloc[0] == "T00"
        assert np.isclose(obs_list["score"].iloc[0], 3.0)
        assert obs_list["target_id"].iloc[1] == "T01"
        assert np.isclose(obs_list["score"].iloc[1], 1.0)

        T00 = outputs_mgr.target_lookup["T00"]
        T01 = outputs_mgr.target_lookup["T01"]
        assert T00.get_latest_obs_rank(lasilla) == 1
        assert T01.get_latest_obs_rank(lasilla) == 2

        outputs_path = outputs_mgr.path_manager.outputs_path
        exp_path = outputs_path / "ranked_lists/obs_lasilla.csv"
        assert exp_path.exists()

    def test__rank_at_obs_str(self, outputs_mgr: OutputsManager, t_fixed: Time):
        # Act
        obs_list = outputs_mgr.rank_targets_on_obs_score(
            "astrolab", plots=False, write_list=True, t_ref=t_fixed
        )

        # Assert
        assert isinstance(obs_list, pd.DataFrame)
        assert len(obs_list) == 2
        assert obs_list["target_id"].iloc[0] == "T01"
        assert np.isclose(obs_list["score"].iloc[0], 2.0)
        assert obs_list["target_id"].iloc[1] == "T00"
        assert np.isclose(obs_list["score"].iloc[1], 0.5)

        T00 = outputs_mgr.target_lookup["T00"]
        T01 = outputs_mgr.target_lookup["T01"]
        assert T01.get_latest_obs_rank("astrolab") == 1
        assert T00.get_latest_obs_rank("astrolab") == 2

        outputs_path = outputs_mgr.path_manager.outputs_path
        exp_path = outputs_path / "ranked_lists/obs_astrolab.csv"
        assert exp_path.exists()

    def test__rank_obs_with_plots(
        self,
        outputs_mgr_with_plots: OutputsManager,
        lasilla: Observer,
        t_fixed: Time,
    ):
        # Arrange
        outputs_mgr = outputs_mgr_with_plots  # short name is nice

        # Act
        obs_list = outputs_mgr.rank_targets_on_obs_score(
            lasilla, plots=True, write_list=True, t_ref=t_fixed
        )

        # Assert
        outputs_path = outputs_mgr.path_manager.outputs_path
        T00_new_lc_path = outputs_path / "plots/obs_lasilla/001_T00_lc.png"
        assert T00_new_lc_path.exists()
        T01_new_lc_path = outputs_path / "plots/obs_lasilla/002_T01_lc.png"
        assert T01_new_lc_path.exists()

        T00_new_vis_path = outputs_path / "plots/obs_lasilla/001_T00_lasilla_vis.png"
        assert T00_new_vis_path.exists()
        T01_new_vis_path = outputs_path / "plots/obs_lasilla/002_T01_lasilla_vis.png"
        assert T01_new_vis_path.exists()


class Test__BuildAllRankedLists:
    def test__all_lists_and_plots(
        self,
        outputs_mgr_with_plots: OutputsManager,
        lasilla: Observer,
        t_fixed: Time,
    ):
        # Arrange
        outputs_mgr = outputs_mgr_with_plots  # short name is nice

        # Act
        outputs_mgr.build_ranked_target_lists()

        # Assert
        outputs_path = outputs_mgr.path_manager.outputs_path
        assert isinstance(outputs_mgr.science_ranked_list, pd.DataFrame)
        assert set(outputs_mgr.obs_ranked_lists.keys()) == set(["lasilla", "astrolab"])

        # check lists are made
        outputs_path = outputs_mgr.path_manager.outputs_path

        science_exp_path = outputs_path / "ranked_lists/science_score.csv"
        assert science_exp_path.exists()
        lasilla_exp_path = outputs_path / "ranked_lists/obs_lasilla.csv"
        assert lasilla_exp_path.exists()
        astrolab_exp_path = outputs_path / "ranked_lists/obs_astrolab.csv"
        assert astrolab_exp_path.exists()

        T00_science_lc_path = outputs_path / "plots/science_score/001_T00_lc.png"
        assert T00_science_lc_path.exists()
        T00_lasilla_lc_path = outputs_path / "plots/obs_lasilla/001_T00_lc.png"
        assert T00_lasilla_lc_path.exists()
        T01_astr_vis_path = outputs_path / "plots/obs_lasilla/002_T01_lasilla_vis.png"
        assert T01_astr_vis_path.exists()


class Test__VisibleTargets:

    def test__vis_targets_at_obs(
        self, om_vis_targets: OutputsManager, lasilla: Observer, t_fixed: Time
    ):
        # Arrange
        outputs_mgr = om_vis_targets

        # Act
        vis_list = outputs_mgr.create_visible_target_list_for_obs(
            lasilla, t_ref=t_fixed
        )
        print(vis_list)

        # Assert
        assert isinstance(vis_list, pd.DataFrame)
        assert len(vis_list) == 3

        assert vis_list["target_id"].iloc[0] == "Tv03"  # at midnight lst -30.0
        assert vis_list["target_id"].iloc[1] == "Tv00"  # at midnight lst
        assert vis_list["target_id"].iloc[2] == "Tv01"  # at midnight lst +30.0
        # no Tv04 -

    def test__visible_targets(self, om_vis_targets: OutputsManager, t_fixed: Time):
        # Arrange
        outputs_mgr = om_vis_targets

        # Act
        outputs_mgr.create_visible_target_lists(t_ref=t_fixed)

        # Assert
        assert set(outputs_mgr.obs_visible_lists.keys()) == set(["lasilla", "astrolab"])

    def test__no_ephem_warns(
        self, om_vis_targets: OutputsManager, lasilla: Observer, t_fixed: Time
    ):
        # Arrange
        outputs_mgr = om_vis_targets
        for target_id, target in outputs_mgr.target_lookup.items():
            target.ephem_info = {}

        # Act
        with pytest.warns(MissingEphemInfoWarning):
            vis_list = outputs_mgr.create_visible_target_list_for_obs(
                lasilla, t_ref=t_fixed
            )

        # Assert
        assert isinstance(vis_list, pd.DataFrame)
        assert vis_list.empty
        exp_keys = ["target_id", "score", "transit_mjd", "transit_time"]
        assert set(vis_list.columns) == set(exp_keys)
