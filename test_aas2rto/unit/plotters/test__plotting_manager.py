import pytest

from astropy import units as u
from astropy.time import Time

from astroplan import Observer

import matplotlib.pyplot as plt

from aas2rto.exc import UnexpectedKeysWarning
from aas2rto.observatory_manager import ObservatoryManager
from aas2rto.path_manager import PathManager
from aas2rto.plotting.plotting_manager import PlottingManager
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup


@pytest.fixture
def mod_tl(tlookup: TargetLookup, t_fixed: Time):
    t0 = t_fixed - 2.0 * u.day
    t1 = t_fixed - 1.0 * u.day

    tlookup["T00"].update_science_rank_history(3, t_ref=t0)
    tlookup["T00"].update_science_rank_history(2, t_ref=t1)
    tlookup["T00"].update_science_rank_history(1, t_ref=t_fixed)
    tlookup["T00"].update_obs_rank_history(1, "lasilla", t_ref=t1)
    tlookup["T00"].update_obs_rank_history(2, "lasilla", t_ref=t_fixed)

    tlookup["T01"].update_science_rank_history(4, t_ref=t0)
    tlookup["T01"].update_science_rank_history(6, t_ref=t1)
    tlookup["T01"].update_science_rank_history(8, t_ref=t_fixed)
    return tlookup


@pytest.fixture(autouse=True)
def close_all_plots():
    # Arrange
    pass  # Code BEFORE yield in fixture is setup. No setup here...

    yield  # Test is run here

    # Cleanup
    plt.close("all")  # Code AFTER yield in fixture is cleanup/teardown


# ===== define some test plotters here ===== #


def mock_lc_plot(target: Target, t_ref: Time):
    return plt.Figure(figsize=(1, 1))


def mock_other_plot(target: Target, t_ref: Time):
    return plt.Figure(figsize=(1, 1))


def bad_plotting_func(target: Target, t_ref: Time):
    raise ValueError


# ===== Actual tests start here ===== #


class Test__PlottingMgrInit:
    def test__plotting_mgr_init(
        self, tlookup: Target, path_mgr: PathManager, obs_mgr: ObservatoryManager
    ):
        # Act
        pm = PlottingManager({}, tlookup, path_mgr, obs_mgr)  # noqa: F841

    def test__bad_config_warns(
        self, tlookup: Target, path_mgr: PathManager, obs_mgr: ObservatoryManager
    ):
        # Arrange
        config = {"bad_key": 100.0}

        # Act
        with pytest.warns(UnexpectedKeysWarning):
            pm = PlottingManager(config, tlookup, path_mgr, obs_mgr)  # noqa: F841


class Test__PlotAllLCs:
    def test__plot_all_lcs(self, plotting_mgr: PlottingManager, tmp_path):
        # Arrange
        tl = plotting_mgr.target_lookup
        plot_path = plotting_mgr.path_manager.scratch_path / "lc"

        # Act
        plotted, skipped = plotting_mgr.plot_all_target_lightcurves(mock_lc_plot)

        # Assert
        assert set(plotted) == set(["T00", "T01"])
        assert set(skipped) == set()

        assert tl["T00"].lc_fig_path == plot_path / "T00_lc.png"
        assert tl["T00"].lc_fig_path.exists()

        assert tl["T01"].lc_fig_path == plot_path / "T01_lc.png"
        assert tl["T01"].lc_fig_path.exists()

    def test__skip_non_updated(self, plotting_mgr: PlottingManager, t_fixed: Time):
        # Arrange
        tl = plotting_mgr.target_lookup
        plotting_mgr.plot_all_target_lightcurves(mock_lc_plot)
        tl["T00"].updated = True

        # Act
        plotted, skipped = plotting_mgr.plot_all_target_lightcurves(mock_lc_plot)

        # Assert
        assert set(plotted) == set(["T00"])
        assert set(skipped) == set(["T01"])

    def test__replot_old_lc(self, plotting_mgr: PlottingManager):
        # Arrange
        t_now = Time.now()
        t_later = t_now + 3.0 * u.hour
        t_much_later = t_now + 12.0 * u.hour
        plotting_mgr.plot_all_target_lightcurves(mock_lc_plot, t_ref=t_now)

        # Act
        plotted_1, skipped_1 = plotting_mgr.plot_all_target_lightcurves(
            mock_lc_plot, t_ref=t_later
        )
        plotted_2, skipped_2 = plotting_mgr.plot_all_target_lightcurves(
            mock_lc_plot, t_ref=t_much_later
        )

        # Assert
        assert set(plotted_1) == set()
        assert set(skipped_1) == set(["T00", "T01"])

        assert set(plotted_2) == set(["T00", "T01"])
        assert set(skipped_2) == set()

    def test__no_lazy_replot_all_lc(self, plotting_mgr: PlottingManager, t_fixed: Time):
        # Arrange
        plotting_mgr.config["lazy_plotting"] = False
        plotting_mgr.plot_all_target_lightcurves(mock_lc_plot)

        # Act
        plotted, skipped = plotting_mgr.plot_all_target_lightcurves(mock_lc_plot)

        # Assert
        assert set(plotted) == set(["T00", "T01"])
        assert set(skipped) == set()


class Test__PlotVisAtObsMethod:
    def test__plot_all_at_obs(
        self, plotting_mgr: PlottingManager, lasilla: Observer, t_fixed: Time
    ):
        # Arrange
        tl = plotting_mgr.target_lookup
        vis_plot_path = plotting_mgr.path_manager.scratch_path / "vis"

        # Act
        plotted, skipped = plotting_mgr.plot_all_target_visibilities_for_observatory(
            lasilla, t_ref=t_fixed
        )

        # Assert
        assert set(plotted) == set(["T00", "T01"])
        assert set(skipped) == set()

        assert set(tl["T00"].vis_fig_paths.keys()) == set(["lasilla"])
        exp_T00_path = vis_plot_path / "T00_lasilla_vis.png"
        assert tl["T00"].vis_fig_paths["lasilla"] == exp_T00_path

        assert set(tl["T01"].vis_fig_paths.keys()) == set(["lasilla"])
        exp_T00_path = vis_plot_path / "T01_lasilla_vis.png"
        assert tl["T01"].vis_fig_paths["lasilla"] == exp_T00_path

    def test__lazy_no_replot_vis(
        self, plotting_mgr: PlottingManager, lasilla: Observer
    ):
        # Arrange
        plotting_mgr.plot_all_target_visibilities_for_observatory(lasilla)

        # Act
        plotted, skipped = plotting_mgr.plot_all_target_visibilities_for_observatory(
            lasilla
        )

        # Assert
        assert set(plotted) == set()
        assert set(skipped) == set(["T00", "T01"])

    def test__replot_old_vis(self, plotting_mgr: PlottingManager, lasilla: Observer):
        # Arrange
        t_now = Time.now()
        t_later = t_now + 3.0 * u.hour
        t_much_later = t_now + 12.0 * u.hour
        plotting_mgr.plot_all_target_visibilities_for_observatory(lasilla, t_ref=t_now)

        # Act
        pl_1, sk_1 = plotting_mgr.plot_all_target_visibilities_for_observatory(
            lasilla, t_ref=t_later
        )
        pl_2, sk_2 = plotting_mgr.plot_all_target_visibilities_for_observatory(
            lasilla, t_ref=t_much_later
        )

        # Assert
        assert set(pl_1) == set()
        assert set(sk_1) == set(["T00", "T01"])

        assert set(pl_2) == set(["T00", "T01"])
        assert set(sk_2) == set()

    def test__no_lazy_replot_all_vis(
        self, plotting_mgr: PlottingManager, lasilla: Observer
    ):
        # Arrange
        plotting_mgr.config["lazy_plotting"] = False
        plotting_mgr.plot_all_target_visibilities_for_observatory(lasilla)

        # Act
        plotted, skipped = plotting_mgr.plot_all_target_visibilities_for_observatory(
            lasilla
        )

        # Assert
        assert set(plotted) == set(["T00", "T01"])
        assert set(skipped) == set()


class Test__PlotVisAllMethod:
    def test__plot_all_vis(self, plotting_mgr: PlottingManager):
        # Act
        plotting_mgr.plot_all_target_visibilities()

        # Assert
        plot_path = plotting_mgr.path_manager.scratch_path / "vis"
        T00 = plotting_mgr.target_lookup["T00"]
        T01 = plotting_mgr.target_lookup["T01"]
        assert set(T00.vis_fig_paths.keys()) == set(["astrolab", "lasilla"])
        assert T00.vis_fig_paths["astrolab"] == plot_path / "T00_astrolab_vis.png"
        assert T00.vis_fig_paths["lasilla"] == plot_path / "T00_lasilla_vis.png"
        assert set(T01.vis_fig_paths.keys()) == set(["astrolab", "lasilla"])

    # TODO: add the same tests above... but they are not critical.


class Test__PlotTargetRankHist:
    def test__plot_rank_hist(
        self, plotting_mgr: PlottingManager, mod_tl: TargetLookup, t_fixed: Time
    ):
        # Arrange
        plots_path = plotting_mgr.path_manager.scratch_path
        plotting_mgr.target_lookup = mod_tl

        # Act
        plotted, skipped = plotting_mgr.plot_rank_histories(t_ref=t_fixed)

        # Assert
        assert set(plotted) == set(["T00", "T01"])
        assert set(skipped) == set()

        exp_path = plots_path / "rank_histories.png"
        assert exp_path.exists()


class Test__PlotAddtlTargetFigs:
    def test__mock_other_plot(self, plotting_mgr: PlottingManager, t_fixed: Time):
        # Act
        plotted, skipped = plotting_mgr.plot_additional_target_figures(
            mock_other_plot, t_ref=t_fixed
        )

        # Assert
        plot_path = (
            plotting_mgr.path_manager.scratch_path / "extra_plots/mock_other_plot"
        )
        assert plot_path.is_dir()
        assert set(plotted) == set(["T00", "T01"])
        assert set(skipped) == set()

    def test__mock_other_plot_lazy(self, plotting_mgr: PlottingManager):
        # Arange
        t_now = Time.now()
        t_later = t_now + 1.0 * u.hour
        t_much_later = t_now + 1.0 * u.day
        plotting_mgr.plot_additional_target_figures(mock_other_plot, t_ref=t_now)

        # Act
        plotted_1, skipped_1 = plotting_mgr.plot_additional_target_figures(
            mock_other_plot, t_ref=t_later
        )
        plotted_2, skipped_2 = plotting_mgr.plot_additional_target_figures(
            mock_other_plot, t_ref=t_much_later
        )

        # Assert
        assert set(plotted_1) == set()
        assert set(skipped_1) == set(["T00", "T01"])
        assert set(plotted_2) == set(["T00", "T01"])
        assert set(skipped_2) == set()

    def test__bad_func_no_raise(self, plotting_mgr: PlottingManager, t_fixed: Time):
        # Act
        with pytest.warns(UserWarning):
            plotted, skipped = plotting_mgr.plot_additional_target_figures(
                bad_plotting_func, t_ref=t_fixed
            )
