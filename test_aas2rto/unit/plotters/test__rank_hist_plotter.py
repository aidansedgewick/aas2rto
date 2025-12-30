import pytest

import matplotlib.pyplot as plt

from astropy import units as u
from astropy.time import Time

from astroplan import Observer

from aas2rto.exc import UnknownObservatoryWarning
from aas2rto.plotting.rank_hist_plotter import RankHistoryPlotter, plot_rank_histories
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

    # Act
    yield  # Test is run here

    # Cleanup
    plt.close("all")  # Code AFTER yield in fixture is cleanup/teardown


class Test__PlotterInit:

    def test__plotter_init(self):
        # Act
        plotter = RankHistoryPlotter()

        # Assert
        assert isinstance(plotter.fig, plt.Figure)
        assert isinstance(plotter.ax, plt.Axes)
        assert isinstance(plotter.targets_plotted, list)
        assert len(plotter.targets_plotted) == 0
        assert isinstance(plotter.targets_skipped, list)
        assert len(plotter.targets_skipped) == 0
        assert not plotter.axes_formatted


class Test__PlotRanksMethod:
    def test__plot_ranks(self, mod_tl: TargetLookup, t_fixed: Time):
        # Arrange
        plotter = RankHistoryPlotter()

        # Act
        t_plotted, t_skipped = plotter.plot_ranks(mod_tl, t_ref=t_fixed)
        assert set(t_plotted) == set(["T00", "T01"])
        assert set(t_skipped) == set()

    def test__plot_if_prev_high_rank(self, mod_tl: TargetLookup, t_fixed: Time):
        # Arrange
        plotter = RankHistoryPlotter(minimum_rank=5)  # T01 RECENTLY had rank<5.

        # Act
        t_plotted, t_skipped = plotter.plot_ranks(mod_tl, t_ref=t_fixed)

        # Assert
        assert set(t_plotted) == set(["T00", "T01"])
        assert set(t_skipped) == set()

    def test__plot_high_rank_recent_only(self, mod_tl: TargetLookup, t_fixed: Time):
        # Arrange
        plotter = RankHistoryPlotter(minimum_rank=5, lookback=1.5)

        # Act
        t_plotted, t_skipped = plotter.plot_ranks(mod_tl, t_ref=t_fixed)

        # Assert
        assert set(t_plotted) == set(["T00"])
        assert set(t_skipped) == set(["T01"])

    def test__no_fail_no_rank_hist(self, tlookup: TargetLookup, t_fixed: Time):
        # Arrange
        assert len(tlookup["T00"].science_rank_history) == 0
        plotter = RankHistoryPlotter()

        # Act
        t_plotted, t_skipped = plotter.plot_ranks(tlookup, t_ref=t_fixed)

        # Assert]
        assert set(t_plotted) == set()
        assert set(t_skipped) == set(["T00", "T01"])

    def test__plot_ranks_at_obs(
        self, mod_tl: TargetLookup, lasilla: Observer, t_fixed: Time
    ):
        # Arrange()
        plotter = RankHistoryPlotter()

        # Act
        with pytest.warns(UnknownObservatoryWarning):
            t_plotted, t_skipped = plotter.plot_ranks(
                mod_tl, observatory=lasilla, t_ref=t_fixed
            )

        # Assert
        assert set(t_plotted) == set(["T00"])
        assert set(t_skipped) == set(["T01"])

    def test__plot_ranks_obs_name(self, mod_tl: TargetLookup, t_fixed: Time):
        # Arrange()
        plotter = RankHistoryPlotter()

        # Act
        with pytest.warns(UnknownObservatoryWarning):
            t_plotted, t_skipped = plotter.plot_ranks(
                mod_tl, observatory="lasilla", t_ref=t_fixed
            )

        # Assert
        assert set(t_plotted) == set(["T00"])
        assert set(t_skipped) == set(["T01"])


class Test__FormatAxesMethod:
    def test__fmt_axes_no_fail(self, mod_tl: TargetLookup, t_fixed: Time):
        # Arrange
        plotter = RankHistoryPlotter()
        plotter.plot_ranks(mod_tl, t_ref=t_fixed)  # avoid dubiousYearWarning

        # Act
        plotter.format_axes()

    def test__fmt_axes_many_days(self, mod_tl: TargetLookup, t_fixed: Time):
        # Arrange
        plotter = RankHistoryPlotter(lookback=23)
        plotter.plot_ranks(mod_tl, t_ref=t_fixed)  # avoid dubiousYearWarning

        # Act
        plotter.format_axes()

    def test__fmt_axes_few_days(self, mod_tl: TargetLookup, t_fixed: Time):
        # Arrange
        plotter = RankHistoryPlotter(lookback=3)
        plotter.plot_ranks(mod_tl, t_ref=t_fixed)  # avoid dubiousYearWarning

        # Act
        plotter.format_axes()


class Test__PlotClassMethod:
    def test__plot_cls_method(self, mod_tl: TargetLookup, t_fixed: Time):
        # Act
        plotter = RankHistoryPlotter.plot(mod_tl, t_ref=t_fixed)

        # Assert
        assert set(plotter.targets_plotted) == set(["T00", "T01"])
        assert plotter.axes_formatted


class Test__PlotFunc:
    def test__rank_hist_func(self, mod_tl: TargetLookup, t_fixed: Time):
        # Act
        fig = plot_rank_histories(mod_tl, t_ref=t_fixed)

        # Assert
        assert isinstance(fig, plt.Figure)
