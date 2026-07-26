from logging import getLogger

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from astropy.time import Time

from astroplan import Observer

from aas2rto import utils
from aas2rto.target import Target
from aas2rto.target_lookup import TargetLookup

logger = getLogger(__name__.split(".")[-1])

matplotlib.use("Agg")


def plot_rank_histories(
    target_lookup: TargetLookup,
    observatory: Observer = None,
    t_ref: Time = None,
    return_plotter=False,
    **kwargs,
):
    plotter = RankHistoryPlotter.plot(
        target_lookup, observatory=observatory, t_ref=t_ref, **kwargs
    )
    if return_plotter:
        return plotter
    return plotter.fig


class RankHistoryPlotter:

    @classmethod
    def plot(
        cls,
        target_lookup: TargetLookup,
        observatory: Observer = None,
        t_ref: Time = None,
        **kwargs,
    ):
        plotter = cls(**kwargs)
        plotter.plot_ranks(target_lookup, observatory=observatory, t_ref=t_ref)
        plotter.format_axes(t_ref=t_ref)
        return plotter

    def __init__(self, minimum_rank: int = 16, lookback: float = 4.0):
        self.minimum_rank = minimum_rank
        self.lookback = lookback

        self.init_fig()

        self.targets_plotted = []  # mainly for testing...
        self.targets_skipped = []
        self.axes_formatted = False

    def init_fig(self):
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(right=0.8)

    def plot_ranks(
        self,
        target_lookup: TargetLookup,
        observatory: Observer = None,
        t_ref: Time = None,
        **pl_kwargs,
    ):
        t_ref = t_ref or Time.now()

        ls_list = ["-", "--", ":"]
        color_list = [f"C{cc}" for cc in [0, 1, 2, 3, 4, 6, 8, 9]]  # No 5 or 7

        handles = []
        ranks_plotted = []

        for ii, (target_id, target) in enumerate(target_lookup.items()):
            if observatory is None:
                rank_history = target.get_science_rank_history(t_ref=t_ref)
            else:
                rank_history = target.get_obs_rank_history(observatory, t_ref=t_ref)

            if len(rank_history) == 0:
                self.targets_skipped.append(target_id)
                continue

            recent_mask = t_ref.mjd - rank_history["mjd"] <= self.lookback
            recent_history = rank_history[recent_mask]

            if all(recent_history["ranking"].values > self.minimum_rank):
                self.targets_skipped.append(target_id)
                continue

            last_rank = int(recent_history["ranking"].iloc[-1])  # DEFINITELY int.
            last_mjd = recent_history["mjd"].iloc[-1]

            rank_label = f"{last_rank}: {target_id}"

            rank_idx = last_rank - 1
            if last_rank > self.minimum_rank:
                color = "grey"
                lw = 1
                ls = "-"
            else:
                color_idx = rank_idx % len(color_list)  # eg. 8->0, 13->5
                color = color_list[color_idx]
                ls_idx = min(rank_idx // 8, 2)  # can't go higher than 2...!
                ls = ls_list[ls_idx]
                lw = 2
            plotting_kwargs = dict(label=rank_label, color=color, ls=ls, lw=lw)
            plotting_kwargs.update(**pl_kwargs)

            zorder = max(self.minimum_rank + 10 - rank_idx, 1)
            adjustment = np.random.uniform(-0.1, 0.1, 1)
            xdat = np.concat([recent_history["mjd"].values, [t_ref.mjd]])  # points NOW
            ydat = np.concat([recent_history["ranking"].values, [last_rank]])

            lines = self.ax.step(
                xdat,
                ydat + adjustment,  # small offset so no overlap
                where="post",
                zorder=zorder,
                **plotting_kwargs,
            )
            handles.append(lines[0])
            ranks_plotted.append(last_rank - 1)  # from rank to idx
            self.targets_plotted.append(target_id)

            text_kwargs = dict(ha="left", va="center")
            if last_rank <= self.minimum_rank:
                text_x = t_ref.mjd + 0.1
                self.ax.text(text_x, last_rank, rank_label, color=color, **text_kwargs)

        N_plotted = len(self.targets_plotted)
        N_skipped = len(self.targets_skipped)
        logger.info(f"plotted {N_plotted}, skipped {N_skipped}")

        order = np.argsort(ranks_plotted)
        ordered_handles = [handles[idx] for idx in order]
        # self.ax.legend(handles=ordered_handles)
        return self.targets_plotted, self.targets_skipped

    def format_axes(self, t_ref: Time = None):
        t_ref = t_ref or Time.now()

        self.ax.set_ylim(self.minimum_rank + 0.5, 0.5)
        yticks = np.arange(1, self.minimum_rank + 1, 1)
        if len(yticks) > 20:
            yticks = yticks[::2]
        yticklabels = [f"{int(tick)}" for tick in yticks]
        self.ax.set_yticks(yticks, labels=yticklabels)

        # xmin, xmax = self.ax.get_xlim()
        # xscale = max(5.0, self.lookback)
        # self.ax.set_xlim(xmax - xscale, xmax + 0.1)

        self.ax.set_xlim(t_ref.mjd - self.lookback, t_ref.mjd)

        self.ax.set_ylabel("Rank")
        # self.ax.set_xlabel("Time")
        try:
            self.set_readable_xticks()
        except Exception as e:
            logger.error("no xticks format")

        self.axes_formatted = True

    def set_readable_xticks(self):
        xmin, xmax = self.ax.get_xlim()

        # decide how many ticks there should be
        ticks = np.arange(xmin // 1 + 1, xmax // 1 + 1, 1.0)
        if len(ticks) < 4:
            ticks = np.arange(xmin // 1 + 1, xmax // 1 + 1, 0.5)
        if len(ticks) > 12:
            spacing = len(ticks) // 6
            ticks = ticks[::spacing]

        t_grid = [Time(tick, format="mjd") for tick in ticks]
        print(ticks)

        labels = []
        for ii, t in enumerate(t_grid):
            ymdhms = t.ymdhms

            label = t.strftime("%d")
            if ii == 0:
                label = t.strftime("%d\n%b")
            if int(ymdhms.day) == 0:
                label = t.strftime("%d\n%b")

            if ymdhms.hour != 0:
                label = t.strftime("%d\n%H:%M")

            labels.append(label)

        self.ax.set_xticks(ticks, labels=labels)
