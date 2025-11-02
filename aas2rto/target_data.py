import warnings
from typing import Union, Dict

import numpy as np

import pandas as pd

from astropy.table import Table, vstack
from astropy.table import unique as unique_table

from aas2rto.exc import (
    MissingDateError,
    SettingLightcurveDirectlyWarning,
    UnknownPhotometryTagWarning,
)


class TargetData:
    default_valid_tags = ("valid", "detection", "det")
    default_badqual_tags = ("badquality", "badqual", "dubious")
    default_nondet_tags = ("nondet", "upperlim", "ulimit", "ulim")

    date_columns = ("jd", "mjd", "JD", "MJD")

    def __init__(
        self,
        lightcurve: pd.DataFrame = None,
        include_badqual=False,
        probabilities: pd.DataFrame = None,
        parameters: dict = None,
        cutouts: dict = None,
        meta: dict = None,
        valid_tags: tuple = default_valid_tags,
        badqual_tags: tuple = default_badqual_tags,
        nondet_tags: tuple = default_nondet_tags,
    ):
        if isinstance(valid_tags, str):
            valid_tags = [valid_tags]
        if isinstance(badqual_tags, str):
            badqual_tags = [badqual_tags]
        if isinstance(nondet_tags, str):
            nondet_tags = [nondet_tags]
        self.valid_tags = tuple(valid_tags)
        self.badqual_tags = tuple(badqual_tags)
        self.nondet_tags = tuple(nondet_tags)

        if lightcurve is not None:
            self.add_lightcurve(lightcurve.copy(), include_badqual=include_badqual)
        else:
            self.remove_lightcurve()  # set everything to None

        self.meta = meta or {}
        self.probabilities = probabilities or {}
        self.parameters = parameters or {}

        self.cutouts = self.empty_cutouts()
        cutouts = cutouts or {}
        self.cutouts.update(cutouts)

    def __setattr__(self, name, value):
        if name == "lightcurve":
            msg = (
                "\nYou should use the `target_data.add_lightcurve(lc, tag_col=<tag>)` method."
                "\nIf the column `tag` is avalable, this will correctly set the attributes:"
                "    `target_data.detections`, `target_data.badqual` and `target_data.non_detections`"
                "\nYou can choose to include badqual "
            )
            warnings.warn(SettingLightcurveDirectlyWarning(msg))
        super().__setattr__(name, value)

    def add_lightcurve(
        self,
        lightcurve: Union[pd.DataFrame, Table],
        tag_col="tag",
        date_col=None,
        include_badqual=False,
    ):
        """Does NOT mark a target as updated."""

        lightcurve = lightcurve.copy()

        date_col = date_col or self.get_date_column(lightcurve)
        if isinstance(lightcurve, pd.DataFrame):
            lightcurve.sort_values(date_col, inplace=True)
        if isinstance(lightcurve, Table):
            lightcurve.sort(date_col)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=SettingLightcurveDirectlyWarning)
            # We don't need to be warned here - we are correctly setting the attributes!
            self.lightcurve = lightcurve

        if include_badqual:
            detection_tags = self.valid_tags + self.badqual_tags
            badqual_tags = ()
        else:
            detection_tags = self.valid_tags
            badqual_tags = self.badqual_tags
        nondet_tags = self.nondet_tags
        all_tags = detection_tags + badqual_tags + nondet_tags
        if tag_col in self.lightcurve.columns:
            det_mask = np.isin(self.lightcurve[tag_col], detection_tags)
            self.detections = self.lightcurve[det_mask]

            badqual_mask = np.isin(self.lightcurve[tag_col], badqual_tags)
            self.badqual = self.lightcurve[badqual_mask]

            nondet_mask = np.isin(self.lightcurve[tag_col], nondet_tags)
            self.non_detections = self.lightcurve[nondet_mask]

            known_tag_mask = np.isin(lightcurve[tag_col], all_tags)
            if not all(known_tag_mask):
                unknown_tags = self.lightcurve[tag_col][~known_tag_mask]
                msg = (
                    f"\nin {tag_col}: {unknown_tags}\nexpected:"
                    f"    valid: {self.valid_tags}"
                    f"    badqual: {self.badqual_tags}"
                    f"    nondet: {self.nondet_tags}"
                )
                warnings.warn(UnknownPhotometryTagWarning(msg))
        else:
            self.detections = None  # self.lightcurve.copy()
            self.badqual = None
            self.non_detections = None
        return

    def remove_lightcurve(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=SettingLightcurveDirectlyWarning)
            # Here we're allowed
            self.lightcurve = None
        self.detections = None
        self.badqual = None
        self.non_detections = None

    def get_date_column(self, lightcurve):
        date_col = None
        for col in self.date_columns:
            if col in lightcurve.columns:
                date_col = col
                return date_col
        msg = (
            f"lightcurve columns {lightcurve.columns} should contain "
            f"a date column: {self.date_columns}"
        )
        raise MissingDateError(msg)

    def empty_cutouts(self) -> Dict[str, np.ndarray]:
        return {}
