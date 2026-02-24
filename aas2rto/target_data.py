from __future__ import annotations

import copy
import warnings
from logging import getLogger
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


logger = getLogger(__name__.split(".")[-1])


class TargetData:
    default_valid_tags = ("valid", "detection", "det")
    default_badqual_tags = ("badquality", "badqual", "dubious")
    default_nondet_tags = ("nondet", "upperlim", "ulimit", "ulim")

    date_columns = ("jd", "mjd", "JD", "MJD", "midpointMjdTai")

    setting_lc_msg = (
        "\nYou should use the `target_data.add_lightcurve(lc, tag_col=<tag>)` method."
        "\nIf the column `tag` is avalable, this will correctly set  extra attributes:"
        "    `target_data.detections`, `target_data.badqual` and `target_data.non_detections`"
        "\nYou can choose to include badqual in `detections` with `include_badqual=True`"
    )

    def __init__(
        self,
        lightcurve: pd.DataFrame | Table = None,
        include_badqual: bool = False,
        # probabilities: pd.DataFrame | Table = None,
        parameters: dict = None,
        cutouts: dict = None,
        meta: dict = None,
        tag_col: str = "tag",
        date_col: str = None,
        valid_tags: tuple = default_valid_tags,
        badqual_tags: tuple = default_badqual_tags,
        nondet_tags: tuple = default_nondet_tags,
    ):
        """
        Parameters
        ----------
        lightcurve : pd.DataFrame | astropy.table.Table, optional
        include_badqual : bool, default = False
            Choose whether or not `lightcurve` rows which are tagged (`tag`) as
            'badqual' are included in `detections` or `badqual`
        parameters : dict, optional
        cutouts : dict[str, np.array], optional
        meta : dict, optional
        tag_col : str, default = "tag"
            The lightcurve column to use to decide if rows should be included in
            `detections`, `badqual` or `non_detections` when calling `add_lightcurve()`
            If not present, these extra attributes are set to `None`
        date_col : str, default = <>
            choose first available of 'jd' 'mjd' 'JD' 'MJD' 'midpointMjdTai'.
            Only used for sorting...
        valid_tags : tuple[str], default ("valid", "detection", "det")
            all `lightcurve` rows with `tag_col` values in this tuple are included in
            attr `detections`
        badqual_tags : tuple[str], default ("badquality", "badqual", "dubious")
            all `lightcurve` rows with `tag_col` values in this tuple are included in
            attr `badqual` (if `include_badqual=False`, see above...)
        nondet_tags : tuple[str], default = ("nondet", "upperlim", "ulimit", "ulim")
            all `lightcurve` rows with `tag_col` values in this tuple are included in
            attr `non_detections`
        """

        if isinstance(valid_tags, str):
            valid_tags = [valid_tags]
        if isinstance(badqual_tags, str):
            badqual_tags = [badqual_tags]
        if isinstance(nondet_tags, str):
            nondet_tags = [nondet_tags]
        self.valid_tags = tuple(valid_tags)
        self.badqual_tags = tuple(badqual_tags)
        self.nondet_tags = tuple(nondet_tags)

        self.tag_col = tag_col
        self.date_col = date_col

        if lightcurve is not None:
            self.add_lightcurve(
                copy.deepcopy(lightcurve), include_badqual=include_badqual
            )
        else:
            self.remove_lightcurve()  # set everything to None

        self.parameters = parameters or {}
        self.meta = meta or {}
        # self.probabilities = probabilities or {}

        self.cutouts: dict[str, np.ndarray] = cutouts or {}

    def __setattr__(self, name, value):
        """If user does `tdata.lightcurve = <data>`, warn them they should use
        `tdata.add_lightcurve(<data>)`, as this will make sure
        `tdata.detections`, `tdata.badqual` and `tdata.non_detections` are also set.

        Otherwise, behave as expected."""

        if name == "lightcurve":
            warnings.warn(SettingLightcurveDirectlyWarning(self.setting_lc_msg))
        super().__setattr__(name, value)

    def add_lightcurve(
        self,
        lightcurve: Union[pd.DataFrame, Table],
        tag_col: str = None,
        date_col: str = None,
        include_badqual: bool = False,
    ):
        """
        Set the attribute `lightcurve`, and if `tag_col` is present, also add
        attributes `detections`, `badqual` and `non_detections`.

        Does NOT mark a target as updated.
        Parameters
        ----------
        lightcurve : `pd.DataFrame` | `astropy.table.Table`

        """

        lightcurve = lightcurve.copy()

        tag_col = tag_col or self.tag_col
        date_col = date_col or self.date_col or self.get_date_column(lightcurve)
        if date_col is not None:
            if isinstance(lightcurve, pd.DataFrame):
                try:
                    lightcurve.sort_values(date_col, inplace=True, ignore_index=True)
                except Exception as e:
                    print(lightcurve[date_col])
                    raise
            if isinstance(lightcurve, Table):
                lightcurve.sort(date_col)
        else:
            pass  # Maybe add warning that data is unsorted?

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=SettingLightcurveDirectlyWarning)
            # No need to be warn here - we are correctly setting the attributes!
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

            # Get any data that's referred to as
            det_mask = np.isin(self.lightcurve[tag_col], detection_tags)
            self.detections = self.lightcurve[det_mask]

            badqual_mask = np.isin(self.lightcurve[tag_col], badqual_tags)
            self.badqual = self.lightcurve[badqual_mask]

            nondet_mask = np.isin(self.lightcurve[tag_col], nondet_tags)
            self.non_detections = self.lightcurve[nondet_mask]

            known_tag_mask = np.isin(lightcurve[tag_col], all_tags)
            if not all(known_tag_mask):
                unknown_data = self.lightcurve[~known_tag_mask]
                logger.info(f"unknown data!\n{unknown_data}")
                unknown_tags = np.unique(unknown_data[tag_col])
                msg = (
                    f"\nin {tag_col}: \033[32;1m{unknown_tags}\033[0m\nexpected:\n"
                    f"    valid: {self.valid_tags}\n"
                    f"    badqual: {self.badqual_tags}\n"
                    f"    non_detections: {self.nondet_tags}\n"
                )
                warnings.warn(UnknownPhotometryTagWarning(msg))
        else:
            self.detections = None  # self.lightcurve.copy()
            self.badqual = None
            self.non_detections = None
        return

    def remove_lightcurve(self):
        """
        Set all attributes `lightcurve`, `detections`, `badqual`, `nondetections`
        to `None`.
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=SettingLightcurveDirectlyWarning)
            # Here we're allowed
            self.lightcurve = None
        self.detections = None
        self.badqual = None
        self.non_detections = None

    def get_date_column(self, lightcurve: pd.DataFrame | Table):
        date_col = None
        for col in self.date_columns:
            if col in lightcurve.columns:
                date_col = col
                return date_col
        return None
        # Don't raise error, just don't sort the data...
