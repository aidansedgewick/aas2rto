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

    # def update_lc_continuous(
    #     self,
    #     updates: pd.DataFrame,
    #     column="mjd",
    #     keep_updates=False,
    #     **kwargs,
    # ):
    #     """
    #     Parameters
    #     ----------
    #     updates : pd.DataFrame or astropy.table.Table
    #         the updates you want to include in your updated lightcurve.
    #     column

    #     """

    #     if self.lightcurve is None:
    #         self.add_lightcurve(updates, **kwargs)
    #         return updates

    #     if type(self.lightcurve) != type(updates):
    #         msg = (
    #             f"Both existing lightcurve (type={type(self.lightcurve)}) "
    #             f"and updates (type={type(updates)}) should be the same type."
    #         )
    #         raise TypeError(msg)

    #     if keep_updates:
    #         existing_mask = self.lightcurve[column] < updates[column].min()
    #         existing = self.lightcurve[existing_mask]
    #         # updates = updates # Obvious!
    #     else:
    #         existing = self.lightcurve
    #         updates_mask = updates[column] > self.lightcurve[column].max()
    #         updates = updates[updates_mask]

    #     if isinstance(updates, pd.DataFrame):
    #         updated_lc = pd.concat([self.lightcurve, updates], ignore_index=True)
    #     if isinstance(updates, Table):
    #         updated_lc = vstack([self.lightcurve, updates])

    #     self.add_lightcurve(updated_lc, **kwargs)
    #     return updated_lc

    # def update_lc_exact_match(
    #     self,
    #     updates: pd.DataFrame,
    #     column: str,
    #     sort_column=None,
    #     keep_updates=True,
    #     ignore_values=None,
    #     **kwargs,
    # ):
    #     """
    #     Concatenate existing lightcurve and updates, remove duplicated values.
    #     Which duplicated values are removed depends on `keep_updates`.

    #     Parameters
    #     ----------
    #     updates : `pd.DataFrame` or `astropy.table.Table`
    #         the updates to include
    #     column : str, default="candid"
    #         Which column to look for duplicates
    #     ignore_values : list or tuple, optional
    #         If provided, do not discard duplicate rows with these values.
    #         - need this the matching column is often obsID, which is LongInt.
    #             Therefore often need eg. -1 or -99 for 'missing' value.
    #             LongInt will be destroyed (converted to float) if np.nan is
    #             used for missing values.
    #     """

    #     keep = "last" if keep_updates else "first"

    #     if self.lightcurve is None:
    #         self.add_lightcurve(updates, **kwargs)
    #         return updates

    #     if type(self.lightcurve) != type(updates):
    #         msg = (
    #             f"Both existing lightcurve (type={type(self.lightcurve)}) "
    #             f"and updates (type={type(updates)}) should be the same type."
    #         )
    #         raise TypeError(msg)

    #     if isinstance(updates, pd.DataFrame):
    #         if updates is None:
    #             concat_lc = self.lightcurve
    #         else:
    #             concat_lc = pd.concat([self.lightcurve, updates], ignore_index=True)

    #         unique_mask = ~concat_lc.duplicated(column, keep=keep)  # "keep" are False
    #         if ignore_values is not None:
    #             ignore_mask = updated_lc[column].isin(ignore_values)
    #             unique_mask = unique_mask | ignore_mask  # unique or ignorred
    #         null_mask = concat_lc[column].isnull()

    #         print(concat_lc)

    #         updated_lc = concat_lc[unique_mask | null_mask]  # null is fine.
    #         if sort_column is not None:
    #             updated_lc.sort_values(sort_column, ignore_index=True, inplace=True)
    #     elif isinstance(updates, Table):
    #         concat_lc = vstack([self.lightcurve, updates])
    #         if ignore_values is not None:
    #             ignore_mask = np.isin(concat_lc[column], ignore_values)
    #             # separate rows w/ values we don't want to drop, add them back later
    #             ignore_lc = concat_lc[ignore_mask]
    #             # find those we can drop
    #             non_ignore_lc = concat_lc[~ignore_mask]
    #             unique_non_ignore = unique_table(non_ignore_lc, keys=column, keep=keep)
    #             # add them back here
    #             updated_lc = vstack([ignore_lc, unique_non_ignore])
    #         else:
    #             updated_lc = unique_table(concat_lc, keys=column, keep=keep)

    #         print(updated_lc)
    #         if sort_column is not None:
    #             updated_lc.sort(sort_column)

    #     self.add_lightcurve(updated_lc, **kwargs)
    #     return updated_lc
