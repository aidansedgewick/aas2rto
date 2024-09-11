import warnings
from typing import Union, Dict

import numpy as np

import pandas as pd

from astropy.table import Table, vstack
from astropy.table import unique as unique_table

from dk154_targets.exc import (
    MissingDateError,
    SettingLightcurveDirectlyWarning,
    UnknownPhotometryTagWarning,
)


class TargetData:
    default_valid_tags = ("valid", "detection", "det")
    default_badqual_tags = ("badquality", "badqual", "dubious")
    default_nondet_tags = ("upperlim", "nondet")

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
        self.meta = dict()
        meta = meta or {}

        self.valid_tags = tuple(valid_tags)
        self.badqual_tags = tuple(badqual_tags)
        self.nondet_tags = tuple(nondet_tags)

        if lightcurve is not None:
            self.add_lightcurve(lightcurve.copy(), include_badqual=include_badqual)
        else:
            self.remove_lightcurve()
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
        include_badqual=False,
    ):
        lightcurve = lightcurve.copy()

        date_col = self.get_date_column(lightcurve)
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
            self.detections = self.lightcurve[
                np.isin(self.lightcurve[tag_col], detection_tags)
            ]
            self.badqual = self.lightcurve[
                np.isin(self.lightcurve[tag_col], badqual_tags)
            ]
            self.non_detections = self.lightcurve[
                np.isin(self.lightcurve[tag_col], nondet_tags)
            ]

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
            self.detections = self.lightcurve.copy()
            self.badqual = None
            self.non_detections = None
        return

    def remove_lightcurve(self):
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", category=SettingLightcurveDirectlyWarning
            )  # TODO on 3.1
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

    # def integrate_lightcurve_updates(
    #     self,
    #     updates: pd.DataFrame,
    #     column: str = "candid",
    #     continuous=False,
    #     keep_updates=True,
    #     **kwargs,
    # ):
    #     """
    #     combine updates into a lightcurve.

    #     Parameters
    #     ----------
    #     updates : pd.DataFrame or astropy.table.Table
    #         pd.DataFrame, the updates you want to include in your updated lightcurve.
    #     column : str, default="candid"
    #         the column to check for matches, and remove repeated rows
    #     continuous : bool, default=False
    #         If True, remove any rows in existing LC after the first value in updates
    #             eg. if column="mjd", and existing LC has rows mjd=60000, 60010,
    #             and updates has mjd rows = 60005, 60009, 60012:
    #             if keep_updates is True:
    #                 all existing LC rows with mjd>60005 will be discarded
    #             if keep_updates is False:
    #                 all updates rows mjd<60010 will be discarded.

    #         If continuous=False, then rows with exact matching

    #     **kwargs
    #         keyword arguments are passed to add_lightcurve
    #     """
    #     if updates is None:
    #         logger.warning("updates is None")
    #         return None

    #     print(updates)

    #     if self.lightcurve is None:
    #         self.add_lightcurve(updates, **kwargs)
    #     if column not in self.lightcurve.columns:
    #         raise ValueError(f"{column} not in both lightcurve columns")
    #     if column not in updates.columns:
    #         raise ValueError(f"{column} not in both updates columns")
    #     if continuous:
    #         updated_lightcurve = self.integrate_continuous(
    #             updates, column, keep_updates=keep_updates
    #         )
    #     else:
    #         updated_lightcurve = self.integrate_equality(
    #             updates, column, keep_updates=keep_updates
    #         )

    #     print(updated_lightcurve)
    #     self.add_lightcurve(updated_lightcurve, **kwargs)

    def integrate_lightcurve_updates_continuous(
        self, updates: pd.DataFrame, column="mjd", keep_updates=True, ignore_values=None
    ):
        """
        Parameters
        ----------
        updates : pd.DataFrame or astropy.table.Table
            the updates you want to include in your updated lightcurve.
        column

        """

        if not (isinstance(updates, pd.DataFrame) or isinstance(updates, Table)):
            raise TypeError(
                f"updates should be `pd.DataFrame` or `astropy.table.Table, "
                f"not {type(updates)}"
            )

        if keep_updates:
            existing_mask = self.lightcurve[column] < updates[column].min()
            existing = self.lightcurve[
                existing_mask
            ]  # Keep everything before updates start.
            # updates = updates # Obvious!
        else:
            existing = self.lightcurve
            updates_mask = updates[column] > self.lightcurve[column].max()
            updates = updates[updates_mask]

        updated_lightcurve = None
        if isinstance(updates, pd.DataFrame):
            return pd.concat([self.lightcurve, updates], ignore_index=True)
        if isinstance(updates, Table):
            return vstack([self.lightcurve, updates])
        raise ValueError("should not have made it here!")

    def integrate_lightcurve_updates_equality(
        self,
        updates: pd.DataFrame,
        column="candid",
        keep_updates=True,
        ignore_values=None,
        **kwargs,
    ):
        """
        Concatenate existing lightcurve and updates, remove duplicated values.
        Which duplicated values are removed depends on `keep_updates`.

        Parameters
        ----------
        updates : `pd.DataFrame` or `astropy.table.Table`
            the updates to include
        column : str, default="candid"
            Which column to look for duplicates
        ignore_values : list or tuple, optional
            If provided, do not discard duplicate rows with these values.


        Example
        -------
        """

        keep = "last" if keep_updates else "first"
        updated_lightcurve = None

        if self.lightcurve is not None and type(self.lightcurve) != type(updates):
            msg = (
                f"Both existing lightcurve (type={type(self.lightcurve)}) "
                f"and updates (type={type(updates)}) should be the same type."
            )
            raise TypeError(msg)

        if isinstance(updates, pd.DataFrame):
            if updates is None:
                updated_lightcurve = self.lightcurve
            else:
                updated_lightcurve = pd.concat(
                    [self.lightcurve, updates], ignore_index=True
                )

            unique_rows_mask = ~updated_lightcurve.duplicated(column, keep=keep)
            if ignore_values is not None:
                relevant_mask = ~updated_lightcurve[column].isin(ignore_values)
            else:
                relevant_mask = np.full(len(updated_lightcurve), True)
            updated_lightcurve = updated_lightcurve[unique_rows_mask | ~relevant_mask]

        if isinstance(updates, Table):
            concat_lightcurve = vstack([self.lightcurve, updates])
            updated_lightcurve = unique_table(concat_lightcurve, keys=column, keep=keep)
        if updated_lightcurve is None:
            raise TypeError(
                f"updates should be `pd.DataFrame` or `astropy.table.Table, not {type(updates)}"
            )
        self.add_lightcurve(updated_lightcurve, **kwargs)
        return updated_lightcurve
