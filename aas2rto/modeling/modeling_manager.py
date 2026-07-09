from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import partial
from logging import getLogger
from multiprocessing import Pool
from typing import Any, Callable

import tqdm

from astropy.time import Time

from aas2rto import utils
from aas2rto.path_manager import PathManager
from aas2rto.target_lookup import TargetLookup

logger = getLogger(__name__.split(".")[-1])


@dataclass
class ModelingResult:
    target_id: str
    model: Any
    success: bool
    reason: str
    comments: list[str]


def modeling_wrapper(func, target, t_ref=None):
    try:
        result = func(target, t_ref=t_ref)
        if isinstance(result, tuple) and len(result) == 2:
            model, comments = result
        else:
            model = result
            comments = ["no comments"]
        success = True
        reason = "success"
    except Exception as e:
        model = None
        comments = [f"Model failed"]
        success = False
        reason = e
    return ModelingResult(target.target_id, model, success, reason, comments)


def pool_modeling_wrapper(args_kwargs):
    args, kwargs = args_kwargs
    return modeling_wrapper(*args, **kwargs)


class ModelingManager:

    default_config = {
        "lazy_modeling": True,
        "ncpu": None,
    }

    def __init__(
        self,
        modeling_config: dict,
        target_lookup: TargetLookup,
        path_manager: PathManager,
    ):
        self.config = self.default_config.copy()
        self.config.update(modeling_config)
        utils.check_unexpected_config_keys(
            self.config,
            expected=self.default_config.keys(),
            name="modeling_manager",
        )

        ncpu = self.config["ncpu"]
        if ncpu is not None:
            if not isinstance(ncpu, int):
                raise ValueError(f"'ncpu' must be integer, not {type(ncpu)}")

        self.target_lookup = target_lookup
        self.path_manager = path_manager

    # def recover_models_from_file(self, )

    def build_target_models(
        self, modeling_functions: list[Callable], t_ref: Time = None
    ):

        lazy = self.config["lazy_modeling"]
        ncpu = self.config["ncpu"]

        if not isinstance(modeling_functions, list):
            modeling_functions = [modeling_functions]

        for modeling_func in modeling_functions:

            if modeling_func is None:
                continue

            try:
                model_key = modeling_func.__name__
            except AttributeError as e:
                model_key = type(modeling_func).__name__
                msg = f"\n    Your modeling_function {model_key} should have attribute __name__."
                logger.warning(msg)
                warnings.warn(UserWarning(msg))

            skipped = []
            targets_to_model = []
            for target_id, target in self.target_lookup.items():
                model_exists = model_key in target.models
                if (not target.updated) and lazy and model_exists:
                    skipped.append(target_id)
                    continue
                targets_to_model.append(target)

            if lazy and len(skipped) > 0:
                logger.info(
                    f"skip {len(skipped)} non-updated targets (lazy_modeling=True)"
                )

            if len(targets_to_model) == 0:
                logger.info("no targets to model - continue!")
                continue  # NOT return - we are still in a loop...

            logger.info(f"build {model_key} models for {len(targets_to_model)}")
            ncpu = self.config.get("ncpu", None)
            serial = True  # ncpu is not None
            # if ncpu is not None:
            #    serial = False
            #    logger.info("cannot currently multiprocess models")
            if ncpu is None:  # serial:  #
                logger.info("build in serial")
                result_list = []
                for target in tqdm.tqdm(targets_to_model, total=len(targets_to_model)):
                    result = modeling_wrapper(modeling_func, target, t_ref=t_ref)
                    result_list.append(result)
            else:
                logger.info(f"build with {ncpu} Pool workers")
                with Pool(ncpu) as pool:
                    frozen_args = partial(modeling_wrapper, modeling_func, t_ref=t_ref)
                    iterator = pool.imap_unordered(frozen_args, targets_to_model)

                    result_list: list[ModelingResult] = []
                    for result in tqdm.tqdm(iterator, total=len(targets_to_model)):
                        result_list.append(result)

            built = []
            failed = []
            no_model = []
            fail_str = ""
            for result in result_list:
                target_id = result.target_id
                target = self.target_lookup.get(target_id)
                target.models[model_key] = result.model
                target.models_t_ref[model_key] = t_ref.mjd
                target.model_comments[model_key] = result.comments
                if result.success:
                    if result.model is None:
                        no_model.append(target_id)
                    else:
                        built.append(target_id)
                else:
                    failed.append(target_id)
                    fail_str = fail_str + f"{target_id}: {result.reason}\n"

            logger.info(f"{model_key} built:{len(built)}, {len(no_model)} 'None'")
            if len(failed) > 0:
                logger.warning(f"failed: {len(failed)}, reasons:\n{fail_str}")
