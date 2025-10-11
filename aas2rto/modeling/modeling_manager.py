from logging import getLogger
from multiprocessing import Pool
from typing import Callable, List

from astropy.time import Time

from aas2rto import utils
from aas2rto.path_manager import PathManager
from aas2rto.target_lookup import TargetLookup

logger = getLogger(__name__.split(".")[-1])


class ModelingResult:
    def __init__(self, target_id, model, success, reason):
        self.target_id = target_id
        self.model = model
        self.success = success
        self.reason = reason


def modeling_wrapper(func, target, t_ref=None):
    try:
        model = func(target, t_ref=t_ref)
        success = True
        reason = "success"
    except Exception as e:
        model = None
        success = False
        reason = e
    return ModelingResult(target.target_id, model, success, reason)


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

        self.target_lookup = target_lookup
        self.path_manager = path_manager

    def build_target_models(
        self, modeling_functions: List[Callable], t_ref: Time = None
    ):

        lazy = self.config["lazy_modeling"]
        ncpu = self.config["ncpu"]

        if not isinstance(modeling_functions, list):
            modeling_functions = [modeling_functions]

        for modeling_func in modeling_functions:

            try:
                model_key = modeling_func.__name__
            except AttributeError as e:
                model_key = type(modeling_func).__name__
                msg = f"\n    Your modeling_function {model_key} should have attribute __name__."
                logger.warning(msg)

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
                for target in targets_to_model:
                    result = modeling_wrapper(modeling_func, target, t_ref=t_ref)
                    result_list.append(result)
            else:
                if not isinstance(ncpu, int):
                    raise ValueError(f"'ncpu' must be integer, not {type(ncpu)}")
                logger.info(f"build with {ncpu} Pool workers")
                with Pool(ncpu) as p:
                    args_kwargs = []
                    for target in targets_to_model:
                        # make a 2-tuple: args (n-tuple) and kwargs,
                        # so we can pass this data as a single arg to Pool.map.
                        dat = ((modeling_func, target), dict(t_ref=t_ref))
                        args_kwargs.append(dat)
                    result_list = p.map(pool_modeling_wrapper, args_kwargs)

            built = []
            failed = []
            no_model = []
            fail_str = ""
            for result in result_list:
                target_id = result.target_id
                target = self.target_lookup.get(target_id)
                target.models[model_key] = result.model
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
