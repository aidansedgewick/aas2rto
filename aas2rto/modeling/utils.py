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


def pool_modeling_wrapper(func, target, t_ref=None):
    raise NotImplementedError
