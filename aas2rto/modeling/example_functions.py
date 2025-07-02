from astropy.time import Time

from aas2rto import Target


class EmptyModel:
    def __init__(self):
        pass


def empty_modeling(target: Target, t_ref: Time = None):
    return EmptyModel()
