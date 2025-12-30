# Exceptions


class MissingCoordinatesError(Exception):
    pass


class MissingDateError(Exception):
    pass


class MissingKeysError(Exception):
    pass


class MissingRequiredConfigKeyError(Exception):
    pass


class MissingTargetIdError(Exception):
    pass


class NotATargetError(Exception):
    pass


class UnexpectedKeysError(Exception):
    pass


# Warnings


class DuplicateDataWarning(UserWarning):
    pass


class InvalidEphemWarning(UserWarning):
    pass


class MissingColumnWarning(UserWarning):
    pass


class MissingEphemInfoWarning(UserWarning):
    pass


class MissingFileWarning(UserWarning):
    pass


class MissingKeysWarning(UserWarning):
    pass


class MissingMediaWarning(UserWarning):
    pass


class SettingLightcurveDirectlyWarning(UserWarning):
    pass


class UnexpectedKeysWarning(UserWarning):
    pass


class UnknownObservatoryWarning(UserWarning):
    pass


class UnknownPhotometryTagWarning(UserWarning):
    pass


class UnknownTargetWarning(UserWarning):
    pass
