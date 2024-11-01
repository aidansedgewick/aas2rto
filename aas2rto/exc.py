# Exceptions


class BadKafkaConfigError(Exception):
    pass


class MissingCoordinatesError(Exception):
    pass


class MissingDateError(Exception):
    pass


class MissingObjectIdError(Exception):
    pass


class NotATargetError(Exception):
    pass


# Warnings


class DuplicateDataWarning(UserWarning):
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
