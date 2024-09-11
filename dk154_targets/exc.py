class BadKafkaConfigError(Exception):
    pass


class MissingObjectIdError(Exception):
    pass


class MissingCoordinatesError(Exception):
    pass


class MissingDateError(Exception):
    pass


class NotATargetError(Exception):
    pass


class UnknownObservatoryWarning(UserWarning):
    pass


class UnexpectedKeysWarning(UserWarning):
    pass


class MissingKeysWarning(UserWarning):
    pass


class MissingMediaWarning(UserWarning):
    pass


class SettingLightcurveDirectlyWarning(UserWarning):
    pass


class UnknownPhotometryTagWarning(UserWarning):
    pass
