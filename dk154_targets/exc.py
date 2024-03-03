class BadKafkaConfigError(Exception):
    pass


class MissingObjectIdError(Exception):
    pass


class MissingCoordinatesError(Exception):
    pass


class MissingDateError(Exception):
    pass


class UnknownObservatoryWarning(UserWarning):
    pass


class UnexpectedKeysWarning(UserWarning):
    pass


class MissingKeysWarning(UserWarning):
    pass
