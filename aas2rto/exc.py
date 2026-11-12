###==================== Exceptions ====================###

# === Base exceptions


class ConfigError(Exception):
    """Inherits from Exception.
    All config-related Exceptions should inherit from this class"""

    pass


class TargetError(Exception):
    pass


# === Config exceptions


class BadKafkaConfigError(ConfigError):
    pass


class MissingKeysError(ConfigError):
    pass


class MissingRequiredConfigKeyError(ConfigError):
    pass


class UnexpectedKeysError(ConfigError):
    pass


# === Target exceptions


class MissingCoordinatesError(TargetError):
    pass


class MissingDateError(TargetError):
    pass


class MissingTargetIdError(TargetError):
    pass


class NotATargetError(TargetError):
    pass


# === Misc exceptions


class UnknownQueryManagerError(Exception):
    pass


### ==================== Warnings ==================== ###

# === Base warnings


class ConfigWarning(UserWarning):
    pass


class TargetWarning(UserWarning):
    pass


# === Config warnings


class MissingKeysWarning(ConfigWarning):
    pass


class UnexpectedKeysWarning(ConfigWarning):
    pass


# === Target/data warnings


class DuplicateDataWarning(TargetWarning):
    pass


class InvalidEphemWarning(TargetWarning):
    pass


class MissingEphemInfoWarning(TargetWarning):
    pass


class MissingCompiledLightcurveWarning(TargetWarning):
    pass


class SettingLightcurveDirectlyWarning(TargetWarning):
    pass


class UnknownPhotometryTagWarning(TargetWarning):
    pass


class UnknownTargetWarning(TargetWarning):
    pass


# === Misc warnings


class MissingColumnWarning(UserWarning):
    pass


class MissingFileWarning(UserWarning):
    pass


class MissingMediaWarning(UserWarning):
    pass


class UnknownObservatoryWarning(UserWarning):
    pass
