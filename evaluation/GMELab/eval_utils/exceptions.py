class ConfigurationError(Exception):
    def __init__(self, message="Misconfiguration exception.") -> None:
        self.message = message
        super().__init__(self.message)


class ConfigurationWarning(Warning):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)
