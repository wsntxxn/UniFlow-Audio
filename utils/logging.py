from pathlib import Path
from dataclasses import dataclass
import logging


@dataclass
class LoggingLogger:

    filename: str | Path
    level: str = "INFO"

    def create_instance(self, ):
        filename = self.filename.__str__()
        formatter = logging.Formatter("[%(asctime)s] - %(message)s")

        logger = logging.getLogger(__name__ + "." + filename)
        logger.setLevel(getattr(logging, self.level))

        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger
