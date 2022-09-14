from time import asctime
from os import path, mkdir
from logging import StreamHandler, Logger, Formatter, getLogger, basicConfig, DEBUG

from constants.constants import LOGS


class Logr:
    """Class for setting up the logging module"""

    def __init__(self) -> None:
        self.__time_created = None
        self.__log_file = None
        self.__encoding = "utf8"
        self.__date_format_str = "%m-%d-%Y %H:%M:%S"
        self.__config_format_str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        self.__console_formatter = Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%m-%d-%Y %H:%M:%S"
        )

    @property
    def time_created(self) -> str:
        return self.__time_created

    @property
    def log_file(self) -> str:
        return self.__log_file

    @property
    def console_formatter(self) -> str:
        return self.__console_formatter

    @property
    def config_format_str(self) -> str:
        return self.__config_format_str

    @property
    def encoding(self) -> str:
        return self.__encoding

    @property
    def date_format_str(self) -> str:
        return self.__date_format_str

    def create_directory(self, dir_name: str) -> None:
        """Creates new directory if directory name does not exist"""
        if not path.exists(dir_name):
            mkdir(dir_name)

    def __create_console_handler(self, level: int, format_str: str) -> StreamHandler:
        """Returns a StreamHandler dedicated to handling terminal/console output"""
        console_handler = StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(format_str)
        return console_handler

    def __create_logging_handler(self, level: int, logger_name: str) -> Logger:
        """Returns a logging handler dedicated to writing to .log files"""
        logger = getLogger(logger_name)
        logger.setLevel(level)
        return logger

    def setup_logger(self, logger_name: str) -> Logger:
        """Sets up the logging module"""
        self.__time_created = asctime().replace(":", "-")
        self.__log_file = (
            LOGS + "/" + self.__time_created + "/" + self.__time_created + ".log"
        )

        self.create_directory(LOGS)
        self.create_directory(LOGS + "/" + self.__time_created)

        basicConfig(
            filename=self.__log_file,
            encoding=self.__encoding,
            level=DEBUG,
            format=self.__config_format_str,
            datefmt=self.__date_format_str,
        )

        logger = self.__create_logging_handler(DEBUG, logger_name)
        console_handler = self.__create_console_handler(DEBUG, self.console_formatter)

        # add console handler to logger
        logger.addHandler(console_handler)
        return logger
