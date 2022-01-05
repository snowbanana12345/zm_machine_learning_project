import logging
import definitions
import os

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.DEBUG)
file_handler = logging.FileHandler(os.path.join(definitions.LOGS_FOLDER_PATH, "logging_test.log"))
formatter = logging.Formatter("[%(asctime)s] : %(levelname)s : %(name)s : %(message)s ")
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logger.debug("debug_test_successful")
logger.info("info_test_successful")
logger.warning("warning_test_successful")
logger.error("error_test_successful")
logger.critical("error_test_successful")



