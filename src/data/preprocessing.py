import logging

import pandas as pd


class Preprocessing:

    def __init__(self):
        """

        """

        logging.basicConfig(level=logging.INFO, format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)

    def exc(self, frame: pd.DataFrame):

        summary = frame['movieId'].value_counts()
        self.logger.info(type(summary))
        self.logger.info(summary)
