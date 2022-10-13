"""
Module: films - reads & structures a MovieLens data set.
"""

import os
import logging

import pandas as pd

import config


class Films:

    def __init__(self):
        """

        """

        self.data_directory = config.Config().data_directory

        logging.basicConfig(level=logging.INFO, format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)

    def __read(self, pathway: str) -> pd.DataFrame:
        """
        Reads-in data from a CSV file

        :param pathway: The <directory path name> + <file name> + <file extension> string
        :return:
        """

        try:
            frame = pd.read_csv(filepath_or_buffer=os.path.join(self.data_directory, pathway))
        except OSError as err:
            raise Exception(err.strerror) from err

        return frame

    def exc(self):

        ratings = self.__read(pathway='ratings.csv')
        self.logger.info(ratings.info())

        movies = self.__read(pathway='movies.csv')
        self.logger.info(movies.info())

        links = self.__read(pathway='links.csv')
        self.logger.info(links.info())

        print(movies.join(movies.genres.str.get_dummies().astype(bool)).head())
