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

    def __movies(self) -> pd.DataFrame:

        # the data
        movies = self.__read(pathway='movies.csv')
        self.logger.info(movies.info())

        # one-hot-encoding of the <genres> field - each element of the field becomes a column
        movies = movies.join(movies.genres.str.get_dummies().astype(bool))
        movies.drop(columns='genres', inplace=True)
        self.logger.info(movies.info())

        return movies

    def __ratings(self):

        # the data
        ratings = self.__read(pathway='ratings.csv')
        self.logger.info(ratings.info())

        return ratings

    def exc(self):

        movies = self.__movies()
        ratings = self.__ratings()
        frame = ratings.merge(movies, on='movieId', how='left')

        return frame
