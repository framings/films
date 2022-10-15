import logging

import pandas as pd
import numpy as np


class Preprocessing:

    def __init__(self):
        """

        """

        logging.basicConfig(level=logging.ERROR, format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.WARNING)

    def __frequency(self, data: pd.DataFrame):

        frequency = data['movieId'].value_counts().rename('frequency').to_frame()
        frequency.reset_index(drop=False, inplace=True)
        frequency.rename(columns={'index': 'movieId'}, inplace=True)
        self.logger.info(frequency)

        return frequency

    def __reduce(self, data: pd.DataFrame, frequency: pd.DataFrame, limit: int):

        # focus on films that have at least <limit> number of ratings
        frequency = frequency.copy().loc[frequency['frequency'] >= limit, :]
        self.logger.info(f'The number of films that have at least {limit} ratings: {frequency.shape}')

        # hence
        reduced = data.copy().loc[data['movieId'].isin(frequency['movieId']), :]

        return reduced

    @staticmethod
    def __restructure(data: pd.DataFrame):

        restructured = data.sample(frac=1, replace=False, axis=0, random_state=5)
        restructured['t'] = np.arange(stop=restructured.shape[0])
        restructured.index = restructured['t']
        
        return restructured

    @staticmethod
    def __extend(data: pd.DataFrame):

        extended = data.copy()
        extended['liked'] = np.where(extended['rating'] < 4.5, 0, 1)

        return extended

    def exc(self, data: pd.DataFrame, limit: int):

        # the number of film rating records
        self.logger.info(f'The initial number of observations: {data.shape}')

        # the number of ratings per film
        frequency = self.__frequency(data=data)
        self.logger.info(f'The number of distinct films: {frequency.shape}')

        # focus on films that have at least <limit> number of ratings
        reduced = self.__reduce(data=data, frequency=frequency, limit=limit)
        self.logger.info(f'Hence, the final number of observations: {reduced.shape}')

        # restructure
        restructured = self.__restructure(data=reduced)
        self.logger.info(restructured.head())

        # extend
        extended = self.__extend(data=restructured)
        self.logger.info(extended.head())

        return extended
