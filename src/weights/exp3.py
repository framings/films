"""
Module: weights/exp3.py
"""
import logging

import numpy as np
import pandas as pd


class EXP3:
    """
    Class: EXP3
    """

    def __init__(self):
        """

        """

        self.default_fraction = 0.0

        # logging
        logging.basicConfig(level=logging.INFO, format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)

    def __fraction(self, metric: float, probability: float):
        """

        :param metric:
        :param probability:
        :return:
        """

        return np.where(np.isnan(metric), self.default_fraction, metric/probability)

    def __update(self, factors: pd.DataFrame, latest: pd.DataFrame, gamma: float):
        """

        :param factors: The data frame of distinct film identification codes along with their
                        weights & probabilities.  This program updates the weights of the films in <latest>
        :param latest: The latest additions to history.
        :param gamma: An algorithm constant.
        :return:
        """

        if latest.empty:
            return factors
        else:
            temporary = factors.copy()
            temporary['state'] = temporary['movieId'].isin(latest['movieId'].array)
            temporary = temporary.merge(latest, on='movieId', how='left')
            temporary['fraction'] = self.__fraction(metric=temporary['metric'], probability=temporary['probability'])
            temporary['weight'] = temporary['weight'].array * np.exp(gamma * temporary['fraction'] / temporary.shape[0])

            indices = temporary.index[temporary['metric'].notna()]
            factors.loc[indices, 'weight'] = temporary.loc[indices, 'weight'].array

            return factors

    def exc(self, factors: pd.DataFrame, latest: pd.DataFrame, gamma: float):
        """

        :param factors: The data frame of distinct film identification codes along with their
                        weights & probabilities.  This program updates the weights of the films in <latest>
        :param latest: The latest additions to history.
        :param gamma: An algorithm constant.
        :return:
        """

        return self.__update(factors=factors, latest=latest, gamma=gamma)
