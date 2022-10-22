import numpy as np
import pandas as pd


class EXP3:

    def __init__(self):
        """

        """

    @staticmethod
    def __fraction(metric: float, probability: float):
        """

        :param metric:
        :param probability:
        :return:
        """

        return np.where(np.isnan(metric), 0, metric/probability)

    def __update(self, factors: pd.DataFrame, latest: pd.DataFrame, gamma: float):
        """

        :param factors:
        :param latest:
        :param gamma:
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

        :param factors:
        :param latest:
        :param gamma:
        :return:
        """

        return self.__update(factors=factors, latest=latest, gamma=gamma)
