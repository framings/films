"""
Module: exp3
"""
import collections
import logging

import numpy as np
import pandas as pd

import config
import src.functions.replay
import src.weights.exp3


class EXP3:
    """
    Class: EXP3
    """

    def __init__(self,
                 data: pd.DataFrame,
                 args: collections.namedtuple(typename='Arguments',
                                              field_names=['slate_size', 'batch_size', 'average_window'])):
        """

        :param data: The modelling data set in focus
        :param args: The modelling arguments
        """

        self.data = data
        self.args = args
        self.replay = src.functions.replay.Replay(data=self.data, args=self.args)
        self.weights = src.weights.exp3.EXP3()

        # arms
        self.arms = self.data['movieId'].unique()
        self.n_arms = self.arms.shape[0]

        # the random number generator instance
        self.rng = np.random.default_rng(seed=config.Config().seed)

        # logging
        logging.basicConfig(level=logging.INFO, format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)

    def __probabilities(self, weights: pd.Series, gamma: float):
        """

        :param weights:
        :param gamma:
        :return:
        """

        quotient: float = gamma / self.n_arms
        calculations: pd.Series = (1.0 - gamma) * weights.divide(weights.sum()) + quotient
        probabilities = calculations.array

        return probabilities

    def __draw(self, factors: pd.DataFrame):

        recommendations = self.rng.choice(a=factors['movieId'], size=self.args.slate_size, p=factors['probability'],
                                          replace=False)

        return recommendations

    def score(self, history: pd.DataFrame, factors: pd.DataFrame, boundary: int, gamma: float):
        """

        :return:
        """

        # recommendations
        factors['probability'] = self.__probabilities(weights=factors['weight'], gamma=gamma)
        recommendations = self.__draw(factors=factors)

        # replay
        history = self.replay.exc(history=history.copy(), boundary=boundary, recommendations=recommendations)

        return history, factors

    def exc(self, gamma: float):
        """

        :param gamma:
        :return:
        """

        # the empty history data frame - consider appending a <scoring_round> field
        history = pd.DataFrame(data=None, columns=self.data.columns)
        history = history.astype(self.data.dtypes.to_dict())

        # initial weights - a list of ones of length self.n_arms
        factors = pd.DataFrame(data={'movieId': self.arms,
                                     'weight': [1.0] * self.n_arms,
                                     'probability': [0.0] * self.n_arms})

        # learning
        for index in range((self.data.shape[0] // self.args.batch_size)):

            # temporary break point
            if index > 500000:
                break

            # hence
            boundary = index * self.args.batch_size
            history, factors= self.score(history=history, factors=factors, boundary=boundary, gamma=gamma)

            # latest
            latest = history[history['scoring_round'] == boundary]
            latest = latest.copy()[['movieId', 'liked']].groupby(by='movieId').agg(metric=('liked', 'sum'))
            latest.reset_index(drop=False, inplace=True)

            # update weights
            factors = self.weights.exc(factors=factors, latest=latest, gamma=gamma)

        # reviewing
        self.logger.info(factors['probability'].array)

        # in summary
        # ... the raw <rewards> values are the values of field <liked>
        # ... therefore, the <cumulative> values are just the cumulative sum values of field <liked>
        history['gamma'] = gamma
        history['cumulative'] = history['liked'].cumsum(axis=0)
        history['MA'] = history['liked'].rolling(window=self.args.average_window).mean()

        return history
