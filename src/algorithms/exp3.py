import collections
import logging

import numpy as np
import pandas as pd

import config
import src.functions.replay


class EXP3:

    def __init__(self,
                 data: pd.DataFrame,
                 args: collections.namedtuple(typename='Arguments',
                                              field_names=['slate_size', 'batch_size', 'average_window'])):
        """

        :param data:
        :param args:
        """

        self.data = data
        self.args = args
        self.replay = src.functions.replay.Replay(data=self.data, args=self.args)

        # arms
        self.arms = self.data['movieId'].unique()

        # the random number generator instance
        self.rng = np.random.default_rng(seed=config.Config().seed)

        # logging
        logging.basicConfig(level=logging.INFO, format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def __probabilities(weights: pd.Series, gamma: float):

        total: float = weights.sum()
        length: int = weights.shape[0]
        quotient: float = gamma / length

        calculations: pd.Series = (1.0 - gamma) * weights.divide(total) + quotient

        return calculations.array

    def __draw(self, factors: pd.DataFrame):

        recommendations = self.rng.choice(a=factors['movieId'],
                                          size=self.args.slate_size, p=factors['probability'],
                                          replace=False)

        return recommendations

    def score(self, history: pd.DataFrame, factors: pd.DataFrame, boundary: int, gamma: float):
        """

        :return:
        """

        factors['probability'] = self.__probabilities(weights=factors['weight'], gamma=gamma)
        recommendations = self.__draw(factors=factors)

        '''
        REPLAY ->
        '''

        history = self.replay.exc(history=history.copy(), boundary=boundary, recommendations=recommendations)

        return history, factors

    @staticmethod
    def __fraction(value: float, probability: float):

        return np.where(np.isnan(value), 0, value/probability)

    def __update(self, factors: pd.DataFrame, actions: pd.DataFrame, gamma: float):
        """

        :param factors:
        :param actions:
        :param gamma:
        :return:
        """

        if actions.empty:
            return factors
        else:
            focus = factors.copy()
            excerpt = actions[['movieId', 'liked']].groupby(by='movieId').agg(value=('liked', 'mean'))
            excerpt.reset_index(drop=False, inplace=True)

            focus['state'] = focus['movieId'].isin(excerpt['movieId'].array)
            focus = focus.merge(excerpt, on='movieId', how='left')
            self.logger.info(focus)

            focus['fraction'] = self.__fraction(value=focus['value'], probability=focus['probability'])
            self.logger.info(focus)

            focus['weight'] = focus['weight'].array * np.exp(gamma * focus['fraction'] / focus.shape[0])

            indices = focus.index[focus['state']]
            factors.loc[indices, 'weight'] = focus.loc[indices, 'weight'].array

            self.logger.info(focus.index[focus['state']])
            self.logger.info(focus.iloc[indices, ])

            return factors

    def exc(self, gamma: float):
        """

        :param gamma:
        :return:
        """

        # the empty history data frame - consider appending a <scoring_round> field
        history = pd.DataFrame(data=None, columns=self.data.columns)
        history = history.astype(self.data.dtypes.to_dict())

        # initial weights - a list of ones of length self.arms.shape[0]
        factors = pd.DataFrame(data={'movieId': self.arms, 'weight': [1.0] * self.arms.shape[0],
                                     'probability': [0.0] * self.arms.shape[0]})

        for index in range((self.data.shape[0] // self.args.batch_size)):

            # temporary break point
            if index > 500:
                break

            # hence
            boundary = index * self.args.batch_size
            history, factors = self.score(history=history, factors=factors, boundary=boundary, gamma=gamma)
            factors = self.__update(factors=factors, actions=history[history['scoring_round'] == boundary], gamma=gamma)

        history['gamma'] = gamma
        history['cumulative'] = history['liked'].cumsum(axis=0)
        history['MA'] = history['liked'].rolling(window=self.args.average_window).mean()

        return history
