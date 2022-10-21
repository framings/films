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

        recommendations = self.rng.choice(a=factors['movieID'], size=self.args.slate_size, p=factors['probability'], replace=False)

        return recommendations

    def score(self, history: pd.DataFrame, factors: pd.DataFrame, boundary: int, gamma: float):
        """

        :return:
        """

        factors[:, 'probability'] = self.__probabilities(weights=factors['weight'], gamma=gamma)
        recommendations = self.__draw(factors=factors)

        '''
        REPLAY ->
        '''

        history = self.replay.exc(history=history.copy(), boundary=boundary, recommendations=recommendations)

        return history, factors

    @staticmethod
    def __fraction(state: bool, value: float, probability: float):

        if state:
            return value/probability
        else:
            return 0.0

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
            focus['state'] = focus['movieId'].array.isin(excerpt['movieId'])
            focus = focus.merge(excerpt, on='movieId', how='left')
            focus['fraction'] = self.__fraction(state=focus['state'], value=focus['value'], probability=focus['probability'])
            focus['weight'].array * np.exp(gamma * focus['fraction'] / focus.shape[0])

    def exc(self, gamma: float):
        """

        :param gamma:
        :return:
        """

        # the empty history data frame - consider appending a <scoring_round> field
        history = pd.DataFrame(data=None, columns=self.data.columns)
        history = history.astype(self.data.dtypes.to_dict())

        # initial weights - a list of ones of length self.arms.shape[0]
        factors = pd.DataFrame(data={'movieID': self.arms,
                                     'weight': [1.0] * self.arms.shape[0],
                                     'probability': [0.0] * self.arms.shape[0]})

        for index in range((self.data.shape[0] // self.args.batch_size)):

            # temporary break point
            if index > 100000:
                break

            # hence
            boundary = index * self.args.batch_size
