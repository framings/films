"""
Module: bayesian UCB
"""
import collections
import logging

import numpy as np
import pandas as pd

import src.functions.replay


class BayesianUCB:
    """
    Class: BayesianUCB
    """

    def __init__(self,
                 data: pd.DataFrame,
                 args: collections.namedtuple(typename='Arguments',
                                              field_names=['slate_size', 'batch_size', 'average_window'])):
        """

        :param data:
        """

        self.data = data
        self.args = args
        self.replay = src.functions.replay.Replay(data=self.data, args=self.args)

        # arms
        self.arms = self.data['movieId'].unique()

        # logging
        logging.basicConfig(level=logging.INFO, format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)

        # rewards
        self.Rewards = collections.namedtuple(typename='Rewards', field_names=['rewards', 'cumulative', 'running'])

    def score(self, history: pd.DataFrame, boundary: int, critical_value: float):
        """
        A reference for scores['ucb'] formula: https://www.itl.nist.gov/div898/handbook/prc/section1/prc14.htm

        :param history:
        :param boundary:
        :param critical_value:
        :return:
        """

        # the UCB policy applies to the historic dataset prior & equal to the current step
        excerpt = history.loc[history['t'] <= boundary, ]
        if excerpt.shape[0] == 0:
            recommendations: np.ndarray = np.random.choice(a=self.arms, size=self.args.slate_size, replace=False)
        else:
            scores = excerpt[['movieId', 'liked']].groupby(by='movieId').agg(mean=('liked', 'mean'),
                                                                             count=('liked', 'count'),
                                                                             std=('liked', 'std'))
            scores['ucb'] = scores['mean'] + np.true_divide(critical_value * scores['std'], np.sqrt(scores['count']))

            scores['movieId'] = scores.index
            scores = scores.sort_values('ucb', ascending=False)
            recommendations: np.ndarray = scores.loc[scores.index[0:self.args.slate_size], 'movieId'].values

        '''
        REPLAY ->
        '''

        history = self.replay.exc(history=history.copy(), boundary=boundary, recommendations=recommendations)

        return history

    def exc(self, critical_value: float):
        """

        :return:
        """

        # the empty history data frame - consider appending a <scoring_round> field
        history = pd.DataFrame(data=None, columns=self.data.columns)
        history = history.astype(self.data.dtypes.to_dict())

        for index in range((self.data.shape[0] // self.args.batch_size)):

            # temporary break point
            if index > 1000000:
                break

            # hence
            boundary = index * self.args.batch_size
            history = self.score(history=history, boundary=boundary, critical_value=critical_value)

        # In summary
        # ... the raw <rewards> values are the values of field <liked>
        # ... therefore, the <cumulative> values are just the cumulative sum values of field <liked>
        history['critical_value'] = critical_value
        history['cumulative'] = history['liked'].cumsum(axis=0)
        history['MA'] = history['liked'].rolling(window=self.args.average_window).mean()

        return history
