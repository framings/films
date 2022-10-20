"""
Module: epsilongreedy
"""
import collections
import logging

import numpy as np
import pandas as pd

import config
import src.functions.replay


class EpsilonGreedy:
    """
    Class: EpsilonGreedy
    """

    def __init__(self,
                 data: pd.DataFrame,
                 args: collections.namedtuple(typename='Arguments',
                                              field_names=['slate_size', 'batch_size', 'average_window'])):
        """

        :param data: The modelling data set in focus
        :param args: Modelling parameters
        """

        self.data = data
        self.args = args
        self.replay = src.functions.replay.Replay(data=self.data, args=self.args)

        # the arms
        self.arms = self.data['movieId'].unique()

        # the random number generator instance
        self.rng = np.random.default_rng(seed=config.Config().seed)

        # logging
        logging.basicConfig(level=logging.INFO, format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)

    def score(self, history: pd.DataFrame, boundary: int, epsilon: float) -> pd.DataFrame:
        """
        Reference: https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html#\
            numpy.random.Generator.choice

        :param history:
        :param boundary:
        :param epsilon:
        :return:
        """

        # the epsilon greedy policy applies to the historic dataset prior & equal to the current step
        excerpt = history.loc[history['t'] <= boundary, ]

        # the likelihood of an explore option is epsilon
        explore = self.rng.binomial(n=1, p=epsilon, size=1)
        if explore == 1 or excerpt.shape[0] == 0:
            # a temporary recommendation function
            recommendations: np.ndarray = self.rng.choice(a=self.arms, size=self.args.slate_size, replace=False)
        else:
            # exploit
            scores = excerpt[['movieId', 'liked']].groupby(by='movieId').agg(mean=('liked', 'mean'),
                                                                             count=('liked', 'count'))
            scores['movieId'] = scores.index
            scores = scores.sort_values('mean', ascending=False)
            recommendations: np.ndarray = scores.loc[scores.index[0:self.args.slate_size], 'movieId'].values

        '''
        REPLAY ->
        '''

        history = self.replay.exc(history=history.copy(), boundary=boundary, recommendations=recommendations)

        return history

    def exc(self, epsilon: float) -> pd.DataFrame:
        """

        :param epsilon: The fraction of time steps to explore
        :return:
        """

        # the empty history data frame - consider appending a <scoring_round> field
        history = pd.DataFrame(data=None, columns=self.data.columns)
        history = history.astype(self.data.dtypes.to_dict())

        for index in range((self.data.shape[0] // self.args.batch_size)):

            # Temporary break point
            if index > 100000:
                break

            # Hence
            boundary = index * self.args.batch_size
            history = self.score(history=history, boundary=boundary, epsilon=epsilon)

        # In summary
        # ... the raw <rewards> values are the values of field <liked>
        # ... therefore, the <cumulative> values are just the cumulative sum values of field <liked>
        history['epsilon'] = epsilon
        history['cumulative'] = history['liked'].cumsum(axis=0)
        history['MA'] = history['liked'].rolling(window=self.args.average_window).mean()

        return history
