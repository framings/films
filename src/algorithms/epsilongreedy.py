"""
Module: epsilon greedy
"""
import collections
import logging

import numpy as np
import pandas as pd


class EpsilonGreedy:
    """
    Class: EpsilonGreedy
    """

    def __init__(self,
                 data: pd.DataFrame,
                 args: collections.namedtuple(typename='Arguments',
                                              field_names=['slate_size', 'batch_size', 'average_window']),
                 epsilon: float):
        """

        :param data: The modelling data set in focus
        :param args: Modelling parameters
        :param epsilon: The fraction of time steps to explore
        """

        self.data = data
        self.arms = self.data['movieId'].unique()
        self.epsilon = epsilon

        self.slate_size = args.slate_size
        self.batch_size = args.batch_size
        self.average_window = args.average_window

        # logging
        logging.basicConfig(level=logging.INFO, format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)

        # rewards
        self.Rewards = collections.namedtuple(typename='Rewards', field_names=['rewards', 'cumulative', 'running'])

    def score(self, history: pd.DataFrame, boundary: int):
        """

        :param history:
        :param boundary:
        :return:
        """

        # the epsilon greedy policy applies to the historic dataset prior & equal to the current step
        excerpt = history.loc[history['t'] <= boundary, ]

        # the likelihood of an explore option is self.epsilon
        explore = np.random.binomial(n=1, p=self.epsilon, size=1)
        if explore == 1 or excerpt.shape[0] == 0:
            # a temporary recommendation function
            recommendations: np.ndarray = np.random.choice(a=self.arms, size=self.slate_size, replace=False)
        else:
            # exploit
            scores = excerpt[['movieId', 'liked']].groupby(by='movieId').agg(mean=('liked', 'mean'),
                                                                             count=('liked', 'count'))
            scores['movieId'] = scores.index
            scores = scores.sort_values('mean', ascending=False)
            recommendations: np.ndarray = scores.loc[scores.index[0:self.slate_size], 'movieId'].values

        '''
        REPLAY ->
        '''

        # the latest actions set starts from the latest lower boundary, and has self.batch_size records
        actions = self.data[boundary:(boundary + self.batch_size)]

        # the intersection of actions & recommendations via `movieId`
        actions = actions.copy().loc[actions['movieId'].isin(recommendations), :]

        # labelling the actions
        actions['scoring_round'] = boundary

        # in summary
        history = pd.concat([history, actions], axis=0)
        action_score = actions[['movieId', 'liked']]

        return history, action_score

    def exc(self):
        """

        :return:
        """

        # the empty history data frame - consider appending a <scoring_round> field
        history = pd.DataFrame(data=None, columns=self.data.columns)
        history = history.astype(self.data.dtypes.to_dict())

        # rewards
        rewards = []

        for index in range((self.data.shape[0] // self.batch_size)):

            # temporary break point
            if index > 9999:
                break

            # hence
            boundary = index * self.batch_size
            history, action_score = self.score(history=history, boundary=boundary)
            if action_score is not None:
                values = action_score['liked'].tolist()
                rewards.extend(values)

        # metrics
        cumulative = np.cumsum(rewards, dtype='float64')
        running = pd.Series(rewards).rolling(window=self.average_window).mean().iloc[self.average_window:].values

        return self.Rewards(rewards=rewards, cumulative=cumulative, running=running)
