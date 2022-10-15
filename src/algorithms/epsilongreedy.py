"""
Module: epsilon greedy
"""
import collections
import logging

import numpy as np
import pandas as pd

import config


class EpsilonGreedy:
    """
    Class: EpsilonGreedy
    """

    def __init__(self, data: pd.DataFrame, epsilon: float):
        """

        :param data: The modelling data set in focus
        :param epsilon: The fraction of time steps to explore
        """

        self.data = data
        self.arms = self.data['movieId'].unique()
        self.epsilon = epsilon

        configurations = config.Config()
        self.slate_size = configurations.slate_size
        self.batch_size = configurations.batch_size
        self.average_window = configurations.average_window

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
        self.logger.info(f'Explore: {explore}')

        if explore == 1 or excerpt.shape[0] == 0:
            # a temporary recommendation function
            recommendations: np.ndarray = np.random.choice(a=self.arms, size=self.slate_size, replace=False)
            self.logger.info(f'Recommendations: {recommendations}')
        else:
            # exploit
            scores = excerpt[['movieId', 'liked']].groupby(by='movieId').agg(mean=('liked', 'mean'),
                                                                             count=('liked', 'count'))
            scores['movieId'] = scores.index
            scores = scores.sort_values('mean', ascending=False)
            recommendations: np.ndarray = scores.loc[scores.index[0:self.slate_size], 'movieId'].values

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

        # the empty history data frame - consider appending a <scoring_round> field
        history = pd.DataFrame(data=None, columns=self.data.columns)
        history = history.astype(self.data.dtypes.to_dict())

        # rewards
        rewards = []

        for index in range((self.data.shape[0] // self.batch_size)):

            if index > 99999:
                break

            # hence
            boundary = index * self.batch_size
            history, action_score = self.score(history=history, boundary=boundary)
            if action_score is not None:
                values = action_score['liked'].tolist()
                rewards.extend(values)

        # history
        self.logger.info(f'History:\n {history}')

        # metrics
        cumulative = np.cumsum(rewards, dtype='float64')
        running = cumulative
        self.logger.info(running[self.average_window:])
        running[self.average_window:] = running[self.average_window:] - running[:-self.average_window]
        running = running[self.average_window - 1:] / self.average_window

        return self.Rewards(rewards=rewards, cumulative=cumulative, running=running)