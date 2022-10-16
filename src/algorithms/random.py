"""
Module: replay
"""
import collections
import logging
import pandas as pd
import numpy as np

import config


class Random:
    """
    Class: Replay
    """

    def __init__(self, data: pd.DataFrame):
        """

        :param data: The modelling data set in focus
        """

        self.data = data
        self.arms = self.data['movieId'].unique()

        # configurations
        configurations = config.Config()
        self.batch_size = configurations.batch_size
        self.slate_size = configurations.slate_size
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

        # a temporary recommendation function, it recommends self.slate_size films
        recommendations: np.ndarray = np.random.choice(a=self.arms, size=self.slate_size, replace=False)

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
            if index > 99999:
                break

            # hence
            boundary = index * self.batch_size
            history, action_score = self.score(history=history, boundary=boundary)
            if action_score is not None:
                values = action_score['liked'].tolist()
                rewards.extend(values)

        # metrics
        cumulative = np.cumsum(rewards, dtype='float64')
        running = cumulative
        running[self.average_window:] = running[self.average_window:] - running[:-self.average_window]
        running = running[self.average_window - 1:] / self.average_window

        return self.Rewards(rewards=rewards, cumulative=cumulative, running=running)
