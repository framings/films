"""
Module: replay
"""
import collections
import logging
import pandas as pd
import numpy as np


class Random:
    """
    Class: Replay
    """

    def __init__(self, 
                 data: pd.DataFrame,
                 args: collections.namedtuple(typename='Arguments',
                                              field_names=['slate_size', 'batch_size', 'average_window'])):
        """

        :param data: The modelling data set in focus
        """

        self.data = data
        self.args = args

        # arms
        self.arms = self.data['movieId'].unique()

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

        # a temporary recommendation function, it recommends self.args.slate_size films
        recommendations: np.ndarray = np.random.choice(a=self.arms, size=self.args.slate_size, replace=False)

        '''
        REPLAY ->
        '''

        # the latest actions set starts from the latest lower boundary, and has self.args.batch_size records
        actions = self.data[boundary:(boundary + self.args.batch_size)]

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

        for index in range((self.data.shape[0] // self.args.batch_size)):

            # temporary break point
            if index > 9999:
                break

            # hence
            boundary = index * self.args.batch_size
            history, action_score = self.score(history=history, boundary=boundary)
            if action_score is not None:
                values = action_score['liked'].tolist()
                rewards.extend(values)

        # metrics
        cumulative = np.cumsum(rewards, dtype='float64')
        running = cumulative
        running[self.args.average_window:] = running[self.args.average_window:] - running[:-self.args.average_window]
        running = running[self.args.average_window - 1:] / self.args.average_window

        return self.Rewards(rewards=rewards, cumulative=cumulative, running=running)
