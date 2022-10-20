"""
Module: bayesian UCB
"""
import collections
import logging

import numpy as np
import pandas as pd


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

    def exc(self, critical_value: float):
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
            history, action_score = self.score(history=history, boundary=boundary, critical_value=critical_value)
            if action_score is not None:
                values = action_score['liked'].tolist()
                rewards.extend(values)

        # metrics
        cumulative = np.cumsum(rewards, dtype='float64')
        running = pd.Series(rewards).rolling(window=self.args.average_window).mean().iloc[self.args.average_window:].values

        return self.Rewards(rewards=rewards, cumulative=cumulative, running=running)
