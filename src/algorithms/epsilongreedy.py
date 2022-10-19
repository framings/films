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
                                              field_names=['slate_size', 'batch_size', 'average_window'])):
        """

        :param data: The modelling data set in focus
        :param args: Modelling parameters
        """

        self.data = data
        self.arms = self.data['movieId'].unique()

        self.slate_size = args.slate_size
        self.batch_size = args.batch_size
        self.average_window = args.average_window

        # logging
        logging.basicConfig(level=logging.INFO, format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)

        # rewards
        self.Rewards = collections.namedtuple(typename='Rewards',
                                              field_names=['rewards', 'cumulative', 'running', 'history'])

    def score(self, history: pd.DataFrame, boundary: int, epsilon: float):
        """

        :param history:
        :param boundary:
        :param epsilon:
        :return:
        """

        # the epsilon greedy policy applies to the historic dataset prior & equal to the current step
        excerpt = history.loc[history['t'] <= boundary, ]

        # the likelihood of an explore option is epsilon
        explore = np.random.binomial(n=1, p=epsilon, size=1)
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

    def exc(self, epsilon: float):
        """

        :param epsilon: The fraction of time steps to explore
        :return:
        """

        # the empty history data frame - consider appending a <scoring_round> field
        history = pd.DataFrame(data=None, columns=self.data.columns)
        history = history.astype(self.data.dtypes.to_dict())

        # rewards
        rewards = []

        for index in range((self.data.shape[0] // self.batch_size)):

            # Temporary break point
            if index > 1000000:
                break

            # Hence
            boundary = index * self.batch_size
            history, action_score = self.score(history=history, boundary=boundary, epsilon=epsilon)

            # Note that if <action_score> is empty then <history>'s input & output lengths will
            # be the same
            if action_score is not None:
                values = action_score['liked'].tolist()
                rewards.extend(values)

        # Metrics: the rewards series should be calculable via <history>, check
        cumulative = np.cumsum(rewards, dtype='float64')
        running = pd.Series(rewards).rolling(window=self.average_window).mean().iloc[self.average_window:].values

        # In progress
        history['epsilon'] = epsilon
        history['MA'] = history['liked'].rolling(window=self.average_window).mean()

        return self.Rewards(rewards=rewards, cumulative=cumulative, running=running, history=history)
