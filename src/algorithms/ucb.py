
import collections
import logging

import pandas as pd

import config


class UCB:

    def __init__(self, data: pd.DataFrame):
        """

        """

        self.data = data
        self.arms = self.data['movieId'].unique()

        configurations = config.Config()
        self.slate_size = configurations.slate_size
        self.batch_size = configurations.batch_size
        self.average_window = configurations.average_window
        self.ucb_scale = configurations.ucb_scale

        # logging
        logging.basicConfig(level=logging.INFO, format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)

        # rewards
        self.Rewards = collections.namedtuple(typename='Rewards', field_names=['rewards', 'cumulative', 'running'])

    def score(self, history: pd.DataFrame, boundary: int):
        """

        :return:
        """

        # the UCB policy applies to the historic dataset prior & equal to the current step
        excerpt = history.loc[history['t'] <= boundary, ]

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
