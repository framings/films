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
    def __probabilities(weights: list, gamma: float):

        total = float(sum(weights))
        length = len(weights)
        def __probability(weight): return (1.0 - gamma) * (weight / total) + (gamma / length)

        return [__probability(weight=weight) for weight in weights]

    def __draw(self, probabilities):

        recommendations = self.rng.choice(a=self.arms, size=self.args.slate_size, p=probabilities, replace=False)

        return recommendations

    def score(self, history: pd.DataFrame, boundary: int, weights: list, gamma: float):
        """

        :return:
        """

        probabilities = self.__probabilities(weights=weights, gamma=gamma)
        recommendations = self.__draw(probabilities=probabilities)

    def exc(self, gamma: float):
        """

        :param gamma:
        :return:
        """

        # the empty history data frame - consider appending a <scoring_round> field
        history = pd.DataFrame(data=None, columns=self.data.columns)
        history = history.astype(self.data.dtypes.to_dict())

        # initial weights - a list of ones of length self.arms.shape[0]
        weights = [1.0] * self.arms.shape[0]

        for index in range((self.data.shape[0] // self.args.batch_size)):

            # temporary break point
            if index > 100000:
                break

            # hence
            boundary = index * self.args.batch_size
