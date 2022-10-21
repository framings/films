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

    def __draw(self, probabilities):

        recommendations = self.rng.choice(a=self.arms, size=self.args.slate_size, p=probabilities, replace=False)

        return recommendations

    def exc(self, gamma: float):
        """

        :param gamma:
        :return:
        """
