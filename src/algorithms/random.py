"""
Module: random
"""
import collections
import logging

import numpy as np
import pandas as pd

import src.functions.replay


class Random:
    """
    Class: Random
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
        self.replay = src.functions.replay.Replay(data=self.data, args=self.args)

        # arms
        self.arms = self.data['movieId'].unique()

        # logging
        logging.basicConfig(level=logging.INFO, format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)

    def score(self, history: pd.DataFrame, boundary: int) -> pd.DataFrame:
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

        history = self.replay.exc(history=history, boundary=boundary, recommendations=recommendations)

        return history

    def exc(self) -> pd.DataFrame:
        """

        :return:
        """

        # the empty history data frame - consider appending a <scoring_round> field
        history = pd.DataFrame(data=None, columns=self.data.columns)
        history = history.astype(self.data.dtypes.to_dict())

        for index in range((self.data.shape[0] // self.args.batch_size)):

            # temporary break point
            if index > 500000:
                break

            # hence
            boundary = index * self.args.batch_size
            history = self.score(history=history, boundary=boundary)

        # in summary
        # ... the raw <rewards> values are the values of field <liked>
        # ... therefore, the <cumulative> values are just the cumulative sum values of field <liked>
        history['cumulative'] = history['liked'].cumsum(axis=0)
        history['MA'] = history['liked'].rolling(window=self.args.average_window).mean()

        return history
