"""
Module: replay
"""
import collections

import numpy as np
import pandas as pd


class Replay:
    """
    Class: Replay
    """

    def __init__(self, data: pd.DataFrame,
                 args: collections.namedtuple(typename='Arguments',
                                              field_names=['slate_size', 'batch_size', 'average_window'])):
        """

        :param data:
        :param args:
        """

        self.data = data
        self.args = args

    def exc(self, history: pd.DataFrame, boundary: int, recommendations: np.ndarray):
        """

        :param history:
        :param boundary:
        :param recommendations:
        :return:
        """

        # the latest actions set starts from the latest lower boundary, and has self.batch_size records
        actions = self.data[boundary:(boundary + self.args.batch_size)]

        # the intersection of actions & recommendations via `movieId`
        actions = actions.copy().loc[actions['movieId'].isin(recommendations), :]

        # labelling the actions
        actions['scoring_round'] = boundary

        # in summary
        history = pd.concat([history, actions], axis=0)

        return history
