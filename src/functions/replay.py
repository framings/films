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

    def __servings(self, servings: pd.DataFrame,  boundary: int, recommendations: np.ndarray):
        """

        :param servings:
        :param boundary:
        :param recommendations:
        :return:
        """

        latest = self.data[boundary:(boundary + self.args.batch_size)]
        latest = latest.copy()[['userId', 't']].drop_duplicates(ignore_index=True)
        latest['scoring_round'] = boundary

        suggestions = np.expand_dims(recommendations, axis=0)
        suggestions_ = pd.DataFrame(data=np.repeat(suggestions, latest.shape[0], axis=0))

        additions = pd.concat([latest, suggestions_], axis=1)
        additions = additions.melt(id_vars=['userId', 't', 'scoring_round'],
                                   value_vars=np.arange(suggestions.shape[1]),
                                   var_name='digit',
                                   value_name='recommendations')
        additions.drop(columns='digit', inplace=True)

        servings = pd.concat([servings, additions], axis=0)

        return servings

    def __history(self, history: pd.DataFrame, boundary: int, recommendations: np.ndarray):
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

    def exc(self, history: pd.DataFrame, servings: pd.DataFrame, boundary: int, recommendations: np.ndarray):
        """

        :param history:
        :param servings:
        :param boundary:
        :param recommendations:
        :return:
        """

        servings = self.__servings(servings=servings.copy(), boundary=boundary, recommendations=recommendations)
        history = self.__history(history=history.copy(), boundary=boundary, recommendations=recommendations)

        return history, servings
