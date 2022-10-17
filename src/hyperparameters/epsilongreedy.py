import collections

import numpy as np
import dask
import pandas as pd

import src.algorithms.epsilongreedy

import config


class EpsilonGreedy:

    def __init__(self, data: pd.DataFrame):
        """

        :param data: The preprocessed modelling data set in focus
        """

        # the data
        self.data = data

        # slate_size, batch_size, average_window
        self.args = config.Config().hyperparameters()

        # the range hyperparameter values under exploration
        self.__epsilon = np.arange(start=0.01, stop=0.40, step=0.01)

    @dask.delayed
    def __evaluate(self, epsilon: float):
        """

        :param epsilon:
        :return:
        """

        scores = src.algorithms.epsilongreedy.EpsilonGreedy(data=self.data, args=self.args, epsilon=epsilon)

        return scores

    @dask.delayed
    def __aggregate(self, epsilon: float,
                    scores: collections.namedtuple(typename='Rewards',
                                                   field_names=['rewards', 'cumulative', 'running'])) -> dict:
        """

        :param epsilon:
        :param scores:
        :return:
        """

        return {'epsilon': epsilon, 'median': np.median(scores.rewards)}

    def exc(self):
        """

        :return:
        """

        computations = []
        for epsilon in self.__epsilon:
            scores = self.__evaluate(epsilon=epsilon)
            aggregate = self.__aggregate(epsilon=epsilon, scores=scores)
            computations.append(aggregate)

        dask.visualize(computations, filename='epsilonGreedy', format='pdf')
        calculations = dask.compute(computations, scheduler='processes')[0]

        return calculations
