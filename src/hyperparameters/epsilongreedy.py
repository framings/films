import collections

import numpy as np
import dask
import pandas as pd

import src.algorithms.epsilongreedy

import config


class EpsilonGreedy:

    def __init__(self, data: pd.DataFrame):
        """

        """

        self.data = data
        self.args = config.Config().models()
        self.__epsilon = np.arange(start=0.01, stop=0.40, step=0.01)

    @dask.delayed
    def __evaluate(self, epsilon: float):
        """

        :return:
        """

        scores = src.algorithms.epsilongreedy.EpsilonGreedy(data=self.data, args=self.args, epsilon=epsilon)

        return scores

    @dask.delayed
    def __aggregate(self, epsilon: float,
                    scores: collections.namedtuple(typename='',
                                                   field_names=['rewards', 'cumulative', 'running'])) -> dict:
        """

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
