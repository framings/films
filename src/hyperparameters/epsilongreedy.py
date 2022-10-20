"""
Module: epsilongreedy
"""
import logging

import dask
import numpy as np
import pandas as pd

import config
import src.algorithms.epsilongreedy


class EpsilonGreedy:
    """
    Class: EpsilonGreedy

    This class' focus is the optimal <epsilon> hyperparameter.  If large memory machines are accessible, dask
    based computations are worth it.
    """

    def __init__(self, data: pd.DataFrame):
        """
        dask.distributed.Client(n_workers=1, threads_per_worker=2)

        :param data: The preprocessed modelling data set in focus
        """

        # the data
        self.data = data

        # slate_size, batch_size, average_window
        self.args = config.Config().hyperparameters()

        # the range hyperparameter values under exploration
        self.__epsilon = np.arange(start=0.08, stop=0.16, step=0.01)
        self.__epsilongreedy = src.algorithms.epsilongreedy.EpsilonGreedy(data=self.data, args=self.args)

        # logging
        logging.basicConfig(level=logging.INFO, format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)

    @dask.delayed
    def __evaluate(self, epsilon: float):
        """

        :param epsilon:
        :return:
        """

        scores = self.__epsilongreedy.exc(epsilon=epsilon)

        return scores

    def exc(self):
        """

        :return:
        """

        self.logger.info(f'\nThe number of hyperparameter values: {len(self.__epsilon)}')

        computations = []
        for epsilon in self.__epsilon:
            scores = self.__evaluate(epsilon=epsilon)
            computations.append(scores)

        dask.visualize(computations, filename='epsilonGreedy', format='pdf')
        calculations = dask.compute(computations, scheduler='threads', num_workers=3)[0]

        return calculations
