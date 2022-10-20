
import logging

import dask
import numpy as np
import pandas as pd
import scipy.stats

import config
import src.algorithms.bayesianucb


class BayesianUCB:

    def __init__(self, data: pd.DataFrame):
        """
        dask.distributed.Client(n_workers=1, threads_per_worker=2)

        :param data: The preprocessed modelling data set in focus
        """

        # the data
        self.data = data

        # slate_size, batch_size, average_window
        self.args = config.Config().hyperparameters()

        # Note, for a two-tailed-test X% confidence interval: alpha = 1 - X/100
        self.__alpha = 1 - np.arange(start=0.90, stop=0.97, step=0.01)
        self.__critical_value = [scipy.stats.norm.ppf(q=(1 - alpha/2)) for alpha in self.__alpha]
        self.__bayesianucb = src.algorithms.bayesianucb.BayesianUCB(data=self.data, args=self.args)

        # logging
        logging.basicConfig(level=logging.INFO, format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)

    @dask.delayed
    def __evaluate(self, critical_value: float):
        """

        :return:
        """

        scores = self.__bayesianucb.exc(critical_value=critical_value)

        return scores

    def exc(self):
        """

        :return:
        """

        self.logger.info(f'\nThe number of hyperparameter values: {len(self.__critical_value)}')
        indices = np.arange(len(self.__critical_value))

        computations = []
        for index in indices:
            self.logger.info(f'({self.__critical_value[index]}, {self.__alpha[index]})')
            scores = self.__evaluate(critical_value=self.__critical_value[index])
            computations.append(scores)

        dask.visualize(computations, filename='bayesianUCB', format='pdf')
        calculations = dask.compute(computations, scheduler='threads', num_workers=3)[0]

        return calculations
