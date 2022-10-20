
import logging

import dask
import numpy as np
import pandas as pd
import scipy.stats

import config


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
        __alpha = 1 - np.arange(start=0.95, stop=0.99, step=0.01)
        self.__critical_value = [scipy.stats.norm.ppf(q=(1 - alpha/2)) for alpha in __alpha]

        # logging
        logging.basicConfig(level=logging.INFO, format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)

    @dask.delayed
    def __evaluate(self):
        """

        :return:
        """

    def exc(self):
        """

        :return:
        """

        self.logger.info(f'\nThe number of hyperparameter values: {len(self.__critical_value)}')

        for critical_value in self.__critical_value:

            self.logger.info(critical_value)
