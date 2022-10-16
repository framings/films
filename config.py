"""
Module: config - the configuration file
"""
import os
import scipy.stats


class Config:

    def __init__(self):

        # the data directory
        self.data_directory = os.path.join(os.getcwd(), 'data')

        # the number film recommendations to present to a client
        self.slate_size = 5

        # instead of updating an online algorithm after each event, update it after every <batch_event> events
        self.batch_size = 1000

        # running average window
        self.average_window = 50

        # 95% confidence interval parameters w.r.t. two-tailed-test
        alpha = 1 - 0.95
        self.critical_value = scipy.stats.norm.ppf(q=(1 - alpha/2))
