"""
Module: config - the configuration file
"""
import os
import collections


class Config:

    def __init__(self):

        # the data directory
        self.data_directory = os.path.join(os.getcwd(), 'data')

        # the number film recommendations to present to a client
        self.slate_size = 5

        # instead of updating an online algorithm after each event, update it after every <batch_event> events
        self.batch_size = 100

        # running average window
        self.average_window = 200

        # Note, for a two-tailed-test 95% confidence interval:
        # alpha = 1 - 0.95
        # self.critical_value = scipy.stats.norm.ppf(q=(1 - alpha/2))
        self.critical_value = 1.5

    @staticmethod
    def models():
        """
        slate_size: the number film recommendations to present to a client
        batch_size: instead of updating an online algorithm after each event, update it after every <batch_event> events
        average_window: the length of the running average window

        :return:
        """

        Arguments = collections.namedtuple(typename='Arguments',
                                           field_names=['slate_size', 'batch_size', 'average_window'])
        return Arguments(slate_size=5, batch_size=100, average_window=200)

    @staticmethod
    def hyperparameters():
        """
        slate_size: the number film recommendations to present to a client
        batch_size: instead of updating an online algorithm after each event, update it after every <batch_event> events

        :return:
        """

        Arguments = collections.namedtuple(typename='Arguments',
                                           field_names=['slate_size', 'batch_size', 'average_window'])

        return Arguments(slate_size=5, batch_size=10000, average_window=200)
