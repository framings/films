"""
Module: config - the configuration file
"""
import os


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
