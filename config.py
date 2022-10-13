"""
Module: config - the configuration file
"""
import os


class Config:

    def __init__(self):

        self.data_directory = os.path.join(os.getcwd(), 'data')
