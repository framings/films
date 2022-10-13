import logging
import pandas as pd
import numpy as np

import config


class Replay:

    def __init__(self, data: pd.DataFrame):
        """

        """

        self.data = data

        # configurations
        configurations = config.Config()
        self.batch_size = configurations.batch_size
        self.slate_size = configurations.slate_size

        # logging
        logging.basicConfig(level=logging.INFO, format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)

    def score(self, history: pd.DataFrame, boundary: int, recommendations):

        actions = self.data[boundary:(boundary + self.batch_size)]
        self.logger.info(actions.head())

    def exc(self):

        print(self.data.dtypes.to_dict())

        history = pd.DataFrame(data=None, columns=self.data.columns)
        history = history.astype(self.data.dtypes.to_dict())
        self.logger.info(history.info())

        recommendations: np.ndarray = np.random.choice(a=self.data['movieId'].unique(), size=self.slate_size, replace=False)
        self.logger.info(type(recommendations))

        self.score(history=history, boundary=0, recommendations=recommendations)

        for index in range((self.data.shape[0] // self.batch_size)):

            if index > 9:
                break

            self.logger.info(f'index: {index}')
            boundary = index * self.batch_size
            self.logger.info(boundary)
