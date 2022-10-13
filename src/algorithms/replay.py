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

    def score(self, history: pd.DataFrame, boundary: int, recommendations: np.ndarray):

        actions = self.data[boundary:(boundary + self.batch_size)]
        actions = actions.copy().loc[actions['movieId'].isin(recommendations), :]

        actions['scoring_round'] = boundary
        history = pd.concat([history, actions], axis=1)
        action_score = actions[['movieId', 'liked']]

        return history, action_score

    def exc(self):

        # the empty history data frame
        # scoring_round?
        history = pd.DataFrame(data=None, columns=self.data.columns)
        history = history.astype(self.data.dtypes.to_dict())

        # rewards
        rewards = []

        for index in range((self.data.shape[0] // self.batch_size)):

            if index > 27:
                break

            # the lower boundary
            boundary = index * self.batch_size

            # a temporary recommendation function
            recommendations: np.ndarray = np.random.choice(a=self.data['movieId'].unique(), size=self.slate_size, replace=False)
            self.logger.info(recommendations)

            # hence
            history, action_score = self.score(history=history, boundary=boundary, recommendations=recommendations)
            if action_score is not None:
                values = action_score['liked'].tolist()
                rewards.extend(values)

            # thus far
            self.logger.info(np.cumsum(rewards))