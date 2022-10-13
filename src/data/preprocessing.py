import logging

import pandas as pd


class Preprocessing:

    def __init__(self):
        """

        """

        logging.basicConfig(level=logging.INFO, format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)

    def __summary(self, frame: pd.DataFrame):

        summary = frame['movieId'].value_counts().rename('frequency').to_frame()
        summary.reset_index(drop=False, inplace=True)
        summary.rename(columns={'index': 'movieId'}, inplace=True)
        self.logger.info(summary)

        return summary

    def exc(self, frame: pd.DataFrame, limit: int):

        # the number of film rating records
        self.logger.info(f'The initial number of observations: {frame.shape}')

        # the number of ratings per film
        summary = self.__summary(frame=frame)
        self.logger.info(f'The number of distinct films: {summary.shape}')

        # focus on films that have at list <limit> number of ratings
        summary = summary.copy().loc[summary['frequency'] >= limit, :]
        self.logger.info(f'The number of films that have at least {limit} ratings: {summary.shape}')

        # hence
        data = frame.copy().loc[frame['movieId'].isin(summary['movieId']), :]
        self.logger.info(f'Hence, the final number of observations: {data.shape}')
