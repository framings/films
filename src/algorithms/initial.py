import logging

import pandas as pd


class Initial:

    def __init__(self, preprocessed: pd.DataFrame):
        """

        """

        self.preprocessed = preprocessed

        # logging
        logging.basicConfig(level=logging.INFO, format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def __supplement(temporary: pd.DataFrame) -> pd.DataFrame:

        # ensure the <movieId> field exists
        temporary['movieId'] = temporary.index

        # adding a time step field
        temporary['t'] = 0

        # hence, setting the index as time steps
        temporary.index = temporary['t']

        # scores
        temporary['scoring_round'] = 0

        return temporary

    def __above(self):
        """

        :return:
        """

        frame = self.preprocessed.loc[self.preprocessed['liked'] == 1, ]

        # temporary
        temporary: pd.DataFrame = frame.groupby(by='movieId').first()
        temporary = self.__supplement(temporary=temporary.copy())

        return temporary

    def __below(self):
        """

        :return:
        """

        frame = self.preprocessed.loc[self.preprocessed['liked'] == 0, ]

        # temporary
        temporary: pd.DataFrame = frame.groupby(by='movieId').first()
        temporary = self.__supplement(temporary=temporary.copy())

        return temporary

    def exc(self):

        above = self.__above()
        self.logger.info(f'Above:\n {above.head()}')
        self.logger.info(above.info())

        below = self.__below()
        self.logger.info(f'Below:\n {below.head()}')
        self.logger.info(below.info())
