"""
Module interface for running the optimal hyperparameter search programs
"""
import logging
import os
import sys

import pandas as pd


def main():
    """

    :return:
    """

    logger.info('hyperparameters')

    '''
    The Data
    '''

    source = src.data.films.Films().exc()
    logger.info(f"USERS: {source['userId'].unique().shape}")

    preprocessed = src.data.preprocessing.Preprocessing().exc(data=source, limit=1500)
    logger.info('\nPreprocessed:\n')
    logger.info(preprocessed.info())

    '''
    Epsilon Greedy
    '''
    histories = src.hyperparameters.epsilongreedy.EpsilonGreedy(data=preprocessed).exc()
    history = pd.concat(histories)
    logger.info(history)

    metrics = history[['epsilon', 'liked', 'MA']].groupby(by='epsilon').agg(average=('liked', 'mean'),
                                                                            N=('liked', 'count'),
                                                                            max_moving_average=('MA', 'last'))
    metrics.reset_index(drop=False, inplace=True)
    logger.info(metrics)

    '''
    Bayesian UCB
    '''
    histories = src.hyperparameters.bayesianucb.BayesianUCB(data=preprocessed).exc()
    history = pd.concat(histories)
    logger.info(history)

    metrics = history[['critical_value', 'liked', 'MA']].groupby(by='critical_value').agg(average=('liked', 'mean'),
                                                                                          N=('liked', 'count'),
                                                                                          max_moving_average=('MA', 'last'))
    metrics.reset_index(drop=False, inplace=True)
    logger.info(metrics)


if __name__ == '__main__':
    # directories/paths
    root = os.getcwd()
    sys.path.append(root)
    sys.path.append(os.path.join(root, 'src'))

    # logging
    logging.basicConfig(level=logging.INFO, format='\n\n%(message)s\n%(asctime)s.%(msecs)03d',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)

    # functions
    import src.data.films
    import src.data.preprocessing
    import src.hyperparameters.epsilongreedy
    import src.hyperparameters.bayesianucb

    main()
