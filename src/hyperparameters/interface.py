
import logging
import os
import sys

import pandas as pd


def main():

    logger.info('hyperparameters')

    # the data
    data = src.data.films.Films().exc()
    logger.info(f"USERS: {data['userId'].unique().shape}")

    # preprocessing
    preprocessed = src.data.preprocessing.Preprocessing().exc(data=data, limit=1500)
    logger.info('\nPreprocessed:\n')
    logger.info(preprocessed.info())

    # epsilon greedy
    histories = src.hyperparameters.epsilongreedy.EpsilonGreedy(data=preprocessed).exc()
    history = pd.concat(histories)
    logger.info(history)

    metrics = history[['epsilon', 'liked']].groupby(by='epsilon').agg(average=('liked', 'mean'), N=('liked', 'count'))
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

    main()
