"""
Module: main
"""
import logging
import os
import sys


def main():

    logger.info('films')

    # the data
    data = src.data.films.Films().exc()
    logger.info(f'DATA:\n {data.info()}')
    logger.info(f"USERS: {data['userId'].unique().shape}")
    logger.info(data['userId'].value_counts())

    # preprocessing
    preprocessed = src.data.preprocessing.Preprocessing().exc(data=data, limit=1500)
    logger.info(preprocessed.head())

    # algorithms
    scores = src.algorithms.replay.Replay(data=preprocessed).exc()
    logger.info(f'Rewards:\n {scores.rewards}')
    logger.info(f'Running Average Scores:\n {scores.running}')
    logger.info(f'Cumulative Sums:\n {scores.cumulative}')

    scores = src.algorithms.epsilongreedy.EpsilonGreedy(data=preprocessed, epsilon=0.15).exc()
    logger.info(f'Rewards:\n {scores.rewards}')
    logger.info(f'Running Average Scores:\n {scores.running}')
    logger.info(f'Cumulative Sums:\n {scores.cumulative}')

    scores = src.algorithms.ucb.UCB(data=preprocessed).exc()
    logger.info(f'Rewards:\n {scores.rewards}')
    logger.info(f'Running Average Scores:\n {scores.running}')
    logger.info(f'Cumulative Sums:\n {scores.cumulative}')


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
    import src.algorithms.replay
    import src.algorithms.epsilongreedy
    import src.algorithms.ucb

    main()
