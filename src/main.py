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
    logger.info('\n\nDATA:\n')
    logger.info(data.info())
    logger.info(f"USERS: {data['userId'].unique().shape}")
    logger.info(f"Users & Occurrences:\n {data['userId'].value_counts()}")

    # preprocessing
    preprocessed = src.data.preprocessing.Preprocessing().exc(data=data, limit=1500)
    logger.info(preprocessed.head())

    # an option
    src.algorithms.initial .Initial(preprocessed=preprocessed).exc()

    # algorithms
    scores = src.algorithms.random.Random(data=preprocessed).exc()
    logger.info(f'Rewards:\n {scores.rewards}')
    logger.info(f'Running Average Scores:\n {scores.running}')
    logger.info(f'Cumulative Sums:\n {scores.cumulative}')

    scores = src.algorithms.ucb.UCB(data=preprocessed).exc()
    logger.info(f'Rewards:\n {scores.rewards}')
    logger.info(f'Running Average Scores:\n {scores.running}')
    logger.info(f'Cumulative Sums:\n {scores.cumulative}')

    scores = src.algorithms.bayesianucb.BayesianUCB(data=preprocessed).exc()
    logger.info(f'Rewards:\n {scores.rewards}')
    logger.info(f'Running Average Scores:\n {scores.running}')
    logger.info(f'Cumulative Sums:\n {scores.cumulative}')
    
    scores = src.algorithms.epsilongreedy.EpsilonGreedy(data=preprocessed, epsilon=0.15).exc()
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
    import src.algorithms.initial
    import src.algorithms.bayesianucb
    import src.algorithms.random
    import src.algorithms.epsilongreedy
    import src.algorithms.ucb

    main()
