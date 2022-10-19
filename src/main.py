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
    logger.info(f"USERS: {data['userId'].unique().shape}")

    # preprocessing
    preprocessed = src.data.preprocessing.Preprocessing().exc(data=data, limit=1500)
    logger.info('\nPreprocessed:\n')
    logger.info(preprocessed.info())

    # an option
    initial = src.functions.initial.Initial(preprocessed=preprocessed).exc()
    logger.info(f'Initial: {initial.shape}')

    # algorithms
    scores = src.algorithms.random.Random(data=preprocessed).exc()
    logger.info(f'Random\nRewards: {len(scores.rewards)}')
    logger.info(f'Cumulative Sums:\n{scores.cumulative}')
    logger.info(f'Running Average Scores:\n{scores.running}')

    scores = src.algorithms.epsilongreedy.EpsilonGreedy(data=preprocessed, args=args).exc(epsilon=0.1)
    logger.info(f'Epsilon Greedy\nRewards: {len(scores.rewards)}')
    logger.info(f'Cumulative Sums:\n{scores.cumulative}')
    logger.info(f'Running Average Scores:\n{scores.running}')

    scores = src.algorithms.ucb.UCB(data=preprocessed).exc()
    logger.info(f'UCB\nRewards: {len(scores.rewards)}')
    logger.info(f'Cumulative Sums:\n{scores.cumulative}')
    logger.info(f'Running Average Scores:\n{scores.running}')

    scores = src.algorithms.bayesianucb.BayesianUCB(data=preprocessed).exc()
    logger.info(f'Bayesian UCB\nRewards: {len(scores.rewards)}')
    logger.info(f'Cumulative Sums:\n{scores.cumulative}')
    logger.info(f'Running Average Scores:\n{scores.running}')
    

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
    import src.functions.initial
    import src.algorithms.bayesianucb
    import src.algorithms.random
    import src.algorithms.epsilongreedy
    import src.algorithms.ucb
    import config

    args = config.Config().models()

    main()
