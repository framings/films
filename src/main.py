"""
Module: main
"""
import logging
import os
import sys
import scipy.stats


def main():

    logger.info('films')

    # the data
    source = src.data.films.Films().exc()
    logger.info(f"USERS: {source['userId'].unique().shape}")

    # preprocessing
    preprocessed = src.data.preprocessing.Preprocessing().exc(data=source, limit=1500)
    logger.info('\nPreprocessed:\n')
    logger.info(preprocessed.info())

    # algorithms
    scores = src.algorithms.bayesianucb.BayesianUCB(data=preprocessed, args=args).exc(critical_value=critical_value)
    streams.write(data=scores, path=os.path.join(storage, 'bayesianUCB.csv'))
    logger.info(scores.tail())

    scores = src.algorithms.epsilongreedy.EpsilonGreedy(data=preprocessed, args=args).exc(epsilon=epsilon)
    streams.write(data=scores, path=os.path.join(storage, 'epsilonGreedy.csv'))
    logger.info(scores.tail())

    scores = src.algorithms.exp3.EXP3(data=preprocessed, args=args).exc(gamma=gamma)
    streams.write(data=scores, path=os.path.join(storage, 'EXP3.csv'))
    logger.info(scores)

    scores = src.algorithms.random.Random(data=preprocessed, args=args).exc()
    streams.write(data=scores, path=os.path.join(storage, 'random.csv'))
    logger.info(scores.tail())

    scores = src.algorithms.ucb.UCB(data=preprocessed, args=args).exc(scale=scale)
    streams.write(data=scores, path=os.path.join(storage, 'UCB.csv'))
    logger.info(scores.tail())


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
    import src.functions.directories
    import src.functions.streams
    import src.algorithms.bayesianucb
    import src.algorithms.random
    import src.algorithms.epsilongreedy
    import src.algorithms.ucb
    import src.algorithms.exp3
    import config

    # configurations
    args = config.Config().models()
    warehouse = config.Config().warehouse

    # storing outputs
    storage = os.path.join(warehouse, 'scores')
    directories = src.functions.directories.Directories()
    directories.cleanup(path=storage)
    directories.create(path=storage)

    # instances
    streams = src.functions.streams.Streams()

    # bayesian UCB: for a two-tailed-test 92% confidence interval
    alpha = 1 - 0.92
    critical_value = scipy.stats.norm.ppf(q=(1 - alpha/2))

    # epsilon greedy
    epsilon = 0.1

    # EXP3
    gamma = 0.1

    # UCB
    scale = 2.0

    main()
