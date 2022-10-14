import logging
import os
import sys


def main():

    logger.info('films')

    # the data
    data = src.data.films.Films().exc()
    logger.info(data.info())

    # preprocessing
    preprocessed = src.data.preprocessing.Preprocessing().exc(data=data, limit=1500)
    logger.info(preprocessed.head())

    # algorithms
    scores = src.algorithms.replay.Replay(data=preprocessed).exc()
    logger.info(f'Running Average Scores: {scores.running}')
    logger.info(f'Cumulative Sums: {scores.cumulative}')


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

    main()
