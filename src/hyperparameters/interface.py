
import os
import sys

import logging


def main():

    logger.info('hyperparameters')


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
    import src.hyperparameters.epsilongreedy

    main()