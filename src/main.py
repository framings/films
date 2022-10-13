import logging
import os
import sys


def main():

    logger.info('films')

    # the data
    frame = src.data.films.Films().exc()
    logger.info(frame.info())

    # counts
    src.data.preprocessing.Preprocessing().exc(frame=frame)


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

    main()
