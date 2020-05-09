import os
import logging
import time
from pathlib import Path


def init_logger():
    out_dir = './temp_output'
    root_out_dir = Path(out_dir)
    if not root_out_dir.exists():
        print('=> creating {}'.format(root_out_dir))
        root_out_dir.mkdir()
    time_str = time.strftime('%Y%m%d_%H%M')
    log_file = time_str + '.log'
    log_path = root_out_dir / log_file
    log_format = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(log_path), format=log_format)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    console= logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(console)
    return logger

if __name__ == '__main__':
    logger = init_logger()
    logger.info("hellp0")
    logger.debug("aadasda")
