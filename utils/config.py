import os

__proj_dir = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(__proj_dir, 'data')

TRAIN_PIC_DIR = os.path.join(DATA_DIR, 'pic', 'train')
TEST_PIC_DIR = os.path.join(DATA_DIR, 'pic', 'test')
