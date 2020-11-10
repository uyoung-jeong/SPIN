# prepro only human3.6m dataset

import argparse
import config as cfg
from datasets.preprocess import h36m_extract, h36m_extract_custom

parser = argparse.ArgumentParser()
parser.add_argument('--train_files', default=False, action='store_true', help='Extract files needed for training')
parser.add_argument('--eval_files', default=False, action='store_true', help='Extract files needed for evaluation')

if __name__ == '__main__':
    args = parser.parse_args()

    # define path to store extra files
    out_path = cfg.DATASET_NPZ_PATH
    openpose_path = cfg.OPENPOSE_PATH

    if args.train_files:
        pass

    if args.eval_files:
        print('preprocess eval files')
        h36m_extract_custom(cfg.H36M_ROOT, out_path, protocol=1, extract_img=True)
        h36m_extract_custom(cfg.H36M_ROOT, out_path, protocol=2, extract_img=False)

    print('Finished h36m preprocessing')
