# generate .npy camera parameter file for human 3.6m dataset
# original code: https://github.com/karfly/learnable-triangulation-pytorch/blob/master/mvn/datasets/human36m_preprocessing/generate-labels-npy-multiview.py
import argparse
import os
from os import path

import h5py

import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--una_dinosauria_root', type=str, default='/home/uyoung/human_pose_estimation/datasets/Human36M/extra/una-dinosauria-data/h36m')
    parser.add_argument('--out_path', type=str, default='/home/uyoung/human_pose_estimation/SMPL/SPIN/data/dataset_extras')

    return parser.parse_args()

args = get_args()
cameras_params = h5py.File(path.join(args.una_dinosauria_root, 'cameras.h5'), 'r')

retval = {}
retval['subject_names'] = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
retval['camera_names'] = ['54138969', '55011271', '58860488', '60457274']

retval['cameras'] = np.empty(
                (len(retval['subject_names']), len(retval['camera_names'])),
                dtype=[
                        ('R', np.float32, (3,3)),
                        ('t', np.float32, (3,1)),
                        ('K', np.float32, (3,3)),
                        ('dist', np.float32, 5)
                      ]
             )

# Fill retval['cameras']
for subject_idx, subject in enumerate(retval['subject_names']):
    for camera_idx, camera in enumerate(retval['camera_names']):
        assert len(cameras_params[subject.replace('S', 'subject')]) == 4
        camera_params = cameras_params[subject.replace('S', 'subject')]['camera%d' % (camera_idx+1)]
        camera_retval = retval['cameras'][subject_idx][camera_idx]

        def camera_array_to_name(array):
            return ''.join(chr(int(x[0])) for x in array)
        assert camera_array_to_name(camera_params['Name']) == camera

        camera_retval['R'] = np.array(camera_params['R']).T
        camera_retval['t'] = -camera_retval['R'] @ camera_params['T']

        camera_retval['K'] = 0
        camera_retval['K'][:2, 2] = camera_params['c'][:, 0]
        camera_retval['K'][0, 0] = camera_params['f'][0]
        camera_retval['K'][1, 1] = camera_params['f'][1]
        camera_retval['K'][2, 2] = 1.0

        camera_retval['dist'][:2] = camera_params['k'][:2, 0]
        camera_retval['dist'][2:4] = camera_params['p'][:, 0]
        camera_retval['dist'][4] = camera_params['k'][2, 0]


np.save(path.join(args.out_path, 'h36m_cameras.npy'), retval)
