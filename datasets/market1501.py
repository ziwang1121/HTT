# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle
class Market1501(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market1501_to_RGBNT201_dark'

    def __init__(self, root='', verbose=True, **kwargs):
        super(Market1501, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self._check_before_run()

        train = self._process_dir_train(self.train_dir, relabel=True)
        query = self._process_dir_test(self.query_dir, relabel=False)
        gallery = self._process_dir_test(self.gallery_dir, relabel=False)


        if verbose:
            print("=> market1501_to_RGBNT201_dark loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery



        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
    
    def _process_dir_train(self, dir_path, relabel=False):
        img_paths_RGB = glob.glob(osp.join(dir_path, 'RGB', '*.jpg'))
        pid_container = set()
        for img_path_RGB in img_paths_RGB:
            jpg_name = img_path_RGB.split('\\')[-1]
            pid = int(jpg_name.split('_')[0][0:4])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path_RGB in img_paths_RGB:
            img = []
            jpg_name = img_path_RGB.split('\\')[-1]
            img_path_NI = osp.join(dir_path, 'NI', jpg_name)
            img_path_TI = osp.join(dir_path, 'TI', jpg_name)
            img.append(img_path_RGB)
            img.append(img_path_NI)
            img.append(img_path_TI)
            pid = int(jpg_name.split('_')[0][0:4])
            camid = int(jpg_name.split('_')[1][1])
            camid -= 1 # index starts from 0

            timeid = int(jpg_name.split('_')[3][1])
            if timeid != 2:
                timeid = 0
            else:
                timeid = 1
            # print(img, pid, camid, timeid)
            if relabel:
                pid = pid2label[pid]
            data.append((img, pid, camid, timeid))
            # print((img, pid, camid, timeid))
        return data
        
    def _process_dir_test(self, dir_path, relabel=False):
            img_paths_RGB = glob.glob(osp.join(dir_path, 'RGB', '*.jpg'))
            pid_container = set()
            for img_path_RGB in img_paths_RGB:
                jpg_name = img_path_RGB.split('\\')[-1]
                pid = int(jpg_name.split('_')[0][0:4])
                pid_container.add(pid)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}

            data = []
            for img_path_RGB in img_paths_RGB:
                img = []
                jpg_name = img_path_RGB.split('\\')[-1]

                
                img_path_NI = osp.join(dir_path, 'NI', jpg_name)
                img_path_TI = osp.join(dir_path, 'TI', jpg_name)
                
                img.append(img_path_RGB)
                img.append(img_path_NI)
                img.append(img_path_TI)
                pid = int(jpg_name.split('_')[0][0:4])
                camid = int(jpg_name.split('_')[1][1])
                camid -= 1 # index starts from 0
                timeid = int(jpg_name.split('_')[3][1])
                # print(img, pid, camid, timeid)
                if relabel:
                    pid = pid2label[pid]
                data.append((img, pid, camid, timeid))
                # print((img, pid, camid, timeid))
            return data
