import glob
import re
import os.path as osp

from .bases import BaseImageDataset
from random import sample

class RGBNT201(BaseImageDataset):


    dataset_dir = 'RGBNT201'

    def __init__(self, root='', verbose=True, **kwargs):
        super(RGBNT201, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train_171')
        self.query_dir = osp.join(self.dataset_dir, 'test')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')

        self._check_before_run()

        train = self._process_dir_train(self.train_dir, relabel=True)
        query = self._process_dir_test(self.query_dir, relabel=False)
        gallery = self._process_dir_test(self.gallery_dir, relabel=False)
        query_ttt = self._process_dir_test_ttt(self.query_dir, relabel=False)
        gallery_ttt = self._process_dir_test_ttt(self.gallery_dir, relabel=False)



        if verbose:
            print("=> RGBNT201 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery
        self.query_ttt = query_ttt
        self.gallery_ttt = gallery_ttt


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
            pid = int(jpg_name.split('_')[0][0:6])
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
            pid = int(jpg_name.split('_')[0][0:6])
            camid = int(jpg_name.split('_')[1][3])
            camid -= 1 # index starts from 0

            timeid = int(jpg_name.split('_')[2])
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
                pid = int(jpg_name.split('_')[0][0:6])
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
                pid = int(jpg_name.split('_')[0][0:6])
                camid = int(jpg_name.split('_')[1][3])
                camid -= 1 # index starts from 0
                timeid = int(jpg_name.split('_')[2])
                # print(img, pid, camid, timeid)
                if relabel:
                    pid = pid2label[pid]
                data.append((img, pid, camid, timeid))
                # print((img, pid, camid, timeid))
            return data


    def _process_dir_test_ttt(self, dir_path, relabel=False):
            img_paths_RGB = glob.glob(osp.join(dir_path, 'RGB', '*.jpg'))
            pid_container = set()
            for img_path_RGB in img_paths_RGB:
                jpg_name = img_path_RGB.split('\\')[-1]
                pid = int(jpg_name.split('_')[0][0:6])
                pid_container.add(pid)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}

            data = []
            # length = len(img_paths_RGB)
            # img_paths_ttt = sample(img_paths_RGB, int(length/2)*2)
            for img_path_RGB in img_paths_RGB:
                img = []
                jpg_name = img_path_RGB.split('\\')[-1]

                
                img_path_NI = osp.join(dir_path, 'NI', jpg_name)
                img_path_TI = osp.join(dir_path, 'TI', jpg_name)
                
                img.append(img_path_RGB)
                img.append(img_path_NI)
                img.append(img_path_TI)
                pid = int(jpg_name.split('_')[0][0:6])
                camid = int(jpg_name.split('_')[1][3])
                camid -= 1 # index starts from 0
                timeid = int(jpg_name.split('_')[2])
                # print(img, pid, camid, timeid)
                if relabel:
                    pid = pid2label[pid]
                data.append((img, pid, camid, timeid))
                # print((img, pid, camid, timeid))
            return data


