from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        if isinstance(img_path,list):
            # print(img_path)
            img_1 = read_image(img_path[0])
            if self.transform is not None:
                img_1 = self.transform(img_1)

            a,b,c=img_1.shape
            tmp=torch.zeros(a,b,c)

            if not os.path.exists(img_path[1]):
                img_2 = tmp
            else:
                img_2 = read_image(img_path[1])
                if self.transform is not None:
                    img_2 = self.transform(img_2)

            if not os.path.exists(img_path[2]):
                img_3 = tmp
            else:
                img_3 = read_image(img_path[2])
                if self.transform is not None:
                    img_3 = self.transform(img_3)

            return img_1, img_2, img_3, pid, camid, trackid, img_path[0].split('/')[-1]
        else:
            # print(img_path)
            img = read_image(img_path)
            if img.size == (768, 128):
                # print(1)
                #img1 = img
                img_1 = img.crop((0, 0, 256, 128))
                img_2 = img.crop((256, 0, 512, 128))
                img_3 = img.crop((256, 0, 768, 128))
            if img.size == (512, 128):
                # print(2)
                #img1 = img
                img_1 = img.crop((0, 0, 256, 128))
                img_2 = img.crop((256, 0, 512, 128))
                img_3 = img.crop((256, 0, 512, 128))
            if self.transform is not None:
                img_1 = self.transform(img_1)
                img_2 = self.transform(img_2)
                img_3 = self.transform(img_3)

            return img_1, img_2, img_3, pid, camid, trackid, img_path[0].split('/')[-1]