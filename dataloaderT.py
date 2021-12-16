from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
import torch
from torch.utils.data import Dataset, sampler,DataLoader
from torchvision import transforms
import skimage.transform

from kitti_utils import generate_depth_map

def _is_pil_image(img):
    return isinstance(img, Image.Image)
def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})
def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class Seq2DataLoader(object):
    def __init__(self,args,mode):
        if mode == "train":
            self.training_samples = Monodataset(args,mode,transform=ToTensor(mode))
            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=False,
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   )
        elif mode == "online_eval":
            self.testing_samples = Monodataset(args,mode,transform=ToTensor(mode))
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler = None
                                   )
        elif mode == "test":
            self.testing_samples = Monodataset(args,mode,transform=ToTensor(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)
        
        
class Monodataset(Dataset):
    def __init__(self,args,mode,transform=None,is_for_online_eval=False):
        super(Monodataset,self).__init__()
        self.args = args
        self.data_path = args.data_path
        # self.filenames = filenames
        self.dataset = args.dataset
        
        if mode == 'online_eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.height = args.input_height
        self.width = args.input_width
        # self.num_scales = num_scales
        self.interp = Image.ANTIALIAS
        self.frame_idxs = args.frame_ids
        self.is_train = True

        img_ext = '.png' if args.png else '.jpg'
        self.img_ext = img_ext
        self.loader = pil_loader
        self.to_tensorM = transforms.ToTensor()  #此处需要修改 改成自定义的Totensor

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

        self.mode = mode
        self.transform = transform
        self.to_tensorA = ToTensor
        self.is_for_online_eval = is_for_online_eval
        self.do_kb_crop = args.do_kb_crop
        self.do_color_aug = args.do_color_aug
        #随机修改图片的光照、对比度、饱和度和色调 
        #不同的torch vision版本有不同的参数
        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        # 因为monodepth有Multi-scale Estimation 所以在此处进行了resize的准备
        # self.resize = {}
        # for i in range(self.num_scales):
        #     s = 2 ** i
        #     self.resize[i] = transforms.Resize((self.height // s, self.width // s),
        #                                        interpolation=self.interp)

        self.load_depth = self.check_depth()
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        sample_path = self.filenames[index]  # 2011_09_26/2011_09_26_drive_0022_sync 473 r
        line = self.filenames[index].split() # [0]2011_09_26/2011_09_26_drive_0022_sync  [1]473 [2]r
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])       # 场景中的帧号
        else:
            frame_index = 0                  # 否则是第一帧

        if len(line) == 3:                   # 视图：左 or 右
            side = line[2]
        else:
            side = None
        inputs = {}
        inputs_eval = {}
        if self.mode == "train":
            for i in self.frame_idxs:
                # image_path = self.get_image_path(self,folder, frame_index, side) # 'E:\\kitti\\sync\\2011_09_30/2011_09_30_drive_0034_sync\\image_02/data\\0000000865.png' 
                inputs[("color", i)] = self.loader(self.get_image_path(folder, frame_index + i, side))
                if self.do_kb_crop is True:
                    height = inputs[("color", i)].height
                    width = inputs[("color", i)].width
                    top_margin = int(height - 352)
                    left_margin = int((width - 1216) / 2)
                    # depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                    inputs[("color", i)] = inputs[("color", i)].crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                # if self.args.do_random_rotate is True:
                #     random_angle = (random.random() - 0.5) * 2 * self.args.degree
                #     image = self.rotate_image(image, random_angle)
                #     depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
                inputs[("color", i)] = np.asarray(inputs[("color", i)], dtype=np.float32) / 255.0
                inputs[("color", i)] = self.random_crop(inputs[("color", i)], self.height, self.width)
                inputs[("color", i)] = self.train_preprocess(inputs[("color", i)])
                if i == 0:
                    currentImage = inputs[("color", i)]
                elif i == -1 :
                    forwardImage = inputs[("color", i)]
                else:
                    backwardImage = inputs[("color", i)]
            if self.load_depth:
                inputs["depth_gt"] = self.get_depth(folder, frame_index, side)  
                inputs["depth_gt"] = np.expand_dims(inputs["depth_gt"], 0)
                inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))
            inputs["depth_gt"] = np.asarray(inputs["depth_gt"], dtype=np.float32)
            inputs["depth_gt"] = np.expand_dims(inputs["depth_gt"], axis=2)

            if self.dataset == 'nyu':
                inputs["depth_gt"] = inputs["depth_gt"] / 1000.0
            else:
                inputs["depth_gt"] = inputs["depth_gt"] / 256.0

            K = self.K.copy()
            K[0, :] *= self.width # // (2 ** scale)
            K[1, :] *= self.height# // (2 ** scale)
            inv_K = np.linalg.pinv(K)
            inputs[("K", 0)] = torch.from_numpy(K)
            inputs[("inv_K", 0)] = torch.from_numpy(inv_K)  
            sample = {'current_image': currentImage,'forward_image':forwardImage,'backward_image':backwardImage,'depth_gt':inputs["depth_gt"],'K':inputs[("K", 0)],'inv_k':inputs[("inv_K", 0)]}
        else:
            if self.mode == "online_eval":
                self.data_path = self.args.data_path_eval
            else:
                self.data_path = self.args.data_path
            for i in self.frame_idxs:
                # image_path = self.get_image_path(folder,frame_index + i, side)
                inputs_eval[("color", i)] = np.asarray(Image.open(self.get_image_path(folder,frame_index + i, side)), dtype=np.float32) / 255.0    
                if self.args.do_kb_crop is True:
                    height = inputs_eval[("color", i)].shape[0]
                    width = inputs_eval[("color", i)].shape[1]
                    top_margin = int(height - 352)
                    left_margin = int((width - 1216) / 2)
                    inputs_eval[("color", i)] = inputs_eval[("color", i)][top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
            if self.mode == 'online_eval':
                try:
                    inputs_eval["depth_gt"] = self.get_depth(folder, frame_index, side)  
                    inputs_eval["depth_gt"] = np.expand_dims(inputs_eval["depth_gt"], 0)
                    inputs_eval["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    # print('Missing gt for {}'.format(image_path))

                if has_valid_depth:
                    inputs_eval["depth_gt"] = np.asarray(inputs_eval["depth_gt"], dtype=np.float32)
                    inputs_eval["depth_gt"] = np.expand_dims(inputs_eval["depth_gt"], axis=2)
                    if self.args.dataset == 'nyu':
                        inputs_eval["depth_gt"] = inputs_eval["depth_gt"] / 1000.0
                    else:
                        inputs_eval["depth_gt"] = inputs_eval["depth_gt"] / 256.0

            if self.args.do_kb_crop is True:
                height = inputs_eval[("color", 0)].shape[0]
                width = inputs_eval[("color", 0)].shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                inputs_eval[("color", 0)] = inputs_eval[("color", 0)][top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                if self.mode == 'online_eval' and has_valid_depth:
                    inputs_eval["depth_gt"] = inputs_eval["depth_gt"][top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
            for i in self.frame_idxs:
                if i == 0:
                    currentImage = inputs[("color", i)]
                elif i == -1 :
                    forwardImage = inputs[("color", i)]
                else:
                    backwardImage = inputs[("color", i)]
            if self.mode == 'online_eval':
                sample = {'current_image': currentImage,'forward_image':forwardImage,'backward_image':backwardImage,'depth_gt':inputs["depth_gt"], 'has_valid_depth': has_valid_depth,
                          'image_path': self.get_image_path(folder,frame_index + i, side)}
            else:
                sample = {'current_image': currentImage,'forward_image':forwardImage,'backward_image':backwardImage,'depth_gt':inputs["depth_gt"]}
        
        return sample
    # 三个等待子类实现的成员函数    
    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    # def get_depth(self, folder, frame_index, side, do_flip):
    #     calib_path = os.path.join(self.data_path, folder.split("/")[0])

    #     velo_filename = os.path.join(
    #         self.data_path,
    #         folder,
    #         "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

    #     depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
    #     depth_gt = skimage.transform.resize(
    #         depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

    #     if do_flip:
    #         depth_gt = np.fliplr(depth_gt)

    #     return depth_gt
    # 获取图像的工具函数
    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext) # '0000000865.png'
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path
    def get_depth(self, folder, frame_index, side):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))
        # 'E:\\kitti\\sync\\2011_09_30/2011_09_30_drive_0033_sync\\velodyne_points/data/0000001048.bin'
        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        # if do_flip:
        #     depth_gt = np.fliplr(depth_gt)

        return depth_gt
    def random_crop(self, img, height, width):

        assert img.shape[0] >= height
        assert img.shape[1] >= width
        # assert img.shape[0] == depth.shape[0]
        # assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        # depth = depth[y:y + height, x:x + width, :]
        return img
    def train_preprocess(self, image):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            # depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image
    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug
class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        image= sample['image']
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == 'test':
            return {'image': image}

        
        if self.mode == 'train':
    
            return {'image': image}
        else:
            has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'has_valid_depth': has_valid_depth,
                    'image_path': sample['image_path'], 'depth_path': sample['depth_path']}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img



