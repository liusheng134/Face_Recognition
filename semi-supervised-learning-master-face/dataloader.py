import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from dataset import LFW, CFP_FP, AgeDB30, CASIAWebFace, MS1M
import numpy as np
import lmdb
import os.path as osp
import pyarrow as pa
import random
import six
from PIL import Image
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)

class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None,teacher_transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

        self.transform = transform
        self.teacher_transform = teacher_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = loads_pyarrow(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img_student = self.transform(img)

        if self.teacher_transform is not None:
            img_teacher = self.teacher_transform(img)

        im2arr = np.array(img_student)
        im2arr_teacher = np.array(img_teacher)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        return [im2arr,im2arr_teacher], target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
        	# (x,y)表示方形补丁的中心位置
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        print(img)
        return img

class Cutout_half_face(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        probability = np.random.uniform(0, 1)
        if probability < 0.9:
            h = img.size(1)
            w = img.size(2)

            mask = np.ones((h, w), np.float32)

            for n in range(self.n_holes):
                # (x,y)表示方形补丁的中心位置
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)

                mask[y1: y2, x1: x2] = 0.

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask
        else:
            h = img.size(1)
            w = img.size(2)

            mask = np.ones((h, w), np.float32)

            for n in range(self.n_holes):
                # (x,y)表示方形补丁的中心位置
                y = 56
                x = 28
                
                y1 = np.clip(y - 112 // 2, 0, h)
                y2 = np.clip(y + 112 // 2, 0, h)
                x1 = np.clip(x - 56 // 2, 0, w)
                x2 = np.clip(x + 56 // 2, 0, w)

                mask[y1: y2, x1: x2] = 0.

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            
            img = img * mask
            # print(img)
        return img

class Cutout_half_face(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        probability = np.random.uniform(0, 1)
        if probability < 0.9:
            h = img.size(1)
            w = img.size(2)

            mask = np.ones((h, w), np.float32)

            for n in range(self.n_holes):
                # (x,y)表示方形补丁的中心位置
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)

                mask[y1: y2, x1: x2] = 0.

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask
        else:
            h = img.size(1)
            w = img.size(2)

            mask = np.ones((h, w), np.float32)

            for n in range(self.n_holes):
                # (x,y)表示方形补丁的中心位置
                y = 56
                x = 28
                
                y1 = np.clip(y - 112 // 2, 0, h)
                y2 = np.clip(y + 112 // 2, 0, h)
                x1 = np.clip(x - 56 // 2, 0, w)
                x2 = np.clip(x + 56 // 2, 0, w)

                mask[y1: y2, x1: x2] = 0.

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            
            img = img * mask
            # print(img)
        return img

def load_data_train_fix_from_lmdb(batch_size, dataset = 'Faces_emore'):
    
    transform = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        # transforms.Resize((120, 120), interpolation=3),
        # transforms.RandomCrop(112),
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])  # range [0.0, 1.0] -> [-1.0,1.0]
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.Resize((120, 120), interpolation=3),
        transforms.RandomCrop(112),
        transforms.ToTensor(),
        Cutout_half_face(n_holes=2, length=16),
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    root = '/data/face_dataset/LFW/lfw_align_112'
    file_list = '/data/face_dataset/LFW/pairs.txt'
    dataset_LFW = LFW(root, file_list, transform=transform)
    
    root = '/data/face_dataset/shunde'
    file_list = '/data/face_dataset/shunde/pairs.txt'
    privacy = CFP_FP(root, file_list, transform=transform)
    
    root = '/data/face_dataset/CFP-FP/CFP_FP_aligned_112'
    file_list = '/data/face_dataset/CFP-FP/cfp_fp_pair.txt'
    dataset_CFP_FP = CFP_FP(root, file_list, transform=transform)
        
    root = '/data/face_dataset/AgeDB-30/agedb30_align_112'
    file_list = '/data/face_dataset/AgeDB-30/agedb_30_pair.txt'
    dataset_AgeDB30 = AgeDB30(root, file_list, transform=transform)   
    
    if dataset == 'CASIA':
        
        root = 'data_set/CASIA_Webface_Image'
        file_list = 'data_set/CASIA_Webface_Image/webface_align_112.txt'
        dataset_train = CASIAWebFace(root, file_list, transform=transform)
        
    elif dataset == 'Faces_emore':

        path = "/data/face_dataset/ms1m_lmdb/ms1m.lmdb"
        dataset_train = ImageFolderLMDB(path, train_transform,transform) 
        
    else:
        raise NameError('no training data exist!')

    dataset_size = len(dataset_train)
    
    dataloaders = {'train_dataset': DataLoaderX(dataset_train, batch_size=batch_size,pin_memory=True, shuffle=True,num_workers=16),
                    'privacy': DataLoaderX(privacy, batch_size=batch_size,pin_memory=True, shuffle=False, num_workers=4),
                    'LFW': DataLoaderX(dataset_LFW, batch_size=batch_size,pin_memory=True, shuffle=False,num_workers=4),
                    'CFP_FP': DataLoaderX(dataset_CFP_FP, batch_size=batch_size,pin_memory=True, shuffle=False,num_workers=4),
                    'AgeDB30': DataLoaderX(dataset_AgeDB30, batch_size=batch_size,pin_memory=True, shuffle=False,num_workers=4)}

    dataset = {'train_dataset': dataset_train,'LFW': dataset_LFW,'privacy':privacy,
               'CFP_FP': dataset_CFP_FP, 'AgeDB30': dataset_AgeDB30}
    
    dataset_sizes = {'train': len(dataset_train),'LFW': len(dataset_LFW),'privacy':len(privacy),
                     'CFP_FP': len(dataset_CFP_FP), 'AgeDB30': len(dataset_AgeDB30)}
    
    print('training and validation data loaded')
    
    return dataloaders, dataset_sizes, dataset

def load_glint_data_train_from_lmdb(batch_size, dataset = 'Faces_emore'):
    
    transform = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        # transforms.Resize((120, 120), interpolation=3),
        # transforms.RandomCrop(112),
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])  # range [0.0, 1.0] -> [-1.0,1.0]
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.Resize((120, 120), interpolation=3),
        transforms.RandomCrop(112),
        transforms.ToTensor(),
        Cutout(n_holes=2, length=16),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    root = '/data/face_dataset/LFW/lfw_align_112'
    file_list = '/data/face_dataset/LFW/pairs.txt'
    dataset_LFW = LFW(root, file_list, transform=transform)
    
    root = '/data/face_dataset/shunde'
    file_list = '/data/face_dataset/shunde/pairs.txt'
    privacy = CFP_FP(root, file_list, transform=transform)
    
    root = '/data/face_dataset/CFP-FP/CFP_FP_aligned_112'
    file_list = '/data/face_dataset/CFP-FP/cfp_fp_pair.txt'
    dataset_CFP_FP = CFP_FP(root, file_list, transform=transform)
        
    root = '/data/face_dataset/AgeDB-30/agedb30_align_112'
    file_list = '/data/face_dataset/AgeDB-30/agedb_30_pair.txt'
    dataset_AgeDB30 = AgeDB30(root, file_list, transform=transform)   
    
    if dataset == 'CASIA':
        
        root = 'data_set/CASIA_Webface_Image'
        file_list = 'data_set/CASIA_Webface_Image/webface_align_112.txt'
        dataset_train = CASIAWebFace(root, file_list, transform=transform)
        
    elif dataset == 'Faces_emore':

        path = "/data/face_dataset/source_glint.lmdb"
        dataset_train = ImageFolderLMDB(path, train_transform) 
        
    else:
        raise NameError('no training data exist!')

    dataset_size = len(dataset_train)
    
    dataloaders = {'train_dataset': DataLoaderX(dataset_train, batch_size=batch_size,pin_memory=True, shuffle=True,num_workers=16),
                    'privacy': DataLoaderX(privacy, batch_size=batch_size,pin_memory=True, shuffle=False, num_workers=4),
                    'LFW': DataLoaderX(dataset_LFW, batch_size=batch_size,pin_memory=True, shuffle=False,num_workers=4),
                    'CFP_FP': DataLoaderX(dataset_CFP_FP, batch_size=batch_size,pin_memory=True, shuffle=False,num_workers=4),
                    'AgeDB30': DataLoaderX(dataset_AgeDB30, batch_size=batch_size,pin_memory=True, shuffle=False,num_workers=4)}

    dataset = {'train_dataset': dataset_train,'LFW': dataset_LFW,'privacy':privacy,
               'CFP_FP': dataset_CFP_FP, 'AgeDB30': dataset_AgeDB30}
    
    dataset_sizes = {'train': len(dataset_train),'LFW': len(dataset_LFW),'privacy':len(privacy),
                     'CFP_FP': len(dataset_CFP_FP), 'AgeDB30': len(dataset_AgeDB30)}
    
    print('training and validation data loaded')
    
    return dataloaders, dataset_sizes, dataset

def load_data_with_dali(batch_size ,args ,dataset = 'Faces_emore'):
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])  
    
    root = '/data/face_dataset/LFW/lfw_align_112'
    file_list = '/data/face_dataset/LFW/pairs.txt'
    dataset_LFW = LFW(root, file_list, transform=transform)
    
    root = '/data/face_dataset/CFP-FP/CFP_FP_aligned_112'
    file_list = '/data/face_dataset/CFP-FP/cfp_fp_pair.txt'
    dataset_CFP_FP = CFP_FP(root, file_list, transform=transform)
        
    root = '/data/face_dataset/AgeDB-30/agedb30_align_112'
    file_list = '/data/face_dataset/AgeDB-30/agedb_30_pair.txt'
    dataset_AgeDB30 = AgeDB30(root, file_list, transform=transform)  
    
    if dataset == 'CASIA':
        root = 'data_set/CASIA_Webface_Image'
        file_list = 'data_set/CASIA_Webface_Image/webface_align_112.txt'
        dataset_train = CASIAWebFace(root, file_list, transform=transform)
    elif dataset == 'Faces_emore':
        train_loader = get_imagenet_iter_dali(type='train', image_dir='/data/face_dataset/imgs', batch_size=batch_size,
                                          num_threads=4, device_id=args.local_rank, num_gpus=2,world_size=1)

        root = '/data/face_dataset/imgs/'
        file_list = '/data/face_dataset/imgs/faces_emore_align_112.txt'
        dataset_train = MS1M(root, file_list, transform=transform)
    else:
        raise NameError('no training data exist!')

    dataloaders = {'train_dataset': train_loader,
                   'LFW': data.DataLoader(dataset_LFW, batch_size=batch_size,pin_memory=True, shuffle=False),
                   'CFP_FP': data.DataLoader(dataset_CFP_FP, batch_size=batch_size,pin_memory=True, shuffle=False),
                   'AgeDB30': data.DataLoader(dataset_AgeDB30, batch_size=batch_size,pin_memory=True, shuffle=False)}
    
    dataset = {'train_dataset': dataset_train,'LFW': dataset_LFW,
               'CFP_FP': dataset_CFP_FP, 'AgeDB30': dataset_AgeDB30}
    
    dataset_sizes = {'train': len(dataset_train),'LFW': len(dataset_LFW),
                     'CFP_FP': len(dataset_CFP_FP), 'AgeDB30': len(dataset_AgeDB30)}
    
    print('training and validation data loaded')
    
    return dataloaders, dataset_sizes, dataset


def load_data(batch_size, dataset = 'Faces_emore'):
    
    transform = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        # transforms.Resize((120, 120), interpolation=3),
        # transforms.RandomCrop(112),
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])  # range [0.0, 1.0] -> [-1.0,1.0]
    
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    root = '/data/face_dataset/LFW/lfw_align_112'
    file_list = '/data/face_dataset/LFW/pairs.txt'
    dataset_LFW = LFW(root, file_list, transform=transform)
    
    root = '/data/face_dataset/CFP-FP/CFP_FP_aligned_112'
    file_list = '/data/face_dataset/CFP-FP/cfp_fp_pair.txt'
    dataset_CFP_FP = CFP_FP(root, file_list, transform=transform)
        
    root = '/data/face_dataset/AgeDB-30/agedb30_align_112'
    file_list = '/data/face_dataset/AgeDB-30/agedb_30_pair.txt'
    dataset_AgeDB30 = AgeDB30(root, file_list, transform=transform)  
    
    if dataset == 'CASIA':
        
        root = 'data_set/CASIA_Webface_Image'
        file_list = 'data_set/CASIA_Webface_Image/webface_align_112.txt'
        dataset_train = CASIAWebFace(root, file_list, transform=transform)
        
    elif dataset == 'Faces_emore':

        root = '/data/face_dataset/imgs'
        file_list = '/data/face_dataset/imgs/faces_emore_align_112_tiny10.txt'
        dataset_train = MS1M(root, file_list, transform=train_transform) 
    else:
        raise NameError('no training data exist!')

    dataset_size = len(dataset_train)
    train_size = int(0.8 * dataset_size)
    valid_size = dataset_size - train_size
    train_subset, valid_subset = torch.utils.data.random_split(dataset_train, [train_size, valid_size])
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_subset_sampler = torch.utils.data.distributed.DistributedSampler(train_subset)
    valid_subset_sampler = torch.utils.data.distributed.DistributedSampler(valid_subset)
    # 'train_dataset': data.DataLoader(dataset_train, batch_size=batch_size,  sampler=train_sampler),
    dataloaders = {'train_dataset': data.DataLoader(dataset_train, batch_size=batch_size, num_workers=2,pin_memory=True, sampler=train_sampler),
                   'train_subset': data.DataLoader(train_subset, batch_size=batch_size, num_workers=2,pin_memory=True, sampler=train_subset_sampler),
                   'valid_subset': data.DataLoader(valid_subset, batch_size=batch_size,  num_workers=2,pin_memory=True, sampler=valid_subset_sampler),
                   'LFW': data.DataLoader(dataset_LFW, batch_size=batch_size,pin_memory=True, shuffle=False),
                   'CFP_FP': data.DataLoader(dataset_CFP_FP, batch_size=batch_size,pin_memory=True, shuffle=False),
                   'AgeDB30': data.DataLoader(dataset_AgeDB30, batch_size=batch_size,pin_memory=True, shuffle=False)}
    
    dataset = {'train_dataset': dataset_train,'train_subset': train_subset,'valid_subset': valid_subset,'LFW': dataset_LFW,
               'CFP_FP': dataset_CFP_FP, 'AgeDB30': dataset_AgeDB30}
    
    dataset_sizes = {'train': len(dataset_train),'train_subset':len(train_subset),'valid_subset':len(valid_subset),'LFW': len(dataset_LFW),
                     'CFP_FP': len(dataset_CFP_FP), 'AgeDB30': len(dataset_AgeDB30)}
    
    print('training and validation data loaded')
    
    return dataloaders, dataset_sizes, dataset


    
    transform = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        # transforms.Resize((120, 120), interpolation=3),
        # transforms.RandomCrop(112),
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])  # range [0.0, 1.0] -> [-1.0,1.0]
    
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((120, 120), interpolation=3),
        transforms.RandomCrop(112),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    root = '/data/face_dataset/LFW/lfw_align_112'
    file_list = '/data/face_dataset/LFW/pairs.txt'
    dataset_LFW = LFW(root, file_list, transform=transform)
    
    root = '/data/face_dataset/CFP-FP/CFP_FP_aligned_112'
    file_list = '/data/face_dataset/CFP-FP/cfp_fp_pair.txt'
    dataset_CFP_FP = CFP_FP(root, file_list, transform=transform)
        
    root = '/data/face_dataset/AgeDB-30/agedb30_align_112'
    file_list = '/data/face_dataset/AgeDB-30/agedb_30_pair.txt'
    dataset_AgeDB30 = AgeDB30(root, file_list, transform=transform)  
    
    if dataset == 'CASIA':
        
        root = 'data_set/CASIA_Webface_Image'
        file_list = 'data_set/CASIA_Webface_Image/webface_align_112.txt'
        dataset_train = CASIAWebFace(root, file_list, transform=transform)
        
    elif dataset == 'Faces_emore':

        root = '/data/face_dataset/imgs'
        file_list = '/data/face_dataset/imgs/faces_emore_align_112_tiny10.txt'
        dataset_train = MS1M(root, file_list, transform=train_transform) 
        # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    
    else:
        raise NameError('no training data exist!')

    dataset_size = len(dataset_train)
    train_size = int(0.8 * dataset_size)
    valid_size = dataset_size - train_size
    train_subset, valid_subset = torch.utils.data.random_split(dataset_train, [train_size, valid_size])
    
    # 'train_dataset': data.DataLoader(dataset_train, batch_size=batch_size,  sampler=train_sampler),
    dataloaders = {'train_dataset': DataLoaderX(dataset_train, batch_size=batch_size, num_workers=16,pin_memory=True, shuffle=True),
                   'train_subset': data.DataLoader(train_subset, batch_size=batch_size, shuffle=True),
                   'valid_subset': data.DataLoader(valid_subset, batch_size=batch_size,  shuffle=True),
                   'LFW': data.DataLoader(dataset_LFW, batch_size=batch_size, shuffle=False),
                   'CFP_FP': data.DataLoader(dataset_CFP_FP, batch_size=batch_size, shuffle=False),
                   'AgeDB30': data.DataLoader(dataset_AgeDB30, batch_size=batch_size, shuffle=False)}
    
    dataset = {'train_dataset': dataset_train,'train_subset': train_subset,'valid_subset': valid_subset,'LFW': dataset_LFW,
               'CFP_FP': dataset_CFP_FP, 'AgeDB30': dataset_AgeDB30}
    
    dataset_sizes = {'train': len(dataset_train),'train_subset':len(train_subset),'valid_subset':len(valid_subset),'LFW': len(dataset_LFW),
                     'CFP_FP': len(dataset_CFP_FP), 'AgeDB30': len(dataset_AgeDB30)}
    
    print('training and validation data loaded')
    
    return dataloaders, dataset_sizes, dataset