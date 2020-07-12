import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from dataloader import LFW, CFP_FP, AgeDB30, CASIAWebFace, MS1M
import numpy as np
import lmdb
import os.path as osp
import pyarrow as pa
import six
from PIL import Image

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target

def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)

class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

        self.transform = transform
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
            img = self.transform(img)

        im2arr = np.array(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        return im2arr, target

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

        return img

def load_data_train_fix(batch_size, dataset = 'Faces_emore'):
    
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
    
    root = '/root/faces_emore/LFW/lfw_align_112/'
    file_list = '/root/faces_emore/LFW/pairs.txt'
    dataset_LFW = LFW(root, file_list, transform=transform)
    
    root = '/root/faces_emore/cfp_fp/'
    file_list = '/root/faces_emore/cfp_fp/cfp_fp_pair.txt'
    dataset_CFP_FP = CFP_FP(root, file_list, transform=transform)
        
    root = '/root/faces_emore/agedb_30/'
    file_list = '/root/faces_emore/agedb_30/agedb_30_pair.txt'
    dataset_AgeDB30 = AgeDB30(root, file_list, transform=transform)  
    
    if dataset == 'CASIA':
        
        root = 'data_set/CASIA_Webface_Image'
        file_list = 'data_set/CASIA_Webface_Image/webface_align_112.txt'
        dataset_train = CASIAWebFace(root, file_list, transform=transform)
        
    elif dataset == 'Faces_emore':

        root = '/root/faces_emore/imgs/'
        file_list = '/root/faces_emore/imgs/faces_emore_align_112.txt'
        dataset_train = MS1M(root, file_list, transform=train_transform) 
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    else:
        raise NameError('no training data exist!')

    dataset_size = len(dataset_train)
    
    
    dataloaders = {'train_dataset': data.DataLoader(dataset_train, batch_size=batch_size ,pin_memory=True, sampler=train_sampler),
                   'LFW': data.DataLoader(dataset_LFW, batch_size=batch_size,pin_memory=True, shuffle=False),
                   'CFP_FP': data.DataLoader(dataset_CFP_FP, batch_size=batch_size,pin_memory=True, shuffle=False),
                   'AgeDB30': data.DataLoader(dataset_AgeDB30, batch_size=batch_size,pin_memory=True, shuffle=False)}
    
    dataset = {'train_dataset': dataset_train,'LFW': dataset_LFW,
               'CFP_FP': dataset_CFP_FP, 'AgeDB30': dataset_AgeDB30}
    
    dataset_sizes = {'train': len(dataset_train),'LFW': len(dataset_LFW),
                     'CFP_FP': len(dataset_CFP_FP), 'AgeDB30': len(dataset_AgeDB30)}
    
    print('training and validation data loaded')
    
    return dataloaders, dataset_sizes, dataset

def load_data_train_fix_from_lmdb(batch_size, dataset = 'Faces_emore'):
    
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
    
    root = '/root/faces_emore/LFW/lfw_align_112/'
    file_list = '/root/faces_emore/LFW/pairs.txt'
    dataset_LFW = LFW(root, file_list, transform=transform)
    
    root = '/root/faces_emore/cfp_fp/'
    file_list = '/root/faces_emore/cfp_fp/cfp_fp_pair.txt'
    dataset_CFP_FP = CFP_FP(root, file_list, transform=transform)
        
    root = '/root/faces_emore/agedb_30/'
    file_list = '/root/faces_emore/agedb_30/agedb_30_pair.txt'
    dataset_AgeDB30 = AgeDB30(root, file_list, transform=transform)  
    
    if dataset == 'CASIA':
        
        root = 'data_set/CASIA_Webface_Image'
        file_list = 'data_set/CASIA_Webface_Image/webface_align_112.txt'
        dataset_train = CASIAWebFace(root, file_list, transform=transform)
        
    elif dataset == 'Faces_emore':

        path = "/data/face_dataset/ms1m.lmdb"
        dataset_train = ImageFolderLMDB(path, transform) 
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    else:
        raise NameError('no training data exist!')

    dataset_size = len(dataset_train)
    
    
    dataloaders = {'train_dataset': data.DataLoader(dataset_train, batch_size=batch_size,pin_memory=True, sampler=train_sampler),
                   'LFW': data.DataLoader(dataset_LFW, batch_size=batch_size,pin_memory=True, shuffle=False),
                   'CFP_FP': data.DataLoader(dataset_CFP_FP, batch_size=batch_size,pin_memory=True, shuffle=False),
                   'AgeDB30': data.DataLoader(dataset_AgeDB30, batch_size=batch_size,pin_memory=True, shuffle=False)}
    
    dataset = {'train_dataset': dataset_train,'LFW': dataset_LFW,
               'CFP_FP': dataset_CFP_FP, 'AgeDB30': dataset_AgeDB30}
    
    dataset_sizes = {'train': len(dataset_train),'LFW': len(dataset_LFW),
                     'CFP_FP': len(dataset_CFP_FP), 'AgeDB30': len(dataset_AgeDB30)}
    
    print('training and validation data loaded')
    
    return dataloaders, dataset_sizes, dataset

def load_data_from_lmdb(batch_size, dataset = 'Faces_emore'):
    
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

        path = "/data/face_dataset/ms1m.lmdb"
        dataset_train = ImageFolderLMDB(path, transform)
        
    
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

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, dali_cpu=False, local_rank=0, world_size=1):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        dali_device = "gpu"
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        # self.cutout = Cutout(n_holes=1, length=16)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        # images = self.cutout(images)
        output = self.cmnp(images, mirror=rng)
        return [output, self.labels]



def get_imagenet_iter_dali(type, image_dir, batch_size, num_threads, device_id, num_gpus,  val_size=256,
                           world_size=1,
                           local_rank=0):
    if type == 'train':
        pip_train = HybridTrainPipe(batch_size=batch_size, num_threads=num_threads, device_id=local_rank,
                                    data_dir=image_dir,
                                    world_size=world_size, local_rank=local_rank)
        pip_train.build()
        dali_iter_train = DALIClassificationIterator(pip_train, size=pip_train.epoch_size("Reader") // world_size)
        return dali_iter_train
    elif type == 'val':
        pip_val = HybridValPipe(batch_size=batch_size, num_threads=num_threads, device_id=local_rank,
                                data_dir=image_dir + '/val',
                                size=val_size, world_size=world_size, local_rank=local_rank)
        pip_val.build()
        dali_iter_val = DALIClassificationIterator(pip_val, size=pip_val.epoch_size("Reader") // world_size)
        return dali_iter_val
    elif type == '':
        pip_train = HybridTrainPipe(batch_size=batch_size, num_threads=num_threads, device_id=local_rank,
                                    data_dir=image_dir,
                                    world_size=world_size, local_rank=local_rank)
        pip_train.build()
        dali_iter_train = DALIClassificationIterator(pip_train, size=pip_train.epoch_size("Reader") // world_size)
        return dali_iter_train


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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

class CommonPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(CommonPipeline, self).__init__(batch_size, num_threads, device_id)
        dali_device = "gpu"
        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
        self.cmn = ops.CropMirrorNormalize(device = "gpu",
                                            output_dtype = types.FLOAT,
                                            image_type = types.RGB,
                                            mean = [128., 128., 128.],
                                            std = [1., 1., 1.])
        self.coin = ops.CoinFlip(device = "gpu",probability=0.5)
        self.uniform = ops.Uniform(range = (0.0, 1.0))

    def base_define_graph(self, inputs, labels):
        images = self.decode(inputs)
        output = self.cmn(images, crop_pos_x = self.uniform(),
                          crop_pos_y = self.uniform())
        return (output, labels)

class MXNetReaderPipeline(CommonPipeline):
    def __init__(self, db_folder, batch_size, num_threads, device_id, num_gpus):
        super(MXNetReaderPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.MXNetReader(path = [db_folder+"train.rec"], index_path=[db_folder+"train.idx"],
                                     random_shuffle = True, shard_id = device_id, num_shards = num_gpus)

    def define_graph(self):
        images, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels)

def load_data_with_MXNet_dali(batch_size ,args ,dataset = 'Faces_emore'):
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])  
    
    root = './dataset/LFW/lfw_align_112'
    file_list = './dataset/LFW/pairs.txt'
    dataset_LFW = LFW(root, file_list, transform=transform)
    
    root = './dataset/CFP-FP/CFP_FP_aligned_112'
    file_list = './dataset/CFP-FP/cfp_fp_pair.txt'
    dataset_CFP_FP = CFP_FP(root, file_list, transform=transform)
        
    root = './dataset/AgeDB-30/agedb30_align_112'
    file_list = './dataset/AgeDB-30/agedb_30_pair.txt'
    dataset_AgeDB30 = AgeDB30(root, file_list, transform=transform)  
    
    if dataset == 'CASIA':
        root = 'data_set/CASIA_Webface_Image'
        file_list = 'data_set/CASIA_Webface_Image/webface_align_112.txt'
        dataset_train = CASIAWebFace(root, file_list, transform=transform)
    elif dataset == 'Faces_emore':
        path = "./dataset/"
        pipes = MXNetReaderPipeline(path,batch_size=batch_size, num_threads=4, device_id = args.local_rank, num_gpus = torch.cuda.device_count()) 
        pipes.build()
        train_loader = DALIGenericIterator(pipes, ['data', 'label'], pipes.epoch_size("Reader"))

        # root = '/data/face_dataset/imgs/'
        # file_list = '/data/face_dataset/imgs/faces_emore_align_112.txt'
        # dataset_train = MS1M(root, file_list, transform=transform)
    else:
        raise NameError('no training data exist!')

    dataloaders = {'train_dataset': train_loader,
                   'LFW': data.DataLoader(dataset_LFW, batch_size=batch_size,pin_memory=True, shuffle=False),
                   'CFP_FP': data.DataLoader(dataset_CFP_FP, batch_size=batch_size,pin_memory=True, shuffle=False),
                   'AgeDB30': data.DataLoader(dataset_AgeDB30, batch_size=batch_size,pin_memory=True, shuffle=False)}
    
    dataset = {'LFW': dataset_LFW,
               'CFP_FP': dataset_CFP_FP, 'AgeDB30': dataset_AgeDB30}
    
    dataset_sizes = {'LFW': len(dataset_LFW),
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
        file_list = '/data/face_dataset/imgs/faces_emore_align_112.txt'
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

def load_data_dataparallel(batch_size, dataset = 'Faces_emore'):
    
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
        file_list = '/data/face_dataset/imgs/faces_emore_align_112.txt'
        dataset_train = MS1M(root, file_list, transform=train_transform) 
        # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    
    else:
        raise NameError('no training data exist!')

    dataset_size = len(dataset_train)
    train_size = int(0.8 * dataset_size)
    valid_size = dataset_size - train_size
    train_subset, valid_subset = torch.utils.data.random_split(dataset_train, [train_size, valid_size])
    
    # 'train_dataset': data.DataLoader(dataset_train, batch_size=batch_size,  sampler=train_sampler),
    dataloaders = {'train_dataset': data.DataLoader(dataset_train, batch_size=batch_size,  shuffle=True),
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