import os
import numpy as np
from glob import glob
from PIL import Image, ImageFilter
from tqdm import tqdm

import torch
from torchvision import transforms
#from skimage import feature
#from skimage.color import rgb2gray

PIX2PIX_DATASETS = [
    'facades', 'cityscapes', 'maps', 'edges2shoes', 'edges2handbags']

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def pix2pix_split_images_val(root):
    paths = glob(os.path.join(root, "val/*"))

    a_path = os.path.join(root, "B_val")
    b_path = os.path.join(root, "A_val")

    makedirs(a_path)
    makedirs(b_path)

    for path in tqdm(paths, desc="pix2pix processing"):
        filename = os.path.basename(path)

        a_image_path = os.path.join(a_path, filename)
        b_image_path = os.path.join(b_path, filename)

        if os.path.exists(a_image_path) and os.path.exists(b_image_path):
            continue

        image = Image.open(os.path.join(path)).convert('RGB')
        data = np.array(image)

        height, width, channel = data.shape

        a_image = Image.fromarray(data[:,:width/2].astype(np.uint8))
        b_image = Image.fromarray(data[:,width/2:].astype(np.uint8))

        a_image.save(a_image_path)
        b_image.save(b_image_path)

def pix2pix_split_images(root):
    paths = glob(os.path.join(root, "train/*"))

    a_path = os.path.join(root, "B")
    b_path = os.path.join(root, "A")

    makedirs(a_path)
    makedirs(b_path)

    for path in tqdm(paths, desc="pix2pix processing"):
        filename = os.path.basename(path)

        a_image_path = os.path.join(a_path, filename)
        b_image_path = os.path.join(b_path, filename)

        if os.path.exists(a_image_path) and os.path.exists(b_image_path):
            continue

        image = Image.open(os.path.join(path)).convert('RGB')
        data = np.array(image)

        height, width, channel = data.shape

        a_image = Image.fromarray(data[:,:width/2].astype(np.uint8))
        b_image = Image.fromarray(data[:,width/2:].astype(np.uint8))

        a_image.save(a_image_path)
        b_image.save(b_image_path)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, scale_size, data_type, skip_pix2pix_processing=False):
        self.root = root
        self.data_type = data_type
        if not os.path.exists(self.root):
            raise Exception("[!] {} not exists.".format(root))

        self.name = os.path.basename(root)
        if self.name in PIX2PIX_DATASETS and not skip_pix2pix_processing:
            pix2pix_split_images(self.root)
        if self.name in PIX2PIX_DATASETS and not skip_pix2pix_processing:
            pix2pix_split_images_val(self.root)
        self.paths = glob(os.path.join(self.root, '{}/*'.format(data_type)))
        if len(self.paths) == 0:
            raise Exception("No images are found in {}".format(self.root))
        self.shape = list(Image.open(self.paths[0]).size) + [3]

        self.transform = transforms.Compose([
            transforms.Scale(scale_size),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        if not os.path.exists('data/{}_A.npy'.format(root.split('/')[-1])):
            val_paths = glob(os.path.join(self.root, '{}_val/*'.format(data_type)))
            if len(val_paths)==0:
                raise Exception("No images are found in {}".format(os.path.join(self.root, '{}_val/*'.format(data_type))))
            image = []
            edges = []
            for i in val_paths:
                image.append(self.transform(Image.open(i).convert('RGB')))
                edges.append(self.transform(Image.open(i.replace('/A','/B')).convert('RGB')))

            np.save('data/{}_A.npy'.format(root.split('/')[-1]), np.array(torch.stack(image,0)))
            np.save('data/{}_B.npy'.format(root.split('/')[-1]), np.array(torch.stack(edges,0)))

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert('RGB')
        edges = Image.open(self.paths[index].replace('/A','/B')).convert('RGB')
        #edges = image.filter(ImageFilter.FIND_EDGES)

        #if self.data_type=='B':
        #    image = image.filter(ImageFilter.MinFilter(3))
        return {'image':self.transform(image), 'edges':self.transform(edges)}

    def __len__(self):
        return len(self.paths)

def get_loader(root, batch_size, scale_size, num_workers=2,
               skip_pix2pix_processing=False, shuffle=True):
    a_data_set = \
        Dataset(root, scale_size, "A", skip_pix2pix_processing)#, \
        #Dataset(root, scale_size, "B", skip_pix2pix_processing)
    a_data_loader = torch.utils.data.DataLoader(dataset=a_data_set,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers)
    #b_data_loader = torch.utils.data.DataLoader(dataset=b_data_set,
    #                                            batch_size=batch_size,
    #                                            shuffle=True,
    #                                            num_workers=num_workers)
    a_data_loader.shape = a_data_set.shape
    #b_data_loader.shape = b_data_set.shape

    return a_data_loader#, b_data_loader
