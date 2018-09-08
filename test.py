from models import *
import torch
from torch import nn
import glob
from torchvision import transforms
import tqdm
from config import get_config
from PIL import Image
import argparse
import os

# Usage example
# run test.py --load=../logs/edges2shoes_2018-09-07_17-05-19 --iter=10000 --con=../validation_image/handbag_picking --sty=../validation_image/valid_x_A1

parser = argparse.ArgumentParser()
parser.add_argument("--load", type=str, help="Saved file dir")
parser.add_argument("--iter", type=str, help="Number of iteration")
parser.add_argument("--con", type=str, help="Dir of content images")
parser.add_argument("--sty", type=str, help="Dir of style images")
args = parser.parse_args()

scale_size = 64
transform = transforms.Compose([
    transforms.Scale(scale_size),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
ToPILImage = transforms.ToPILImage()

if __name__=='__main__':
    config, unparsed = get_config()
    conv_dims, deconv_dims = [64, 128, 256, 512], [256, 128, 64]
    a_channel = 3
    b_channel = 3
    num_gpu = torch.cuda.device_count()
    G = GeneratorCNN_g(a_channel+b_channel, b_channel, conv_dims, deconv_dims, num_gpu)
    F = GeneratorCNN(a_channel, a_channel, conv_dims, deconv_dims, num_gpu)

    G = nn.DataParallel(G.cuda(),device_ids=range(torch.cuda.device_count()))
    F = nn.DataParallel(F.cuda(),device_ids=range(torch.cuda.device_count()))
    print('Loading model')
    G.load_state_dict(torch.load(args.load + '/G_{}.pth'.format(args.iter)))
    F.load_state_dict(torch.load(args.load + '/F_{}.pth'.format(args.iter)))
    print('Model loaded')

    if not os.path.exists('./results'):
        os.mkdir('./results')

    list_con = os.listdir(args.con)
    list_sty = os.listdir(args.sty)
    img_con = []
    img_sty = []
    for i in (list_con):
        img_con.append(transform(Image.open(args.con+'/'+i)))
    for j in (list_sty):
        img_sty.append(transform(Image.open(args.sty+'/'+j)))

    with torch.no_grad():
        G.eval()
        F.eval()
        for i, con_tmp in tqdm.tqdm(enumerate(img_con)):
            con_tmp = torch.unsqueeze(con_tmp.cuda(),0)
            if not os.path.exists('./results/{}'.format(list_con[i].split('.')[0])):
                os.mkdir('./results/{}'.format(list_con[i].split('.')[0]))
            for j, sty_tmp in enumerate(img_sty):
                sty_tmp = torch.unsqueeze(sty_tmp.cuda(),0)
                img_out = G(F(con_tmp), sty_tmp)
                img_out = ToPILImage(img_out.data[0].cpu())
                img_out.save('./results/{}/{}_{}.jpg'.format(list_con[i].split('.')[0],
                    list_con[i].split('.')[0],list_sty[j].split('.')[0]))






