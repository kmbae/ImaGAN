from __future__ import print_function

import os
from glob import glob
from tqdm import trange
from itertools import chain

import torch
from torch import nn
import torch.nn.parallel
import torchvision.utils as vutils
from torchvision import transforms
from torch.autograd import Variable
from models import *
from net import *
from data_loader import get_loader
import numpy as np
import ipdb
from tensorboardX import SummaryWriter
from datetime import datetime
import visdom

# Some utilities
class VisdomLine():
    def __init__(self, vis, opts):
        self.vis = vis
        self.opts = opts
        self.win = None

    def Update(self, x, y):
        if self.win is None:
            self.win = self.vis.line(X=x, Y=y, opts=self.opts)
        else:
            self.vis.line(X=x, Y=y, opts=self.opts, win=self.win)

class VisdomImage():
    def __init__(self, vis, opts):
        self.vis = vis
        self.opts = opts
        self.win = None

    def Update(self, image):
        if self.win is None:
            self.win = self.vis.image(image, opts=self.opts)
        else:
            self.vis.image(image, opts=self.opts, win=self.win)


tmp = datetime.now()

writer = SummaryWriter('../runs/' + str(tmp))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Trainer(object):
    def __init__(self, config, a_data_loader, b_data_loader, a1_data_loader, b1_data_loader):
        self.config = config

        self.a_data_loader = a_data_loader
        self.b_data_loader = b_data_loader
        self.a1_data_loader = a1_data_loader
        self.b1_data_loader = b1_data_loader

        self.num_gpu = config.num_gpu
        self.dataset = config.dataset

        self.loss = config.loss
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size
        self.weight_decay = config.weight_decay
        self.cnn_type = config.cnn_type

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step

        self.build_model()

        if self.num_gpu == 1:
            self.F.cuda()
            self.G.cuda()
            self.DF.cuda()
            self.DG.cuda()
        elif self.num_gpu > 1:
            self.F = nn.DataParallel(self.F.cuda(),device_ids=range(torch.cuda.device_count()))
            self.G = nn.DataParallel(self.G.cuda(),device_ids=range(torch.cuda.device_count()))
            self.DF = nn.DataParallel(self.DF.cuda(),device_ids=range(torch.cuda.device_count()))
            self.DG = nn.DataParallel(self.DG.cuda(),device_ids=range(torch.cuda.device_count()))

        if self.load_path:
            self.load_model()

        # For visualization
        self.vis = visdom.Visdom()
        self.vis_image = VisdomImage(self.vis, dict(title='Content / Style / Result / Reconstruct'))


    def build_model(self):
        if self.dataset == 'toy':
            print("No toy!!")
        else:
            a_height, a_width, a_channel = self.a_data_loader.shape
            b_height, b_width, b_channel = self.b_data_loader.shape

            if self.cnn_type == 0:
                #conv_dims, deconv_dims = [64, 128, 256, 512], [512, 256, 128, 64]
                conv_dims, deconv_dims = [64, 128, 256, 512], [256, 128, 64]
            elif self.cnn_type == 1:
                #conv_dims, deconv_dims = [32, 64, 128, 256], [256, 128, 64, 32]
                conv_dims, deconv_dims = [32, 64, 128, 256], [128, 64, 32]
            else:
                raise Exception("[!] cnn_type {} is not defined".format(self.cnn_type))

            self.F = StyleTransferNet()
            self.G = StyleTransferNet()

            self.DF = DiscriminatorCNN(a_channel, 1, conv_dims, self.num_gpu)
            self.DG = DiscriminatorCNN(b_channel, 1, conv_dims, self.num_gpu)

    def load_model(self):
        print("[*] Load models from {}...".format(self.load_path))

        paths = glob(os.path.join(self.load_path, 'G_AB_*.pth'))
        paths.sort()

        if len(paths) == 0:
            print("[!] No checkpoint found in {}...".format(self.load_path))
            return

        idxes = [int(os.path.basename(path.split('.')[0].split('_')[-1])) for path in paths]
        self.start_step = max(idxes)

        if self.num_gpu == 0:
            map_location = lambda storage, loc: storage
        else:
            map_location = None

        G_AB_filename = '{}/G_AB_{}.pth'.format(self.load_path, self.start_step)
        #self.G_AB.load_state_dict(torch.load(G_AB_filename, map_location=map_location))
        self.G.load_state_dict(
            torch.load('{}/G_BA_{}.pth'.format(self.load_path, self.start_step), map_location=map_location))

        self.D_S.load_state_dict(
            torch.load('{}/D_A_{}.pth'.format(self.load_path, self.start_step), map_location=map_location))
        self.D_H.load_state_dict(
            torch.load('{}/D_B_{}.pth'.format(self.load_path, self.start_step), map_location=map_location))

        print("[*] Model loaded: {}".format(G_AB_filename))

    def train(self):
        d = nn.MSELoss()
        bce = nn.BCELoss()

        real_label = 1
        fake_label = 0

        real_tensor = Variable(torch.FloatTensor(self.batch_size))
        _ = real_tensor.data.fill_(real_label)

        fake_tensor = Variable(torch.FloatTensor(self.batch_size))
        _ = fake_tensor.data.fill_(fake_label)

        if self.num_gpu > 0:
            d.cuda()
            bce.cuda()

            real_tensor = real_tensor.cuda()
            fake_tensor = fake_tensor.cuda()

        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(config.optimizer))

        optimizer_F = optimizer(self.F.module.decoder.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.weight_decay)
        optimizer_G = optimizer(self.G.module.decoder.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.weight_decay)
        optimizer_DF = optimizer(self.DF.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.weight_decay)
        optimizer_DG = optimizer(self.DG.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.weight_decay)

        ##
        A1_loader, B1_loader = iter(self.a_data_loader), iter(self.b_data_loader)
        valid_x_A1, valid_x_B1 = torch.Tensor(np.load('../valid_x_A1.npy')), torch.Tensor(np.load('../valid_x_B1.npy'))
        valid_x_A1, valid_x_B1 = self._get_variable(valid_x_A1), self._get_variable(valid_x_B1)
        #self._get_variable(A_loader.next()), self._get_variable(B_loader.next())
        A2_loader, B2_loader = iter(self.a1_data_loader), iter(self.b1_data_loader)
        valid_x_A2, valid_x_B2 = torch.Tensor(np.load('../valid_x_A2.npy')), torch.Tensor(np.load('../valid_x_B2.npy'))
        valid_x_A2, valid_x_B2 = self._get_variable(valid_x_A2), self._get_variable(valid_x_B2)
        ##

        vutils.save_image(valid_x_A1.data, '{}/valid_x_A1.png'.format(self.model_dir))
        vutils.save_image(valid_x_A2.data, '{}/valid_x_A2.png'.format(self.model_dir))

        for step in trange(self.start_step, self.max_step):
            try:
                x_A1 = A1_loader.next()
                x_B1 = B1_loader.next()
            except StopIteration:
                A1_loader = iter(self.a_data_loader)
                B1_loader = iter(self.b_data_loader)
                x_A1 = A1_loader.next()
                x_B1 = B1_loader.next()
            try:
                x_A2 = A2_loader.next()
                x_B2 = B2_loader.next()
            except StopIteration:
                A2_loader = iter(self.a1_data_loader)
                B2_loader = iter(self.b1_data_loader)
                x_A2 = A2_loader.next()
                x_B2 = B2_loader.next()
            if x_A1.size(0) != x_B1.size(0) or x_A2.size(0) != x_B2.size(0) or x_A1.size(0) != x_A2.size(0):
                print("[!] Sampled dataset from A and B have different # of data. Try resampling...")
                continue

            x_A1, x_B1 = self._get_variable(x_A1), self._get_variable(x_B1)
            x_A2, x_B2 = self._get_variable(x_A2), self._get_variable(x_B2)

            batch_size = x_A1.size(0)
            real_tensor.data.resize_(batch_size).fill_(real_label)
            fake_tensor.data.resize_(batch_size).fill_(fake_label)


            # Forward Pass
            L_F_Ada, F12 = self.F(x_B1,x_B2)
            L_G_Ada, G21 = self.G(x_B2,x_B1)
            #ipdb.set_trace()
            L_p_Ada, p = self.G(F12,x_B1)
            L_q_Ada, q = self.F(x_B2,F12)
            L_r_Ada, r = self.F(G21,x_B2)
            L_s_Ada, s = self.G(x_B1,G21)
            L_u_Ada, u = self.F(G21,F12)
            L_v_Ada, v = self.G(F12,G21)

            # Update Discriminators D_F, D_G (F, G are fixed)
            self.DF.zero_grad()
            self.DG.zero_grad()
            L_DF_real, L_DF_fake = bce(self.DF(x_B1), real_tensor), bce(self.DF(F12), fake_tensor)
            L_DG_real, L_DG_fake = bce(self.DG(x_B2), real_tensor), bce(self.DG(G21), fake_tensor)
            L_DF = L_DF_real + L_DF_fake
            L_DG = L_DG_real + L_DG_fake
            L_DF.backward(retain_graph=True)
            L_DG.backward(retain_graph=True)
            optimizer_DF.step()
            optimizer_DG.step()

            # Update F, G to fool the discriminators D_F, D_G + reduce reconstruction loss
            self.F.zero_grad()
            self.G.zero_grad()
            # (1) GAN loss
            L_F_GAN = bce(self.DF(F12), real_tensor)
            L_G_GAN = bce(self.DG(G21), real_tensor)

            # (2) Total loss with Reconstruction Loss
            L_p,L_q,L_r,L_s,L_u,L_v = d(p,x_B1),d(p,x_B2),d(r,x_B2),d(s,x_B1),d(u,x_B2),d(v,x_B1)
            L = (L_F_GAN + L_G_GAN) + 10*(L_p + L_q + L_r + L_s + L_u + L_v) + 3*(L_p_Ada + L_q_Ada + L_r_Ada + L_s_Ada + L_u_Ada + L_v_Ada)
            L.backward()
            optimizer_F.step()
            optimizer_G.step()

            if step % 30 == 0:
                # Display using visdom
                img_cat = torch.cat((x_B1, x_B2, F12, p), dim=3)
                #img_cat = torch.cat((x_B1,x_B2), dim=3)
                img_cat = torch.unbind(img_cat, dim=0)
                img_cat = torch.cat(img_cat, dim=1)
                #img_cat = dm.restore(img_cat.data.cpu())
                restore = transforms.Compose(
                    [
                        transforms.Normalize(mean=[-pow(0.5,0.5), -pow(0.5,0.5), -pow(0.5,0.5)], std=[pow(2,0.5), pow(2,0.5), pow(2,0.5)])
                    ]
                )
                #img_cat = restore(img_cat.data.cpu())
                img_cat = img_cat.data.cpu()
                self.vis_image.Update(torch.clamp(img_cat, 0, 1))

            """
            if step % self.log_step == 0:
                print("[{}/{}] L_DF: {:.4f} Loss_DG: {:.4f}".format(step, self.max_step, L_DF.data[0], L_DG.data[0]))
                print("[{}/{}] L_F_GAN: {:.4f} L_G_GAN: {:.4f}".format(step, self.max_step, L_F_GAN.data[0], L_G_GAN.data[0]))
                print("[{}/{}] L_p: {:.4f} L_q: {:.4f} L_r: {:.4f} L_s: {:.4f} L_u: {:.4f} L_v: {:.4f}".format(step, self.max_step, L_p.data[0], L_q.data[0], L_r.data[0], L_s.data[0], L_u.data[0], L_v.data[0]))

                self.generate_with_A(valid_x_B1, valid_x_B2, self.model_dir, idx=step)
                #self.generate_with_B(valid_x_A1, valid_x_A, self.model_dir, idx=step)
                writer.add_scalars('loss_FG', {'L_F_GAN':L_F_GAN,'L_G_GAN':L_G_GAN,'L_p':L_p,'L_q':L_q,'L_r':L_r,'L_s':L_s,'L_u':L_u,'L_v':L_v},step)
                writer.add_scalars('loss_DF', {'L_DF':L_DF}, step)
                writer.add_scalars('loss_DG', {'L_DG':L_DG}, step)

                # Display using visdom
                img_cat = torch.cat((x_B1, x_B2, F12), dim=3)
                #img_cat = torch.cat((x_B1,x_B2), dim=3)
                img_cat = torch.unbind(img_cat, dim=0)
                img_cat = torch.cat(img_cat, dim=1)
                #img_cat = dm.restore(img_cat.data.cpu())
                restore = transforms.Compose(
                    [
                        transforms.Normalize(mean=[-2.118, -2.036, -1.804],
                                             std=[4.367, 4.464, 4.444]),
                    ]
                )
                img_cat = restore(img_cat.data.cpu())
                self.vis_image.Update(torch.clamp(img_cat, 0, 1))


            if step % self.save_step == self.save_step - 1:
                print("[*] Save models to {}...".format(self.model_dir))

                torch.save(self.F.state_dict(), '{}/F_{}.pth'.format(self.model_dir, step))
                torch.save(self.G.state_dict(), '{}/G_{}.pth'.format(self.model_dir, step))

                torch.save(self.DF.state_dict(), '{}/DF{}.pth'.format(self.model_dir, step))
                torch.save(self.DG.state_dict(), '{}/DG{}.pth'.format(self.model_dir, step))
            """
    def generate_with_A(self, inputs, input_ref, path, idx=None):
        F12 = self.F(inputs,input_ref)
        p = self.G(F12,inputs)
        F12_path = '{}/{}_F12.png'.format(path, idx)
        #x_ABA_path = '{}/{}_x_ABA.png'.format(path, idx)

        vutils.save_image(F12.data, F12_path)
        print("[*] Samples saved: {}".format(F12_path))
        writer.add_image('inputs', inputs[:16], idx)
        writer.add_image('F12', F12[:16], idx)
        writer.add_image('p', p[:16], idx)
        #writer.add_image('x_ABA', x_ABA, idx)
        #vutils.save_image(x_ABA.data, x_ABA_path)
        #print("[*] Samples saved: {}".format(x_ABA_path))

    def generate_with_B(self, inputs, input_ref, path, idx=None):
        x_BA = self.F(inputs,input_ref)
        x_BAB = self.G(x_BA, input_ref)
        x_ABAf = self.F(x_BAB)

        x_BA_path = '{}/{}_x_BA.png'.format(path, idx)
        #x_BAB_path = '{}/{}_x_BAB.png'.format(path, idx)

        vutils.save_image(x_BAB.data, x_BA_path)
        print("[*] Samples saved: {}".format(x_BA_path))

        writer.add_image('x_A2f', x_BA[:16], idx)
        writer.add_image('x_A2valid', inputs[:16], idx)
        writer.add_image('x_A2_1', x_BAB[:16], idx)
        writer.add_image('x_B1_1f', x_ABAf[:16], idx)
        #writer.add_image('x_BAB', x_BAB, idx)
        #vutils.save_image(x_BAB.data, x_BAB_path)
        #print("[*] Samples saved: {}".format(x_BAB_path))

    def generate_infinitely(self, inputs, path, input_type, count=10, nrow=2, idx=None):
        if input_type.lower() == "a":
            iterator = [self.G_AB, self.G_BA] * count
        elif input_type.lower() == "b":
            iterator = [self.G_BA, self.G_AB] * count

        out = inputs
        for step, model in enumerate(iterator):
            out = model(out)

            out_path = '{}/{}_x_{}_#{}.png'.format(path, idx, input_type, step)
            vutils.save_image(out.data, out_path, nrow=nrow)
            print("[*] Samples saved: {}".format(out_path))

    def test(self):
        batch_size = self.config.sample_per_image
        A_loader, B_loader = iter(self.a_data_loader), iter(self.b_data_loader)

        test_dir = os.path.join(self.model_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        step = 0
        while True:
            try:
                x_A, x_B = self._get_variable(A_loader.next()), self._get_variable(B_loader.next())
            except StopIteration:
                print("[!] Test sample generation finished. Samples are in {}".format(test_dir))
                break

            vutils.save_image(x_A.data, '{}/{}_x_A.png'.format(test_dir, step))
            vutils.save_image(x_B.data, '{}/{}_x_B.png'.format(test_dir, step))

            self.generate_with_A(x_A, test_dir, idx=step)
            self.generate_with_B(x_B, test_dir, idx=step)

            self.generate_infinitely(x_A, test_dir, input_type="A", count=10, nrow=4, idx=step)
            self.generate_infinitely(x_B, test_dir, input_type="B", count=10, nrow=4, idx=step)

            step += 1

    def _get_variable(self, inputs):
        if self.num_gpu > 0:
            out = Variable(inputs.cuda())
        else:
            out = Variable(inputs)
        return out
