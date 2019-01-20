from __future__ import print_function

import os
from glob import glob
from tqdm import trange
from itertools import chain

import torch
from torch import nn
import torch.nn.parallel
import torchvision.utils as vutils
import numpy as np
from torch.autograd import Variable
from models import *
from data_loader import get_loader
import math
from tensorboardX import SummaryWriter
from datetime import datetime
import torchvision
import ipdb
tmp = datetime.now()

writer = SummaryWriter('./runs/' + str(tmp))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.in_channels
        for k in m.kernel_size:
            n *=k
        stdv = 1. / math.sqrt(n)
        m.weight.data.uniform_(-stdv, stdv)
        if m.bias is not None:
            m.bias.data.uniform_(-stdv, stdv)

class Trainer(object):
    def __init__(self, config, a_data_loader, a1_data_loader):
        self.config = config

        self.a_data_loader = a_data_loader
        self.a1_data_loader = a1_data_loader

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
        self.identity = config.identity

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step

        self.build_model()

        if torch.cuda.is_available():
            self.G = nn.DataParallel(self.G.cuda(),device_ids=range(torch.cuda.device_count()))
            self.F = nn.DataParallel(self.F.cuda(),device_ids=range(torch.cuda.device_count()))
            self.D_A1 = nn.DataParallel(self.D_A1.cuda(),device_ids=range(torch.cuda.device_count()))
            self.D_A2 = nn.DataParallel(self.D_A2.cuda(),device_ids=range(torch.cuda.device_count()))
            self.D_B1 = nn.DataParallel(self.D_B1.cuda(),device_ids=range(torch.cuda.device_count()))
            self.D_B2 = nn.DataParallel(self.D_B2.cuda(),device_ids=range(torch.cuda.device_count()))

        if self.load_path:
            self.load_model()

    def build_model(self):
        if self.dataset == 'toy':
            self.G_AB = GeneratorFC(2, 2, [config.fc_hidden_dim] * config.g_num_layer)
            self.G_BA = GeneratorFC(2, 2, [config.fc_hidden_dim] * config.g_num_layer)

            self.D_A = DiscriminatorFC(2, 1, [config.fc_hidden_dim] * config.d_num_layer)
            self.D_B = DiscriminatorFC(2, 1, [config.fc_hidden_dim] * config.d_num_layer)
        else:
            a_height, a_width, a_channel = self.a_data_loader.shape
            b_height, b_width, b_channel = self.a1_data_loader.shape

            if self.cnn_type == 0:
                #conv_dims, deconv_dims = [64, 128, 256, 512], [512, 256, 128, 64]
                conv_dims, deconv_dims = [64, 128, 256, 512], [256, 128, 64]
            elif self.cnn_type == 1:
                #conv_dims, deconv_dims = [32, 64, 128, 256], [256, 128, 64, 32]
                conv_dims, deconv_dims = [32, 64, 128, 256], [128, 64, 32]
            else:
                raise Exception("[!] cnn_type {} is not defined".format(self.cnn_type))

            ### Define networks ###
            self.G = GeneratorCNN_g(
                    a_channel+b_channel, b_channel, conv_dims, deconv_dims, self.num_gpu)
            self.F = GeneratorCNN(
                    a_channel, a_channel, conv_dims, deconv_dims, self.num_gpu)

            self.D_A1 = DiscriminatorCNN(
                    a_channel, 1, conv_dims, self.num_gpu)
            self.D_A2 = DiscriminatorCNN(
                    b_channel, 1, conv_dims, self.num_gpu)
            self.D_B1 = DiscriminatorCNN(
                    a_channel, 1, conv_dims, self.num_gpu)
            self.D_B2 = DiscriminatorCNN(
                    b_channel, 1, conv_dims, self.num_gpu)

            self.D_B1.apply(weights_init)
            self.D_B2.apply(weights_init)
            self.D_A1.apply(weights_init)
            self.D_A2.apply(weights_init)
            self.G.apply(weights_init)
            self.F.apply(weights_init)

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
        self.G.load_state_dict(
            torch.load('{}/G_BA_{}.pth'.format(self.load_path, self.start_step), map_location=map_location))

        self.D_A1.load_state_dict(
            torch.load('{}/D_A_{}.pth'.format(self.load_path, self.start_step), map_location=map_location))
        self.D_A2.load_state_dict(
            torch.load('{}/D_B_{}.pth'.format(self.load_path, self.start_step), map_location=map_location))

        print("[*] Model loaded: {}".format(G_AB_filename))

    def train(self):
        ## Define loss function
        d = nn.MSELoss()
        #d = nn.L1Loss()
        bce = nn.BCELoss()

        ## Labels for real and fake data
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

        optimizer_d = optimizer(
            chain(self.D_A1.parameters(), self.D_A2.parameters()),
            lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.weight_decay)
        optimizer_db = optimizer(
            chain(self.D_B1.parameters(), self.D_B2.parameters()),
            lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.weight_decay)
        optimizer_f = optimizer(
            self.F.parameters(),
            lr=self.lr, betas=(self.beta1, self.beta2))
        optimizer_g = optimizer(
            self.G.parameters(),
            lr=self.lr, betas=(self.beta1, self.beta2))

        ### Dataloader ###
        A_loader = iter(self.a_data_loader)
        A1_loader = iter(self.a1_data_loader)
        try:
            valid_x_A, valid_x_B = torch.Tensor(np.load(self.config.dataset_A1+'_A.npy')), torch.Tensor(np.load(self.config.dataset_A1+'_B.npy'))
            valid_x_A, valid_x_B = self._get_variable(valid_x_A), self._get_variable(valid_x_B)
            valid_x_A1, valid_x_B1=torch.Tensor(np.load(self.config.dataset_A2+'_A.npy')), torch.Tensor(np.load(self.config.dataset_A2+'_B.npy'))
            valid_x_A1, valid_x_B1 = self._get_variable(valid_x_A1), self._get_variable(valid_x_B1)
        except:
            raise Exception('Cannot load validation file. Validation data not created')

        vutils.save_image(valid_x_A.data, '{}/valid_x_A1.png'.format(self.model_dir))
        vutils.save_image(valid_x_B.data, '{}/valid_x_B1.png'.format(self.model_dir))
        vutils.save_image(valid_x_A1.data, '{}/valid_x_A2.png'.format(self.model_dir))
        vutils.save_image(valid_x_B1.data, '{}/valid_x_B2.png'.format(self.model_dir))

        ### Training loop ###
        for step in trange(self.start_step, self.max_step):
            try:
                x_A1 = A_loader.next()
            except StopIteration:
                A_loader = iter(self.a_data_loader)
                x_A1 = A_loader.next()
            try:
                x_A2 = A1_loader.next()
            except StopIteration:
                A1_loader = iter(self.a1_data_loader)
                x_A2 = A1_loader.next()

            x_A1, x_B1 = x_A1['image'], x_A1['edges']
            x_A2, x_B2 = x_A2['image'], x_A2['edges']
            if x_A1.size(0) != x_B1.size(0) or x_A2.size(0) != x_B2.size(0) or x_A1.size(0) != x_A2.size(0):
                print("[!] Sampled dataset from A and B have different # of data. Try resampling...")
                continue

            x_A1, x_B1 = self._get_variable(x_A1), self._get_variable(x_B1)
            x_A2, x_B2 = self._get_variable(x_A2), self._get_variable(x_B2)

            batch_size = x_A1.size(0)
            real_tensor.data.resize_(batch_size).fill_(real_label)
            fake_tensor.data.resize_(batch_size).fill_(fake_label)

            ## Update Db network
            self.D_B1.zero_grad()
            self.D_B2.zero_grad()
            #optimizer_db.zero_grad()

            x_A12 = self.F(x_A1).detach()
            x_A21 = self.F(x_A2).detach()
            if self.loss == "log_prob":
                l_d_A_real, l_d_A_fake = bce(self.D_B1(x_B1), real_tensor), bce(self.D_B1(x_A12), fake_tensor)
                l_d_B_real, l_d_B_fake = bce(self.D_B2(x_B2), real_tensor), bce(self.D_B2(x_A21), fake_tensor)
            elif self.loss == "least_square":
                l_d_A_real, l_d_A_fake = \
                    0.5 * torch.mean((self.D_A(x_A) - 1)**2), 0.5 * torch.mean((self.D_A(x_BA))**2)
                l_d_B_real, l_d_B_fake = \
                    0.5 * torch.mean((self.D_B(x_B) - 1)**2), 0.5 * torch.mean((self.D_B(x_AB))**2)
            else:
                raise Exception("[!] Unkown loss type: {}".format(self.loss))

            l_d_A = l_d_A_real + l_d_A_fake
            l_d_B = l_d_B_real + l_d_B_fake

            l_dB = l_d_A + l_d_B
            l_dB.backward()
            optimizer_db.step()

            ## Update D network
            self.D_A1.zero_grad()
            self.D_A2.zero_grad()
            #optimizer_d.zero_grad()

            x_A12 = self.G(self.F(x_A1), x_A2).detach()
            x_A21 = self.G(self.F(x_A2), x_A1).detach()

            x_A121 = self.G(self.F(x_A12), x_A1).detach()
            x_A212 = self.G(self.F(x_A21), x_A2).detach()
            #x_ABA = self.G_BA(x_AB).detach()
            #x_BAB = self.G_AB(x_BA).detach()

            if self.loss == "log_prob":
                l_d_A_real, l_d_A_fake, tmp1 = bce(self.D_A1(x_A1), real_tensor), bce(self.D_A1(x_A12), fake_tensor), bce(self.D_A1(x_A121), fake_tensor)
                l_d_B_real, l_d_B_fake, tmp2 = bce(self.D_A2(x_A2), real_tensor), bce(self.D_A2(x_A21), fake_tensor), bce(self.D_A2(x_A212), fake_tensor)
            elif self.loss == "least_square":
                l_d_A_real, l_d_A_fake = \
                    0.5 * torch.mean((self.D_A(x_A) - 1)**2), 0.5 * torch.mean((self.D_A(x_BA))**2)
                l_d_B_real, l_d_B_fake = \
                    0.5 * torch.mean((self.D_B(x_B) - 1)**2), 0.5 * torch.mean((self.D_B(x_AB))**2)
            else:
                raise Exception("[!] Unkown loss type: {}".format(self.loss))

            l_d_A = l_d_A_real + l_d_A_fake + tmp1
            l_d_B = l_d_B_real + l_d_B_fake + tmp2

            l_dA = l_d_A + l_d_B

            l_dA.backward()
            optimizer_d.step()

            # Update F network
            self.F.zero_grad()
            #optimizer_f.zero_grad()

            x_BA1 = self.F(x_A1)
            x_BA2 = self.F(x_A2)

            l_gan_Af = bce(self.D_B1(x_BA1), real_tensor)
            l_gan_Bf = bce(self.D_B2(x_BA2), real_tensor)

            L_R = d(x_BA1, x_B1) + d(x_BA2, x_B2)
            L_R_adv = l_gan_Af + l_gan_Bf
            l_f = l_gan_Af + l_gan_Bf + L_R

            l_f.backward()
            optimizer_f.step()

            # update G network
            self.G.zero_grad()
            #optimizer_g.zero_grad()

            x_B1A1 = self.F(x_A1).detach()
            x_B2A2 = self.F(x_A2).detach()

            x_A1G = self.G(x_B1A1, x_A2)
            x_A2G = self.G(x_B2A2, x_A1)

            x_B1AG = self.F(x_A1G)
            x_B2AG = self.F(x_A2G)

            l_const_A = d(self.G(x_B1AG,x_A1), x_A1)# + d(self.G(x_B1A1,x_A1), x_A1)#+ d(self.G(x_BA1,x_A21.detach()), x_A1)
            l_const_B = d(self.G(x_B2AG,x_A2), x_A2)# + d(self.G(x_BA2,x_A2.detach()), x_A2) + d(self.G(x_BA2,x_A12.detach()), x_A2)
            l_const_AB = d(x_B1AG, x_B1A1)# + d(self.G(x_AB,x_AB), x_ABd) + d(self.G(x_AB,x_B), x_ABd))
            l_const_BA = d(x_B2AG, x_B2A2)# + d(self.G(x_BA,x_BA), x_BAd) + d(self.G(x_BA,x_A), x_BAd))
            #l_const_B12 = d(self.G(x_B1, x_B2), x_B1) + d(self.G(x_B2, x_B1), x_B2)

            x_B1A1G = self.F(x_A1G.detach())
            x_A1GA1 = self.G(x_B1A1G, x_A1)
            x_B1A1G = self.F(x_A1GA1)
            x_A1GA1G = self.G(x_B1A1G, x_A2)

            x_B2A2G = self.F(x_A2G.detach())
            x_A2GA2 = self.G(x_B2A2G, x_A2)
            x_B2A2G = self.F(x_A2GA2)
            x_A2GA2G = self.G(x_B2A2G, x_A1)
            l_const_A += d(x_A1GA1G, x_A1G.detach())
            l_const_AB += d(x_B1A1G, x_B1A1)
            l_const_A += d(x_A2GA2G, x_A2G.detach())
            l_const_AB += d(x_B2A2G, x_B2A2)

            L_G_cyc = l_const_A + l_const_AB + l_const_B + l_const_BA

            if self.loss == "log_prob":
                l_gan_A = bce(self.D_A1(x_A1G), real_tensor) + bce(self.D_A1(x_A1GA1), real_tensor)
                # + bce(self.D_F(x_B12f, x_B1), real_tensor)
                l_gan_B = bce(self.D_A2(x_A2G), real_tensor) + bce(self.D_A2(x_A2GA2), real_tensor)
                # + bce(self.D_F(x_B21f, x_B2), real_tensor)
            elif self.loss == "least_square":
                l_gan_A = 0.5 * torch.mean((self.D_A1(x_A12) - 1)**2)
                l_gan_B = 0.5 * torch.mean((self.D_A2(x_A21) - 1)**2)
            else:
                raise Exception("[!] Unkown loss type: {}".format(self.loss))
            L_G_adv = l_gan_A + l_gan_B
            l_g = l_gan_A + l_gan_B + l_const_A + l_const_B + l_const_AB + l_const_BA

            # Identity loss
            #print(self.identity)
            if self.identity:
                l_idn = self.identity*(d(x_A2, x_A1G) + d(x_A1, x_A2G))
                l_g += l_idn

            l_g.backward()
            optimizer_g.step()

            if step % self.log_step == 0:
                print("[{}/{}] Loss_D: {:.4f} Loss_G: {:.4f}". \
                        format(step, self.max_step, l_dA.item(), l_g.item()))
                print("[{}/{}] l_d_A_real: {:.4f} l_d_A_fake: {:.4f}, l_d_B_real: {:.4f}, l_d_B_fake: {:.4f}". \
                        format(step, self.max_step, l_d_A_real.item(), l_d_A_fake.item(),
                             l_d_B_real.item(), l_d_B_fake.item()))
                print("[{}/{}] l_const_A: {:.4f} l_const_B: ". \
                        format(step, self.max_step, l_const_A.item()))
                print("[{}/{}] l_gan_A: {:.4f}, l_gan_B: ". \
                        format(step, self.max_step, l_gan_A.item()))

                self.F.eval()
                self.G.eval()
                self.generate_with_A(valid_x_A, valid_x_A1, self.model_dir, idx=step)
                self.generate_with_B(valid_x_A1, valid_x_A, self.model_dir, idx=step)
                self.F.train()
                self.G.train()

                #writer.add_image('x_A1', x_A1[:16], step)
                #writer.add_image('x_B1', x_B1[:16], step)
                #writer.add_image('x_A2', x_A2[:16], step)
                #writer.add_image('x_B2', x_B2[:16], step)
                # Discriminator loss
                writer.add_scalar('L_d_A', l_dA, step)
                writer.add_scalar('L_d_B', l_dB, step)
                # Style removal network loss
                writer.add_scalar('L_R', L_R, step)
                writer.add_scalar('L_R_adv', L_R_adv, step)
                # Generator loss
                writer.add_scalar('L_G', L_G_cyc, step)
                writer.add_scalar('L_G_adv', L_G_adv, step)
                if self.identity:
                    writer.add_scalar('L_ind', l_ind, step)

            if step % self.save_step == 0:
                print("[*] Save models to {}...".format(self.model_dir))

                torch.save(self.G.state_dict(), '{}/G_{}.pth'.format(self.model_dir, step))
                torch.save(self.F.state_dict(), '{}/F_{}.pth'.format(self.model_dir, step))

                torch.save(self.D_A1.state_dict(), '{}/D_A1_{}.pth'.format(self.model_dir, step))
                torch.save(self.D_A2.state_dict(), '{}/D_A2_{}.pth'.format(self.model_dir, step))

                torch.save(self.D_B1.state_dict(), '{}/D_B1_{}.pth'.format(self.model_dir, step))
                torch.save(self.D_B2.state_dict(), '{}/D_B2_{}.pth'.format(self.model_dir, step))

    def generate_with_A(self, inputs, input_ref, path, idx=None, tf_board=True):
        x_AB = self.F(inputs)
        x_ABA = self.G(x_AB, input_ref)
        x_ABAf = self.F(x_ABA)
        x_ABAB = self.G(x_ABAf, inputs)

        x_AB_path = '{}/{}_x_A1G.png'.format(path, idx)

        vutils.save_image(x_ABA.data, x_AB_path)
        if not os.path.isdir('{}/{}_A1'.format(path, idx)):
            os.makedirs('{}/{}_A1'.format(path, idx))
        for i in range(x_ABA.size(0)):
            tmp = x_ABA[i].detach().cpu()
            tmp = torchvision.transforms.ToPILImage()(tmp)
            tmp.save('{}/{}_A1/{}.png'.format(path, idx, i))

        print("[*] Samples saved: {}".format(x_AB_path))
        if tf_board:
            writer.add_image('x_A1f', x_AB[:16], idx)
            writer.add_image('x_A1valid', inputs[:16], idx)
            writer.add_image('x_A1G', x_ABA[:16], idx)
            writer.add_image('x_A1rec', x_ABAB[:16], idx)

    def generate_with_B(self, inputs, input_ref, path, idx=None, tf_board=True):
        x_BA = self.F(inputs)
        x_BAB = self.G(x_BA, input_ref)
        x_ABAf = self.F(x_BAB)
        x_ABAB = self.G(x_ABAf, inputs)

        x_BA_path = '{}/{}_x_A2G.png'.format(path, idx)

        vutils.save_image(x_BAB.data, x_BA_path)
        if not os.path.isdir('{}/{}_A2'.format(path, idx)):
            os.makedirs('{}/{}_A2'.format(path, idx))
        for i in range(x_BAB.size(0)):
            tmp = x_BAB[i].detach().cpu()
            tmp = torchvision.transforms.ToPILImage()(tmp)
            tmp.save('{}/{}_A2/{}.png'.format(path, idx, i))

        print("[*] Samples saved: {}".format(x_BA_path))
        if tf_board:
            writer.add_image('x_A2f', x_BA[:16], idx)
            writer.add_image('x_A2valid', inputs[:16], idx)
            writer.add_image('x_A2G', x_BAB[:16], idx)
            writer.add_image('x_A2rec', x_ABAB[:16], idx)

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
        x_A1, x_B1 = torch.Tensor(np.load('./valid_x_A1.npy')), torch.Tensor(np.load('./valid_x_B1.npy'))
        x_A1, x_B1 = self._get_variable(x_A1), self._get_variable(x_B1)
        x_A2, x_B2=torch.Tensor(np.load('./valid_x_A2.npy')), torch.Tensor(np.load('./valid_x_B2.npy'))
        x_A2, x_B2 = self._get_variable(x_A2), self._get_variable(x_B2)

        test_dir = os.path.join(self.model_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        self.generate_with_A(x_A1, x_A2, test_dir, idx=step)
        self.generate_with_B(x_A2, x_A1, test_dir, idx=step)

        return 0

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
