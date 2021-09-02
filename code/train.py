from __future__ import print_function
import argparse
import os
import numpy as np
import sys
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from model import *
from DataLoader_train import get_Training_Set


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--threads", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=512, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

# Loss weight of L1 pixel-wise loss
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

generator = GeneratorUNet()
discriminator = Discriminator()
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
torch.cuda.set_device(1)

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_pixelwise_1 = criterion_pixelwise.cuda()

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

print('===> Loading datasets')
dataloader = get_Training_Set()
training_data_loader = DataLoader(dataset=dataloader, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def train(epoch):
    epoch_loss_D = 0
    epoch_loss_G = 0
    epoch_loss_Pixel = 0
    epoch_loss_GAN = 0
    for i, batch in enumerate(training_data_loader):
        real_A = Variable(batch["B"])
        real_B = Variable(batch["A"])
        mean_n = Variable(batch["C"])
        var_n = Variable(batch["D"])
        if cuda:
            real_A = real_A.cuda()
            real_B = real_B.cuda()
        valid = torch.tensor(np.ones((real_A.size(0), *patch)), requires_grad=False).float().cuda()
        fake = torch.tensor(np.zeros((real_A.size(0), *patch)), requires_grad=False).float().cuda()

        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G.zero_grad()
        # GAN loss
        fake_B = generator(real_A)
        pred_fake = discriminator(fake_B, real_A)
        pred_fake = pred_fake.float()
        loss_GAN = criterion_GAN(pred_fake, valid)
        # Calculate two losses simultaneously : one between real content and predicted content, and the other between real noise and predicted noise, and their weights are equal
        loss_pixel = criterion_pixelwise(fake_B, real_B)
        # Total loss
        loss_G = loss_GAN + lambda_pixel * loss_pixel
        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        # Real loss
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)
        # Fake loss
        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)
        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        optimizer_D.step()

        epoch_loss_D += loss_D.item()
        epoch_loss_G += loss_G.item()
        epoch_loss_Pixel += loss_pixel.item()
        epoch_loss_GAN += loss_GAN.item()

        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] "
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_pixel.item(),
                loss_GAN.item(),
            )
        )

    if int(epoch) > 150:
        torch.save(generator.state_dict(), r"../result/saved_models/generator_%d.pth" % epoch)

for epoch in range(opt.epoch, opt.n_epochs):
    train(epoch)

# Save model checkpoints
torch.save(generator.state_dict(), r"../result/saved_models/generator_%d.pth" % epoch)
torch.save(discriminator.state_dict(), r"../result/saved_models/discriminator_%d.pth" % epoch)