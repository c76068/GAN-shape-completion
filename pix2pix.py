import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=int, default=0, help="set it to 1 for running on GPU, 0 for CPU")
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--save_name", type=str, default="facades", help="name of saved model")
parser.add_argument("--dataset_folder", type=str,help="path of the data set folder")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--max_files", type=int, default=100000, help="number of max files")
parser.add_argument("--include_stoppage", type=int, default=1, help="Include images that greedy algorithm stopped")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--in_img_format", type=str, default="binary", help="input image format:binary or level_set_fun")
parser.add_argument("--in_frontier", type=int, default=0, help="input shadow boundries if 1")
parser.add_argument("--out_activation", type=str, default="sigmoid", help="activation for generator's outputs")
parser.add_argument("--pixel_loss", type=str, default="L1", help="pixel loss: L1 or BCE")
parser.add_argument("--GAN_loss", type=str, default="BCE", help="GAN loss: MSE or BCE")
parser.add_argument("--lambda_pixel", type=float, default="100.0", help="weight for pixel loss")
parser.add_argument(
    "--sample_interval", type=int, default=500, help="a positive integer N to output several images of results every N batches (default=500)"
)
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)


opt.save_name = opt.save_name + "_batch_size"+str(opt.batch_size) + "_pixel_loss"+opt.pixel_loss + "_lambda_pixel"+str(opt.lambda_pixel) + "_in_img_format"+str(opt.in_frontier)\
+ "_in_frontier"+str(opt.in_frontier) + "_include_stoppage"+str(opt.include_stoppage) + "_out_activation"+opt.out_activation
 

os.makedirs("images/%s" % opt.save_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.save_name, exist_ok=True)
torch.save(opt, "saved_models/%s/opt_train.pth" % opt.save_name)


# Loss functions
if opt.GAN_loss == "MSE":
    criterion_GAN = torch.nn.MSELoss()
elif opt.GAN_loss == "BCE":
    criterion_GAN = torch.nn.BCELoss()

if opt.pixel_loss == "L1":
    criterion_pixelwise = torch.nn.L1Loss()
elif opt.pixel_loss == "BCE":
    criterion_pixelwise = torch.nn.BCELoss()


# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# Initialize generator and discriminator
in_channels = 2 if opt.in_frontier else 1
generator = GeneratorUNet(in_channels=in_channels, out_channels=1)
discriminator = Discriminator(in_channels=in_channels+1)





if torch.cuda.is_available() and opt.cuda == 1:
    device = "cuda:0"
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    criterion_GAN.to(device)
    criterion_pixelwise.to(device)
else:
    device = "cpu"

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.save_name, opt.epoch)))
    discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (opt.save_name, opt.epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders and datasets
input_names = ['vis'] if opt.in_img_format == "binary" else ['horizons']
if opt.in_frontier:
    input_names.append("horizons") 

data = simpleDataset(opt.dataset_folder, max_files=opt.max_files, inputs=input_names, include_stoppage=bool(opt.include_stoppage))

dataloader = DataLoader(
    data,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

val_dataloader = DataLoader(
    data,
    batch_size=10,
    shuffle=True,
    num_workers=1,
)


#Output activation
if opt.out_activation.lower() == 'sigmoid':
    out_activation = nn.Sigmoid()
elif opt.out_activation.lower() == 'tanh':
    out_activation = nn.Tanh()

def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    inputs, targets = next(iter(val_dataloader))
    inputs, targets = inputs.to(device), targets.to(device)
    
    fake_gen = out_activation(generator(inputs))
    if opt.in_img_format.lower=='binary':
        fake_gen = (fake_gen>.5)*1.0
    elif opt.in_img_format.lower=='level_set_fun':
        fake_gen = (fake_gen>0.0)*1.0

    #img_sample = torch.cat((inputs.data, fake_gen.data, targets.data), 1)
    img_sample = torch.cat([inputs[:,i,:,:].unsqueeze(1) for i in range(inputs.shape[1])] + [fake_gen.data, targets.data], -2)
    save_image(img_sample, "images/%s/%s.png" % (opt.save_name, batches_done), nrow=5, normalize=True)


# ----------
#  Training
# ----------

prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, (inputs, targets) in enumerate(dataloader):

        # Model inputs
        inputs, targets = inputs.to(device), targets.to(device)

        # Adversarial ground truths
        valid = torch.ones((inputs.shape[0], *patch)).to(device)
        fake = torch.zeros((inputs.shape[0], *patch)).to(device)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        fake_gen = out_activation(generator(inputs))
        pred_fake = torch.sigmoid(discriminator(fake_gen, inputs))
        loss_GAN = criterion_GAN(pred_fake, valid)


        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_gen, targets)

        # Total loss
        loss_G = loss_GAN + opt.lambda_pixel * loss_pixel

        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = torch.sigmoid(discriminator(targets, inputs))
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = torch.sigmoid(discriminator(fake_gen.detach(), inputs))
        loss_fake = criterion_GAN(pred_fake, fake)


        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)
            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_pixel.item(),
                    loss_GAN.item(),
                    time_left,
                )
            )

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.save_name, epoch))
        torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.save_name, epoch))

torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.save_name, epoch))
torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.save_name, epoch))
