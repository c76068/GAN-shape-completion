import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

#import cv2

from models import *
from datasets import *


parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=int, default=0, help="set it to 1 for running on GPU, 0 for CPU")
parser.add_argument("--input_dir", default=None, help="path to the testing data")
parser.add_argument("--model_path", default=None, help="path to the trained model")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")

opt = parser.parse_args()


test_data_name = opt.input_dir.split('/')[-1]
train_model_dir = opt.model_path.split('/')[1]
model_name = opt.model_path.split('.')[0].split('/')[-1]
save_name =  '_'.join([train_model_dir, model_name, test_data_name])
out_img_dir = "output_imgs/%s/%s" % (train_model_dir, save_name)
out_pred_dir = "saved_preds/%s" % train_model_dir
os.makedirs(out_img_dir , exist_ok=True)
os.makedirs(out_pred_dir, exist_ok=True)

dic = torch.load(opt.model_path)
generator = GeneratorUNet(in_channels=1, out_channels=1)
generator.load_state_dict(dic)

if torch.cuda.is_available() and opt.cuda == 1:
    device = "cuda:0"
else:
    device = "cpu"

generator = generator.to(device).eval()

dataloader = DataLoader(
    simpleDataset(opt.input_dir),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,
)

env, pred, gd_truth = [], [], []
for inputs, targets in dataloader:
    inputs, targets = inputs.to(device), targets.to(device)
    with torch.no_grad():
        fake_gen = torch.sigmoid(generator(inputs))
        fake_gen = (fake_gen>.5)*1.0
        fake_gen = fake_gen.cpu()

    env.append(inputs.cpu())
    pred.append(fake_gen)
    gd_truth.append(targets.cpu())

env = torch.cat(env,0)
pred = torch.cat(pred,0)
gd_truth = torch.cat(gd_truth,0)

save_path = os.path.join(out_pred_dir, save_name+'.pth')
torch.save({'env':env, 'pred':pred, 'gd_truth':gd_truth, 'paths': dataloader.dataset.paths}, save_path)


for i in range(env.shape[0]):
    sample_name = dataloader.dataset.paths[i].split('/')[-1].split('.')[0]
    save_image(env[i,:,:,:].unsqueeze(0), os.path.join(out_img_dir,"%s_env.png" % sample_name), normalize=True)
    save_image(pred[i,:,:,:].unsqueeze(0), os.path.join(out_img_dir,"%s_pred.png" % sample_name), normalize=True)
    save_image(gd_truth[i,:,:,:].unsqueeze(0), os.path.join(out_img_dir,"%s_true.png" % sample_name), normalize=True)
    