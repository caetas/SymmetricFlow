##############################################################################################
############ Code based on: https://bm371613.github.io/conditional-flow-matching/ ############
### and https://github.com/openai/guided-diffusion                                         ###
##############################################################################################

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import zuko
from torchvision.utils import make_grid
import torch.nn.functional as F
from functools import partial
import math
import wandb
from config import models_dir
import os
from torchdiffeq import odeint
from diffusers.models import AutoencoderKL
from accelerate import Accelerator
from collections import OrderedDict
import copy
from abc import abstractmethod
import cv2
from utils.masks import mask_to_class
from torchmetrics import JaccardIndex
from lpips import LPIPS
from sklearn.metrics import jaccard_score
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

@torch.no_grad()
def update_ema(ema_model, model, decay=0.5):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    
    for name, param in model_params.items():
        # if name contains "module" then remove module
        if "module" in name:
            name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def sd_null_condition():
    text = ""
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="tokenizer")
    with torch.no_grad():
        empty_inputs = tokenizer(text, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        emptyembed = text_encoder(empty_inputs.input_ids)[0]
    del text_encoder, tokenizer
    return emptyembed


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_head_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(self,):
        super().__init__()

        self.unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="unet")
        self.null_condition = sd_null_condition()

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        prompt_embeds = self.null_condition.repeat(x.shape[0], 1, 1).to(x.device)
        return self.unet(x, timesteps, prompt_embeds).sample
        

def create_checkpoint_dir():
    '''
    Create a directory to save the model checkpoints
    '''
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(os.path.join(models_dir, 'SemFlow')):
        os.makedirs(os.path.join(models_dir, 'SemFlow'))

class SemFlow(nn.Module):

    def __init__(self, args, img_size=32, in_channels=3):
        '''
        SemFlow module
        :param args: arguments
        :param img_size: size of the image
        :param in_channels: number of input channels
        '''
        super(SemFlow, self).__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vae =  AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").eval().to(self.device) if args.latent else None
        self.channels = in_channels
        self.img_size = img_size

        # If using VAE, change the number of channels and image size accordingly
        if self.vae is not None:
            self.channels = 4
            self.img_size = self.img_size // 8

        self.model = UNetModel()
        self.model.to(self.device)
        self.lr = args.lr
        self.n_epochs = args.n_epochs
        self.sample_and_save_freq = args.sample_and_save_freq
        self.dataset = args.dataset
        self.solver = args.solver
        self.step_size = args.step_size
        self.solver_lib = args.solver_lib
        self.no_wandb = args.no_wandb
        self.warmup = args.warmup
        self.decay = args.decay
        self.snapshot = args.n_epochs//args.snapshots
        self.beta = args.beta
        self.image_weight = args.image_weight
        if args.train:
            self.ema = copy.deepcopy(self.model)
            self.ema_rate = args.ema_rate
            for param in self.ema.parameters():
                param.requires_grad = False

    def forward(self, x, t):
        '''
        Forward pass of the SemFlow module
        :param x: input image
        :param t: time
        '''
        return self.model(x, t)
    
    def semflow_loss(self, x, mask):
        '''
        Symmetrical flow matching loss
        :param x: input image
        :param mask: mask
        Returns:
        - Image Generation Loss, Mask Generation Loss
        '''
        sigma_min = 1e-4
        t = torch.rand(x.shape[0], device=x.device)

        x_t = (1 - (1 - sigma_min) * t[:, None, None, None]) * mask + t[:, None, None, None] * x

        optimal_flow = x - (1 - sigma_min) * mask
        
        predicted_flow = self.forward(x_t, t)

        return (predicted_flow - optimal_flow).square().mean()
    
    @torch.no_grad()
    def encode(self, x):
        '''
        Encode the input image
        :param x: input image
        '''
        # check if it is a distributted model or not
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.vae.module.encode(x)
        else:
            return self.vae.encode(x)
        
    @torch.no_grad()    
    def decode(self, z):
        '''
        Decode the input image
        :param z: input image
        '''
        # check if it is a distributted model or not
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.vae.module.decode(z)
        else:
            return self.vae.decode(z)
    
    @torch.no_grad()
    def sample(self, n_samples, mask, train=True, accelerate=None, fid=False):
        '''
        Sample images
        :param n_samples: number of samples
        :param mask: mask
        :param train: if True, sample during training
        :param accelerate: Accelerator object
        :param fid: if True, return the samples
        '''
        x_0 = mask.clone()

        if train:
            def f(t: float, x):
                return self.ema(x, torch.full(x.shape[:1], t, device=self.device))
        else:
            def f(t: float, x):
                return self.forward(x, torch.full(x.shape[:1], t, device=self.device))
        
        if self.solver_lib == 'torchdiffeq':
            if self.solver == 'euler' or self.solver == 'rk4' or self.solver == 'midpoint' or self.solver == 'explicit_adams' or self.solver == 'implicit_adams':
                samples = odeint(f, x_0, t=torch.linspace(0, 1, 2).to(self.device), options={'step_size': self.step_size}, method=self.solver, rtol=1e-5, atol=1e-5)
            else:
                samples = odeint(f, x_0, t=torch.linspace(0, 1, 2).to(self.device), method=self.solver, options={'max_num_steps': 1//self.step_size}, rtol=1e-5, atol=1e-5)
            samples = samples[1]
        elif self.solver_lib == 'zuko':
            samples = zuko.utils.odeint(f, x_0, 0, 1, phi=self.model.parameters(), atol=1e-5, rtol=1e-5)
        else:
            t=0
            for i in tqdm(range(int(1/self.step_size)), desc='Sampling', leave=False):
                if train:
                    v = self.ema(x_0, torch.full(x_0.shape[:1], t, device=self.device))
                else:
                    v = self.forward(x_0, torch.full(x_0.shape[:1], t, device=self.device))
                x_0 = x_0 + self.step_size * v
                t += self.step_size
            samples = x_0
        
        if self.vae is not None:
            samples = self.decode(samples / 0.18215).sample
            mask = self.decode(mask / 0.18215).sample

        if fid:
            return samples

        samples = samples*0.5 + 0.5
        samples = samples.clamp(0, 1).float()
        mask = mask*0.5 + 0.5
        mask = mask.clamp(0, 1).float()
        
        fig = plt.figure(figsize=(20, 10))
        grid_mask = make_grid(mask, nrow=int(n_samples**0.5), padding=0)
        grid = make_grid(samples, nrow=int(n_samples**0.5), padding=0)
        plt.subplot(1, 2, 1)
        plt.imshow(grid_mask.permute(1, 2, 0).cpu().detach().numpy())
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(grid.permute(1, 2, 0).cpu().detach().numpy())
        plt.axis('off')

        if train:
            if not self.no_wandb:
                accelerate.log({"samples": fig})
        else:
            plt.show()

        plt.close(fig)

    @torch.no_grad()
    def segment(self, n_samples, x, train=True, accelerate=None, eval=False, fine_tune=False):
        '''
        Segment images
        :param n_samples: number of samples
        :param x: input image that we want to segment
        :param train: if True, sample during training
        :param accelerate: Accelerator object
        '''
        x_0 = x.clone()

        if train:
            def f(t: float, x):
                return self.ema(x, torch.full(x.shape[:1], t, device=self.device))
        else:
            def f(t: float, x):
                return self.forward(x, torch.full(x.shape[:1], t, device=self.device))
        
        if self.solver_lib == 'torchdiffeq':
            if self.solver == 'euler' or self.solver == 'rk4' or self.solver == 'midpoint' or self.solver == 'explicit_adams' or self.solver == 'implicit_adams':
                samples = odeint(f, x_0, t=torch.linspace(1, 0, 2).to(self.device), options={'step_size': self.step_size}, method=self.solver, rtol=1e-5, atol=1e-5)
            else:
                samples = odeint(f, x_0, t=torch.linspace(1, 0, 2).to(self.device), method=self.solver, options={'max_num_steps': 1//self.step_size}, rtol=1e-5, atol=1e-5)
            samples = samples[1]
        elif self.solver_lib == 'zuko':
            samples = zuko.utils.odeint(f, x_0, 1, 0, phi=self.model.parameters(), atol=1e-5, rtol=1e-5)
        else:
            t=1
            for i in tqdm(range(int(1/self.step_size)), desc='Sampling', leave=False):
                if train:
                    v = self.ema(x_0, torch.full(x_0.shape[:1], t, device=self.device))
                else:
                    v = self.forward(x_0, torch.full(x_0.shape[:1], t, device=self.device))
                x_0 = x_0 - self.step_size * v
                t -= self.step_size
            samples = x_0

        if fine_tune:
            return samples
        
        if self.vae is not None:
            samples = self.decode(samples / 0.18215).sample
            x = self.decode(x / 0.18215).sample
    
        if eval:
            return samples

        samples = samples*0.5 + 0.5
        samples = samples.clamp(0, 1).float()
        x = x*0.5 + 0.5
        x = x.clamp(0, 1).float()

        # plot two grids side by side, one with the original image and the other with the segmented image
        fig = plt.figure(figsize=(20, 10))
        grid_x = make_grid(x, nrow=int(n_samples**0.5), padding=0)
        grid = make_grid(samples, nrow=int(n_samples**0.5), padding=0)
        plt.subplot(1, 2, 1)
        plt.imshow(grid_x.permute(1, 2, 0).cpu().detach().numpy())
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(grid.permute(1, 2, 0).cpu().detach().numpy())
        plt.axis('off')
        if train:
            if not self.no_wandb:
                accelerate.log({"segmentations": fig})
                plt.close(fig)
        else:
            plt.show()

    def dequantize_mask(self, mask):
        '''
        Dequantize the mask
        :param mask: mask
        :param beta: beta value
        '''
        # noise should be -beta/2 to beta/2 if the image is in the range of 0 to 255
        noise = torch.rand_like(mask) * self.beta - (self.beta/2)
        noise = noise/127.5
        mask = mask + noise

        return mask    

    
    def train_model(self, train_loader, val_loader, verbose=True):
        '''
        Train the model
        :param train_loader: training data loader
        '''
        accelerate = Accelerator(log_with="wandb")
        if not self.no_wandb:
            accelerate.init_trackers(project_name='SemFlow',
            config = {
                        "dataset": self.args.dataset,
                        "batch_size": self.args.batch_size,
                        "n_epochs": self.args.n_epochs,
                        "lr": self.args.lr,
                        "channels": self.channels,
                        "input_size": self.img_size,
                        'model_channels': self.args.model_channels,
                        'num_res_blocks': self.args.num_res_blocks,
                        'attention_resolutions': self.args.attention_resolutions,
                        'dropout': self.args.dropout,
                        'channel_mult': self.args.channel_mult,
                        'conv_resample': self.args.conv_resample,
                        'dims': self.args.dims,
                        'num_heads': self.args.num_heads,
                        'num_head_channels': self.args.num_head_channels,
                        'use_scale_shift_norm': self.args.use_scale_shift_norm,
                        'resblock_updown': self.args.resblock_updown,
                        'use_new_attention_order': self.args.use_new_attention_order,  
                        "ema_rate": self.args.ema_rate,
                        "warmup": self.args.warmup,
                        "latent": self.args.latent,
                        "decay": self.args.decay,
                        "size": self.args.size,   
                },
                init_kwargs={"wandb":{"name": f"SemFlow_{self.args.dataset}"}})

        epoch_bar = tqdm(range(self.n_epochs), desc='Epochs', leave=True)
        create_checkpoint_dir()

        best_loss = float('inf')

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, total_steps=self.n_epochs*len(train_loader), pct_start=self.warmup/self.n_epochs, anneal_strategy='cos', cycle_momentum=False, div_factor=self.lr/1e-6, final_div_factor=1)

        if  self.vae is None:
            train_loader, self.model, optimizer, scheduler, self.ema, val_loader = accelerate.prepare(train_loader, self.model, optimizer, scheduler, self.ema, val_loader)
        else:
            train_loader, self.model, optimizer, scheduler, self.ema, self.vae, val_loader = accelerate.prepare(train_loader, self.model, optimizer, scheduler, self.ema, self.vae, val_loader)


        update_ema(self.ema, self.model, 0)

        for epoch in epoch_bar:
            self.model.train()
            train_loss = 0.0
            for x, mask in tqdm(train_loader, desc='Batches', leave=False, disable=not verbose):
                x = x.to(self.device)
                mask = self.dequantize_mask(mask)
                mask = mask.to(self.device)

                with accelerate.autocast():

                    if self.vae is not None:
                        with torch.no_grad():
                            # if x has one channel, make it 3 channels
                            if x.shape[1] == 1:
                                x = torch.cat((x, x, x), dim=1)
                                mask = torch.cat((mask, mask, mask), dim=1)
                            #x = self.vae.module.encode(x).latent_dist.sample().mul_(0.18215)
                            x = self.encode(x).latent_dist.sample().mul_(0.18215)
                            #mask = self.vae.module.encode(mask).latent_dist.mode().mul_(0.18215)
                            mask = self.encode(mask).latent_dist.mode().mul_(0.18215)

                    optimizer.zero_grad()
                    loss = self.semflow_loss(x, mask)
                    accelerate.backward(loss)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item() * x.shape[0]
                update_ema(self.ema, self.model, self.ema_rate)
            
            accelerate.wait_for_everyone()

            if not self.no_wandb:
                accelerate.log({"Train Loss": train_loss / len(train_loader.dataset)})
                accelerate.log({"Learning Rate": scheduler.get_last_lr()[0]})

            epoch_bar.set_postfix({'Loss': (train_loss) / len(train_loader.dataset)})

            if (epoch+1) % self.sample_and_save_freq == 0 or epoch == 0:
                self.model.eval()
                # one batch from the validation loader
                x, mask = next(iter(val_loader))
                x = x.to(self.device)
                mask = self.dequantize_mask(mask)
                mask = mask.to(self.device)
                if self.vae is not None:
                    with torch.no_grad():
                        if x.shape[1] == 1:
                            x = torch.cat((x, x, x), dim=1)
                            mask = torch.cat((mask, mask, mask), dim=1)
                        x = self.encode(x).latent_dist.sample().mul_(0.18215)
                        mask = self.encode(mask).latent_dist.mode().mul_(0.18215)
                self.sample(x.shape[0], mask, accelerate=accelerate)
                self.segment(x.shape[0], x, accelerate=accelerate)
            
            if (epoch+1) % self.snapshot == 0:
                ema_to_save = accelerate.unwrap_model(self.ema)
                accelerate.save(ema_to_save.state_dict(), os.path.join(models_dir, 'SemFlow', f"{'LatFM' if self.vae is not None else 'FM'}_{self.dataset}_beta{self.beta}_epoch{epoch+1}.pt"))

        accelerate.end_training()

    def load_checkpoint(self, checkpoint_path):
        '''
        Load a model checkpoint
        :param checkpoint_path: path to the checkpoint
        '''
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path, weights_only=False))

    @torch.no_grad()
    def evaluate_segmentation(self, dataloader):
        '''
        Evaluate the segmentation
        :param dataloader: data loader
        '''
        self.model.eval()
        #self.vae.decoder.load_state_dict(torch.load(os.path.join(models_dir, 'vae_decoder_step_5000_coco_384_mse.pt'), weights_only=False))
        self.vae.eval()

        accelerate = Accelerator()
        self.model, self.vae, dataloader = accelerate.prepare(self.model, self.vae, dataloader)

        gt = []
        pred = []
        for x, mask in tqdm(dataloader, desc='Evaluating', leave=True):
            x = x.to(self.device)
            mask = mask.to(self.device)
            gt.append(mask_to_class(mask, self.args.dataset).cpu())

            with accelerate.autocast():

                if self.vae is not None:
                    with torch.no_grad():
                        if x.shape[1] == 1:
                            x = torch.cat((x, x, x), dim=1)
                            mask = torch.cat((mask, mask, mask), dim=1)
                        x = self.encode(x).latent_dist.sample().mul_(0.18215)

                predicted_masks = self.segment(x.shape[0], x, train=False, eval=True)
            pred.append(mask_to_class(predicted_masks.float(), self.args.dataset).cpu())

        gt = torch.cat(gt)
        pred = torch.cat(pred)

        if self.args.dataset == 'coco':
            metric = JaccardIndex(task='multiclass', num_classes=172, ignore_index=171)
            pred[gt == 171] = 171
        else:
            metric = JaccardIndex(task='multiclass', num_classes=19, ignore_index=0)
            pred[gt == 0] = 0

        miou = metric(pred, gt)

        print(f"mIoU: {miou.item()}")

        miou = metric(gt, pred)
        print(f"mIoU: {miou.item()}")

        gt_flat = gt.view(-1).cpu().numpy()
        pred_flat = pred.view(-1).cpu().numpy()
        miou = jaccard_score(gt_flat, pred_flat, average='macro')
        print(f"mIoU: {miou}")
        #ignore index 171
        if self.args.dataset == 'coco':
            miou = jaccard_score(gt_flat, pred_flat, average='macro', labels=[i for i in range(171)])
        else:
            miou = jaccard_score(gt_flat, pred_flat, average='macro', labels=[i for i in range(1,19)])
        print(f"mIoU: {miou}")

        # creaste a directory to save the results
        if not os.path.exists('./../../results'):
            os.makedirs('./../../results')
        if not os.path.exists(f'./../../results/{self.dataset}'):
            os.makedirs(f'./../../results/{self.dataset}')
        
        # save the mIoU to a file
        with open(f'./../../results/{self.dataset}/fm_{self.solver_lib}_solver_{self.solver}_stepsize_{self.step_size}_miou.txt', 'w') as f:
            f.write(str(miou.item()))



    @torch.no_grad()
    def fid_sample(self, dataloader, batch_size=16):
        '''
        Sample images for FID calculation
        :param batch_size: batch size
        '''
        # if self.args.checkpoint contains epoch number, ep = epoch number
        # else, ep = 0
        if 'epoch' in self.args.checkpoint:
            ep = int(self.args.checkpoint.split('epoch')[1].split('.')[0])
        else:
            ep = 0

        if not os.path.exists('./../../fid_samples'):
            os.makedirs('./../../fid_samples')
        if not os.path.exists(f"./../../fid_samples/{self.dataset}"):
            os.makedirs(f"./../../fid_samples/{self.dataset}")
        #add solverlib, solver, stepsize
        if not os.path.exists(f"./../../fid_samples/{self.dataset}/fm_{self.solver_lib}_solver_{self.solver}_stepsize_{self.step_size}_ep{ep}"):
            os.makedirs(f"./../../fid_samples/{self.dataset}/fm_{self.solver_lib}_solver_{self.solver}_stepsize_{self.step_size}_ep{ep}")
        cnt = 0

        lpips_total = []

        lpips_loss = LPIPS(net='alex').to(self.device)
        lpips_loss.eval()

        accelerate = Accelerator()
        self.model, self.vae, dataloader = accelerate.prepare(self.model, self.vae, dataloader)

        for image, mask in tqdm(dataloader, desc='FID Sampling', leave=True):
            image = image.to(self.device)
            mask = mask.to(self.device)
            # repeat mask 17 times
            mask = mask.repeat(17, 1, 1, 1)
            # dequantize the mask
            mask = self.dequantize_mask(mask)

            with accelerate.autocast():

                if self.vae is not None:
                    with torch.no_grad():
                        if image.shape[1] == 1:
                            mask = torch.cat((mask, mask, mask), dim=1)
                        mask = self.encode(mask).latent_dist.mode().mul_(0.18215)
                

                samples = self.sample(mask.shape[0], mask, train=False, fid=True)
            samples = samples.float()

            # get lpips loss between samples and image
            loss = lpips_loss(samples, image).mean()
            lpips_total.append(loss.item())

            samples = samples*0.5 + 0.5
            samples = samples.clamp(0, 1)
            samples = samples.cpu().numpy()
            samples = (samples*255).astype(np.uint8)
            samples = samples.transpose(0, 2, 3, 1)

            for samp in samples:
                cv2.imwrite(f"./../../fid_samples/{self.dataset}/fm_{self.solver_lib}_solver_{self.solver}_stepsize_{self.step_size}_ep{ep}/{cnt}.png", cv2.cvtColor(samp, cv2.COLOR_RGB2BGR) if samp.shape[-1] == 3 else samp)
                cnt += 1
            
            if cnt >= 50000:
                break

        # save the lpips total mean to a file
        with open(f'./../../fid_samples/{self.dataset}/fm_{self.solver_lib}_solver_{self.solver}_stepsize_{self.step_size}_ep{ep}/lpips_total.txt', 'w') as f:
            f.write(str(np.mean(lpips_total)))
        print(f"LPIPS: {np.mean(lpips_total)}")
            
        '''


        for i in tqdm(range(50000//batch_size), desc='FID Sampling', leave=True):
            samps = self.sample(batch_size, train=False, fid=True).cpu().numpy()
            samps = (samps*255).astype(np.uint8)
            samps = samps.transpose(0, 2, 3, 1)
            for samp in samps:
                cv2.imwrite(f"./../../fid_samples/{self.dataset}/fm_{self.solver_lib}_solver_{self.solver}_stepsize_{self.step_size}_ep{ep}/{cnt}.png", cv2.cvtColor(samp, cv2.COLOR_RGB2BGR) if samp.shape[-1] == 3 else samp)
                cnt += 1 

        '''


