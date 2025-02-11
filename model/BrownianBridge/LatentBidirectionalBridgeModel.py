import itertools
import pdb
import random
import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm

from model.BrownianBridge.BidirectionalBridgeModel import BidirectionalBridgeModel
from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler
from model.VQGAN.vqgan import VQModel
from diffusers.models import AutoencoderKL


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LatentBidirectionalBridgeModel(BidirectionalBridgeModel):
    def __init__(self, model_config):
        super().__init__(model_config)
        if self.condition_key == 'vae':
            ckpt_path = "stabilityai/sd-vae-ft-mse"
            self.vqmodel = AutoencoderKL.from_pretrained(ckpt_path)
            self.vqmodel = self.vqmodel.eval()
            self.vqmodel.train = disabled_train
            for param in self.vqmodel.parameters():
                param.requires_grad = False
        else:
            self.vqmodel = VQModel(**vars(model_config.VQGAN.params)).eval()
            self.vqmodel.train = disabled_train
            for param in self.vqmodel.parameters():
                param.requires_grad = False
            print(f"load vqgan from {model_config.VQGAN.params.ckpt_path}")

        # Condition Stage Model
        if self.condition_key == 'nocond':
            self.cond_stage_model = None
        elif self.condition_key == 'SpatialRescaler':
            self.cond_stage_model = SpatialRescaler(**vars(model_config.CondStageParams))
        else:
            self.cond_stage_model = self.vqmodel

    def get_ema_net(self):
        return self

    def get_parameters(self):
        if self.condition_key == 'SpatialRescaler':
            print("get parameters to optimize: SpatialRescaler, UNet")
            params = itertools.chain(self.denoise_fn.parameters(), self.cond_stage_model.parameters())
        else:
            print("get parameters to optimize: UNet")
            params = self.denoise_fn.parameters()
        return params

    def apply(self, weights_init):
        super().apply(weights_init)
        if self.condition_key == 'SpatialRescaler':
            self.cond_stage_model.apply(weights_init)
        return self

    def forward(self, x, x_cond, context=None):
        with torch.no_grad():
            x_latent = self.encode(x, cond=False)
            x_cond_latent = self.encode(x_cond, cond=True)

        context, class_cond = self.get_cond_stage_context(x, x_cond)
        x_latent = x_latent.detach()
        x_cond_latent = x_cond_latent.detach()
        
        b, c, h, w, device, img_size, = *x_latent.shape, x_latent.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x_latent, x_cond_latent, context, t, class_cond=class_cond)

    def get_cond_stage_context(self, x, y):
        class_cond = None
        device = x.device
        
        if self.condition_key == 'nocond':
            context = None
        
        elif self.condition_key == 'SpatialRescaler':
            mask = torch.randint(0, 2, (x.shape[0],), device=device)           # mask [0,1]
            x = self.cond_stage_model.encode(x)
            y = self.cond_stage_model.encode(y)
            
            x_masked = x * mask[:, None, None, None]
            y_masked = y * torch.abs(1 - mask[:, None, None, None])
            context = torch.cat((x_masked, y_masked), dim=1)
        
        elif self.condition_key == 'indicator_1':
            with torch.no_grad():
                class_cond = torch.randint(0, 2, (x.shape[0],), device=device)          # mask [0,1]
                x = self.encode(x, cond=False).detach()
                y = self.encode(y, cond=True).detach()
            
                mask = class_cond.view(x.shape[0], 1, 1, 1).expand(x.shape).bool()      
                context = torch.where(mask, x, y)                                           # 1 is x (forward), 0 is y (backward)

        elif self.condition_key == 'indicator_2':
            with torch.no_grad():
                class_cond = torch.randint(0, 2, (x.shape[0],), device=device)
                x = self.encode(x, cond=False).detach()
                y = self.encode(y, cond=True).detach()
                
                x_masked = x * class_cond[:, None, None, None]
                y_masked = y * torch.abs(1 - class_cond[:, None, None, None])
            
            context = torch.cat((x_masked, y_masked), dim=1)                                # 1 is x (forward), 
                                                                                            # 0 is y (backward)
        else:   
            '''
            self.condition_key == 'vae'
            self.condition_key == 'dual'
            '''
            with torch.no_grad():
                mask = torch.randint(0, 2, (x.shape[0],), device=device)           # mask [0,1]
                x = self.encode(x, cond=False).detach()
                y = self.encode(y, cond=True).detach()
            
                x_masked = x * mask[:, None, None, None]
                y_masked = y * torch.abs(1 - mask[:, None, None, None])
            
            context = torch.cat((x_masked, y_masked), dim=1)
        return context, class_cond

    @torch.no_grad()
    def encode(self, x, cond=True, normalize=None): 
        if isinstance(self.vqmodel, VQModel):
            normalize = self.model_config.normalize_latent if normalize is None else normalize
            model = self.vqmodel
            x_latent = model.encoder(x)
            if not self.model_config.latent_before_quant_conv:
                x_latent = model.quant_conv(x_latent)
            if normalize:
                if cond:
                    x_latent = (x_latent - self.cond_latent_mean) / self.cond_latent_std
                else:
                    x_latent = (x_latent - self.ori_latent_mean) / self.ori_latent_std
            return x_latent

        elif isinstance(self.vqmodel, AutoencoderKL):
            model = self.vqmodel
            x_latent = model.encode(x).latent_dist.sample().mul_(0.18215)
            return x_latent

    @torch.no_grad()
    def decode(self, x_latent, cond=True, normalize=None):
        if isinstance(self.vqmodel, VQModel):
            normalize = self.model_config.normalize_latent if normalize is None else normalize
            if normalize:
                if cond:
                    x_latent = (x_latent) * self.cond_latent_std + self.cond_latent_mean
                else:
                    x_latent = (x_latent) * self.ori_latent_std + self.ori_latent_mean
            model = self.vqmodel
            if self.model_config.latent_before_quant_conv:
                x_latent = model.quant_conv(x_latent)
            x_latent_quant, loss, _ = model.quantize(x_latent)
            out = model.decode(x_latent_quant)
            return out
        
        elif isinstance(self.vqmodel, AutoencoderKL):
            model = self.vqmodel
            out = model.decode(x_latent / 0.18215).sample
            return out
            
        
        
    @torch.no_grad()   
    def p_sample_loop_backward(self, y, context=None, clip_denoised=False, sample_mid_step=False, class_cond=None):
        y_latent = self.encode(y, cond=True)
        if sample_mid_step:
            temp, one_step_temp = super().p_sample_loop_backward(y=y_latent,
                                                     context=context,
                                                     clip_denoised=clip_denoised,
                                                     sample_mid_step=sample_mid_step,
                                                     class_cond=class_cond)
            out_samples = []
            for i in tqdm(range(len(temp)), initial=0, desc="save output sample mid steps", dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(temp[i].detach(), cond=False)
                out_samples.append(out.to('cpu'))
                
            one_step_samples = []
            for i in tqdm(range(len(one_step_temp)), initial=0, desc="save one step sample mid steps",
                          dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(one_step_temp[i].detach(), cond=False)
                one_step_samples.append(out.to('cpu'))
            return out_samples, one_step_samples
            
        else:
            temp = super().p_sample_loop_backward(y=y_latent,
                                            context=context,
                                            clip_denoised=clip_denoised,
                                            sample_mid_step=sample_mid_step,
                                            class_cond=class_cond)
            x_latent = temp
            out = self.decode(x_latent, cond=False)
            return out
    
    @torch.no_grad()
    def sample_backward(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        class_cond = None
        
        if self.condition_key == 'dual' or self.condition_key == 'vae':
            with torch.no_grad():
                y_latent = self.encode(y, cond=True)
                x0_null = torch.zeros_like(y_latent)
            context = torch.cat((x0_null, y_latent), dim=1)
            
        elif self.condition_key == 'indicator_1':
            with torch.no_grad():
                y_latent = self.encode(y, cond=True)
            class_cond = torch.zeros((y.shape[0],), dtype=int, device=y_latent.device)                         #indicator = 0 (backward)
            context = y_latent
            
        elif self.condition_key == 'indicator_2':
            with torch.no_grad():
                y_latent = self.encode(y, cond=True)
                x0_null = torch.zeros_like(y_latent)
            class_cond = torch.zeros((y.shape[0],), dtype=int, device=y_latent.device)                         #indicator = 0 (backward)
            context = torch.cat((x0_null, y_latent), dim=1)
        return self.p_sample_loop_backward(y, context, clip_denoised, sample_mid_step, class_cond)
    
    
    
    @torch.no_grad()   
    def p_sample_loop_forward(self, x, context=None, clip_denoised=False, sample_mid_step=False, class_cond=None):
        x_latent = self.encode(x, cond=False)
        if sample_mid_step:
            temp, one_step_temp = super().p_sample_loop_forward(x=x_latent,
                                                     context=context,
                                                     clip_denoised=clip_denoised,
                                                     sample_mid_step=sample_mid_step,
                                                     class_cond=class_cond)
            out_samples = []
            for i in tqdm(range(len(temp)), initial=0, desc="save output sample mid steps", dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(temp[i].detach(), cond=True)
                out_samples.append(out.to('cpu'))
                
            one_step_samples = []
            for i in tqdm(range(len(one_step_temp)), initial=0, desc="save one step sample mid steps",
                          dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(one_step_temp[i].detach(), cond=True)
                one_step_samples.append(out.to('cpu'))
            return out_samples, one_step_samples
            
        else:
            temp = super().p_sample_loop_forward(x=x_latent,
                                            context=context,
                                            clip_denoised=clip_denoised,
                                            sample_mid_step=sample_mid_step,
                                            class_cond=class_cond)
            y_latent = temp
            out = self.decode(y_latent, cond=True)
            return out

    @torch.no_grad()
    def sample_forward(self, x0, context=None, clip_denoised=True, sample_mid_step=False):
        class_cond = None
        
        if self.condition_key == 'dual' or self.condition_key == 'vae':
            with torch.no_grad():
                x0_latent = self.encode(x0, cond=False)
                y_null = torch.zeros_like(x0_latent)
            context = torch.cat((x0_latent, y_null), dim=1)
            
        elif self.condition_key == 'indicator_1':
            with torch.no_grad():
                x0_latent = self.encode(x0, cond=False)
            class_cond = torch.ones((x0.shape[0],), dtype=int, device=x0_latent.device)                             #indicator = 1 (forward)
            context = x0_latent
            
        elif self.condition_key == 'indicator_2':
            with torch.no_grad():
                x0_latent = self.encode(x0, cond=False)
                y_null = torch.zeros_like(x0_latent)
            class_cond = torch.ones((x0.shape[0],), dtype=int, device=x0_latent.device)                             #indicator = 1 (forward)
            context = torch.cat((x0_latent, y_null), dim=1)
        return self.p_sample_loop_forward(x0, context, clip_denoised, sample_mid_step, class_cond)

    @torch.no_grad()
    def sample_vqgan(self, x):
        x_rec, _ = self.vqmodel(x)
        return x_rec
