import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm.autonotebook import tqdm
import numpy as np

from model.utils import extract, default
from model.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel


class BidirectionalBridgeModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        # model hyperparameters
        model_params = model_config.BB.params
        self.num_timesteps = model_params.num_timesteps
        self.mt_type = model_params.mt_type
        self.max_var = model_params.max_var if model_params.__contains__("max_var") else 0
        self.eta = model_params.eta if model_params.__contains__("eta") else 1
        self.skip_sample = model_params.skip_sample
        self.sample_type = model_params.sample_type
        self.sample_step = model_params.sample_step
        self.steps = None
        self.register_schedule()
        self.p_threshold = model_config.p_threshold
        
        # loss and objective
        self.loss_type = model_params.loss_type
        self.objective = model_params.objective if model_params.__contains__("objective") else "noise"

        # UNet
        self.image_size = model_params.UNetParams.image_size
        self.channels = model_params.UNetParams.in_channels
        self.condition_key = model_params.UNetParams.condition_key

        self.denoise_fn = UNetModel(**vars(model_params.UNetParams))

    def register_schedule(self):
        T = self.num_timesteps

        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError
        m_tminus = np.append(0, m_t[:-1])

        variance_t = 2. * (m_t - m_t ** 2) * self.max_var           # var_t in [0.002 -> 0.5 -> 0.002]
        variance_tminus = np.append(0., variance_t[:-1])            # var_t_minus in [0, 0.002, ...]
        variance_t_tminus = 0
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t
        
        
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

        if self.skip_sample:
            if self.sample_type == 'linear':
                midsteps = torch.arange(self.num_timesteps - 1, 1,
                                        step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
                self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)          # 999 -> [1,0]
            
                self.asc_mid_steps = torch.flip(midsteps.clone(), dims=(0,))         # 1 -> 999
                self.asc_steps = torch.cat((torch.Tensor([0]).long(), self.asc_mid_steps[:-1], \
                    torch.Tensor([self.num_timesteps-2, self.num_timesteps-1]).long()), dim=0)
            
            elif self.sample_type == 'cosine':
                steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
                self.steps = torch.from_numpy(steps)
        else:
            self.steps = torch.arange(self.num_timesteps-1, -1, -1)

    def apply(self, weight_init):
        self.denoise_fn.apply(weight_init)
        return self

    def get_parameters(self):
        return self.denoise_fn.parameters()

    def forward(self, x, y, context=None):
        mask = None
        if self.condition_key == "nocond":
            context = None
        elif self.condition_key == 'dual':
            device = x.device
            # Masking one of two input
            mask = torch.randint(0, 2, (x.shape[0],), device=device)    # mask [0,1]
            x_masked = x * mask[:, None, None, None]                    # 1 -> input x
            y_masked = y * (1 - mask[:, None, None, None])              # 0 -> input y
            context = torch.cat((x_masked, y_masked), dim=1)            # (B,Cx2,H,W)
            del x_masked, y_masked
        else:
        # UNET also input y as context (starting point) for modeling !!!
            context = y if context is None else context
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, y, context, t, class_cond=None, mask=mask)

    def p_losses(self, x0, y, context, t, noise=None, class_cond=None, mask=None):
        """
        model loss
        :param x0: encoded x_ori, E(x_ori) = x0
        :param y: encoded y_ori, E(y_ori) = y
        :param y_ori: original source domain image
        :param t: timestep
        :param noise: Standard Gaussian Noise
        :return: loss
        """
        b, c, h, w = x0.shape
        noise = default(noise, lambda: torch.randn_like(x0))

        x_t, objective = self.q_sample(x0, y, t, noise)
        
        objective_recon = self.denoise_fn(x_t, timesteps=t, context=context, y=class_cond)

        x0_recon = None
        
        if self.loss_type == 'l1':
            recloss = (objective - objective_recon).abs().mean()
        elif self.loss_type == 'l2':
            recloss = F.mse_loss(objective, objective_recon)
        elif 'recon' in self.loss_type :
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
            xT_recon = self.predict_xT_from_objective(x_t, x0, t, objective_recon)
            if self.loss_type == 'recon':
                xT_loss = torch.nn.functional.mse_loss(xT_recon, y)
                x0_loss = torch.nn.functional.mse_loss(x0_recon, x0)
            elif self.loss_type == 'mask_recon':
                '''
                mask = 1 -> input x0 -> take loss on xT
                mask = 0 -> input xT -> take loss on x0
                '''
                xT_recon = xT_recon * mask[:, None, None, None]
                x0_recon = x0_recon * (1 - mask[:, None, None, None])
                
                y = y * mask[:, None, None, None]
                x0 = x0 * (1 - mask[:, None, None, None])
                
                xT_loss = torch.nn.functional.mse_loss(xT_recon, y)
                x0_loss = torch.nn.functional.mse_loss(x0_recon, x0)

            recloss = xT_loss + x0_loss
        else:
            raise NotImplementedError()
        
        if x0_recon is None:
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
        
        log_dict = {
            "loss": recloss,
            "x0_recon": x0_recon
        }
        return recloss, log_dict

    def q_sample(self, x0, y, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x0))
        m_t = extract(self.m_t, t, x0.shape)
        var_t = extract(self.variance_t, t, x0.shape)
        sigma_t = torch.sqrt(var_t)
        
        if self.objective == 'noise':
            objective = noise                                       # predict epsilon
        elif self.objective == 'sum':
            objective = y + x0
        elif self.objective == 'both':
            objective = torch.cat((x0,y), dim=1)
        else:
            raise NotImplementedError()    
            
        return (
            (1. - m_t) * x0 + m_t * y + sigma_t * noise,
            objective
        )

    def predict_x0_from_objective(self, x_t, y, t, objective_recon):
        if self.objective == 'noise':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            x0_recon = (x_t - m_t * y - sigma_t * objective_recon) / (1. - m_t)
        elif self.objective == 'sum':
            x0_recon = objective_recon - y
        elif self.objective == 'both':
            num_channel = x_t.shape[1]
            x0_recon = objective_recon[:,:num_channel,:,:]
        else:
            raise NotImplementedError()   
        return x0_recon
    
    def predict_xT_from_objective(self, x_t, x0, t, objective_recon):
        ##### CHECK HERE #####
        if self.objective == 'noise':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            xT_recon = (x_t - (1-m_t) * x0 - sigma_t * objective_recon) / (m_t)
        elif self.objective == 'sum':
            xT_recon = objective_recon - x0
        elif self.objective == 'both':
            num_channel = x_t.shape[1]
            xT_recon = objective_recon[:,num_channel:,:,:]
        else:
            raise NotImplementedError()   
        return xT_recon

    @torch.no_grad()
    def q_sample_loop(self, x0, y):
        imgs = [x0]
        for i in tqdm(range(self.num_timesteps), desc='q sampling loop', total=self.num_timesteps):
            t = torch.full((y.shape[0],), i, device=x0.device, dtype=torch.long)
            img, _ = self.q_sample(x0, y, t)
            imgs.append(img)
        return imgs

    @torch.no_grad()
    def p_sample_backward(self, x_t, y, context, i, clip_denoised=False, class_cond=None):
        b, *_, device = *x_t.shape, x_t.device
        if self.steps[i] == 0:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context, y=class_cond)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon, x0_recon
        else:
            '''
            self.steps[i] != 0
            => min(var_t) = 0.002
            '''    
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)       #t
            n_t = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)   #t-1

            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context, y=class_cond)        #epsilon_t
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)

            m_t = extract(self.m_t, t, x_t.shape)                                           #m_t
            m_nt = extract(self.m_t, n_t, x_t.shape)                                        #m_{t-1}
            
            '''
            t=0, var_t = 0.002
            '''
            var_t = extract(self.variance_t, t, x_t.shape)                                  
            var_nt = extract(self.variance_t, n_t, x_t.shape)
                                         
            delta2_t = self.eta * (var_t - var_nt * (((1. - m_t) / (1. - m_nt)) ** 2))
            sigma_t = torch.sqrt(delta2_t * var_nt / var_t)
            noise = torch.randn_like(x_t)
                        
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt(var_nt * (var_t - delta2_t)) / var_t * \
                            (x_t - (1. - m_t) * x0_recon - m_t * y)

            return x_tminus_mean + sigma_t * noise, x0_recon
        
    @torch.no_grad()
    def p_sample_loop_backward(self, y, context=None, clip_denoised=True, sample_mid_step=False, class_cond=None):
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context

        if sample_mid_step:
            imgs, one_step_imgs = [y], []
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, x0_recon = self.p_sample_backward(x_t=imgs[-1],
                                                       y=y, 
                                                       context=context, 
                                                       i=i, 
                                                       clip_denoised=clip_denoised, 
                                                       class_cond=class_cond)
                imgs.append(img)
                one_step_imgs.append(x0_recon)
            return imgs, one_step_imgs
        else:
            img = y
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, _ = self.p_sample_backward(x_t=img, 
                                                y=y, 
                                                context=context, 
                                                i=i, 
                                                clip_denoised=clip_denoised,
                                                class_cond=class_cond)
            return img
        
    @torch.no_grad()
    def sample_backward(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        if self.condition_key == 'dual':
            x0_null = torch.zeros_like(y)
            context = torch.cat((x0_null, y), dim=1)
        return self.p_sample_loop_backward(y, context, clip_denoised, sample_mid_step)
    
    @torch.no_grad()
    def p_sample_forward(self, x_t, x, context, i, clip_denoised=False, class_cond=None):
        b, *_, device = *x_t.shape, x_t.device
        if self.asc_steps[i] == (self.num_timesteps - 1):
            t = torch.full((x_t.shape[0],), self.asc_steps[i], device=x_t.device, dtype=torch.long)
            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context, y=class_cond)
            xT_recon = self.predict_xT_from_objective(x_t, x, t, objective_recon=objective_recon)
            if clip_denoised:
                xT_recon.clamp_(-1., 1.)
            return xT_recon, xT_recon
        else:

            t = torch.full((x_t.shape[0],), self.asc_steps[i], device=x_t.device, dtype=torch.long)        #t
            n_t = torch.full((x_t.shape[0],), self.asc_steps[i+1], device=x_t.device, dtype=torch.long)    #t+1

            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context, y=class_cond)             #epsilon_t       
            xT_recon = self.predict_xT_from_objective(x_t, x, t, objective_recon=objective_recon)
            if clip_denoised:
                xT_recon.clamp_(-1., 1.)
            
            m_t = extract(self.m_t, t, x_t.shape)                  #m_{t}
            m_nt = extract(self.m_t, n_t, x_t.shape)               #m_{t+1}
            var_t = extract(self.variance_t, t, x_t.shape)         #sigma_t
            var_nt = extract(self.variance_t, n_t, x_t.shape)      #sigma_{t+1}
            

            delta2_t = self.eta * ((var_nt - var_t * ((1. - m_nt)/ (1. - m_t)) ** 2))
            sigma_t = torch.sqrt(delta2_t)
            
            noise = torch.randn_like(x_t)
            
            x_tplus_mean = (1. - m_nt) * x + m_nt * xT_recon + torch.sqrt((var_nt - delta2_t)/var_t) * \
                            (x_t - (1. - m_t) * x - m_t * xT_recon)

            return x_tplus_mean + sigma_t * noise, xT_recon
        
    @torch.no_grad()
    def p_sample_loop_forward(self, x, context=None, clip_denoised=True, sample_mid_step=False, class_cond=None):
        if self.condition_key == "nocond":
            context = None
        else:
            context = x if context is None else context

        if sample_mid_step:
            imgs, one_step_imgs = [x], []
            
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, x0_recon = self.p_sample_forward(x_t=imgs[-1], 
                                                      x=x, 
                                                      context=context, 
                                                      i=i, 
                                                      clip_denoised=clip_denoised,
                                                      class_cond=class_cond)
                imgs.append(img)
                one_step_imgs.append(x0_recon)
            return imgs, one_step_imgs
        else:
            img = x
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, _ = self.p_sample_forward(x_t=img, 
                                               x=x, 
                                               context=context, 
                                               i=i, 
                                               clip_denoised=clip_denoised,
                                               class_cond=class_cond)
            return img
    
    @torch.no_grad()
    def sample_forward(self, x0, context=None, clip_denoised=True, sample_mid_step=False):
        if self.condition_key == 'dual':
            y_null = torch.zeros_like(x0)
            context = torch.cat((x0, y_null), dim=1)
        return self.p_sample_loop_forward(x0, context, clip_denoised, sample_mid_step)