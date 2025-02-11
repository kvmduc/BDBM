import os

import torch.optim.lr_scheduler
from torch.utils.data import DataLoader

from PIL import Image
from Register import Registers
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
from model.BrownianBridge.BidirectionalBridgeModel import BidirectionalBridgeModel
from model.BrownianBridge.BidirectionalBridgeModel_cont import BidirectionalBridgeModel_continuous
from model.BrownianBridge.LatentBidirectionalBridgeModel import LatentBidirectionalBridgeModel
from runners.DiffusionBasedModelRunners.DiffusionBaseRunner import DiffusionBaseRunner
from runners.utils import weights_init, get_optimizer, get_dataset, make_dir, get_image_grid, save_single_image
from tqdm.autonotebook import tqdm
from torchsummary import summary
import wandb
import time


@Registers.runners.register_with_name('BBDMRunner')
class BBDMRunner(DiffusionBaseRunner):
    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, config):
        if config.model.model_type == "BBDM":
            bbdmnet = BrownianBridgeModel(config.model).to(config.training.device[0])
        elif config.model.model_type == "LBBDM":
            bbdmnet = LatentBrownianBridgeModel(config.model).to(config.training.device[0])
        elif config.model.model_type == "BDBM":
            bbdmnet = BidirectionalBridgeModel(config.model).to(config.training.device[0])
        elif config.model.model_type == "BDBM_cont":
            bbdmnet = BidirectionalBridgeModel_continuous(config.model).to(config.training.device[0])
        elif config.model.model_type == "LBDBM":
            bbdmnet = LatentBidirectionalBridgeModel(config.model).to(config.training.device[0])
        else:
            raise NotImplementedError
        bbdmnet.apply(weights_init)
        return bbdmnet

    def load_model_from_checkpoint(self):
        states = None
        if self.config.model.only_load_latent_mean_std:
            if self.config.model.__contains__('model_load_path') and self.config.model.model_load_path is not None:
                states = torch.load(self.config.model.model_load_path, map_location='cpu')
        else:
            states = super().load_model_from_checkpoint()

        if self.config.model.normalize_latent:
            if states is not None:
                self.net.ori_latent_mean = states['ori_latent_mean'].to(self.config.training.device[0])
                self.net.ori_latent_std = states['ori_latent_std'].to(self.config.training.device[0])
                self.net.cond_latent_mean = states['cond_latent_mean'].to(self.config.training.device[0])
                self.net.cond_latent_std = states['cond_latent_std'].to(self.config.training.device[0])
            else:
                if self.config.args.train:
                    self.get_latent_mean_std()

    def print_model_summary(self, net):
        def get_parameter_number(model):
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_num, trainable_num

        total_num, trainable_num = get_parameter_number(net)
        print("Total Number of parameter: %.2fM" % (total_num / 1e6))
        print("Trainable Number of parameter: %.2fM" % (trainable_num / 1e6))

    def initialize_optimizer_scheduler(self, net, config):
        optimizer = get_optimizer(config.model.BB.optimizer, net.get_parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               mode='min',
                                                               verbose=True,
                                                               threshold_mode='rel',
                                                               **vars(config.model.BB.lr_scheduler)
)
        return [optimizer], [scheduler]

    @torch.no_grad()
    def get_checkpoint_states(self, stage='epoch_end'):
        model_states, optimizer_scheduler_states = super().get_checkpoint_states()
        if self.config.model.normalize_latent:
            if self.config.training.use_DDP:
                model_states['ori_latent_mean'] = self.net.module.ori_latent_mean
                model_states['ori_latent_std'] = self.net.module.ori_latent_std
                model_states['cond_latent_mean'] = self.net.module.cond_latent_mean
                model_states['cond_latent_std'] = self.net.module.cond_latent_std
            else:
                model_states['ori_latent_mean'] = self.net.ori_latent_mean
                model_states['ori_latent_std'] = self.net.ori_latent_std
                model_states['cond_latent_mean'] = self.net.cond_latent_mean
                model_states['cond_latent_std'] = self.net.cond_latent_std
        return model_states, optimizer_scheduler_states

    def get_latent_mean_std(self):
        train_dataset, val_dataset, test_dataset = get_dataset(self.config.data)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config.data.train.batch_size,
                                  shuffle=True,
                                  num_workers=2,
                                  drop_last=True)

        total_ori_mean = None
        total_ori_var = None
        total_cond_mean = None
        total_cond_var = None
        max_batch_num = 30000 // self.config.data.train.batch_size

        def calc_mean(batch, total_ori_mean=None, total_cond_mean=None):
            (x, x_name), (x_cond, x_cond_name) = batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)                  #[KxCxHxW]
            x_cond_latent = self.net.encode(x_cond, cond=True, normalize=False)         
            # x_mean = x_latent.mean(axis=[0, 2, 3], keepdim=True)                      #[1xCx1x1]
            x_mean = x_latent.mean(axis=[0], keepdim=True)                              #[1xCxHxW]
            total_ori_mean = x_mean if total_ori_mean is None else x_mean + total_ori_mean

            # x_cond_mean = x_cond_latent.mean(axis=[0, 2, 3], keepdim=True)              #[1xCx1x1]
            x_cond_mean = x_cond_latent.mean(axis=[0], keepdim=True)                      #[1xCxHxW]
            total_cond_mean = x_cond_mean if total_cond_mean is None else x_cond_mean + total_cond_mean
            return total_ori_mean, total_cond_mean

        def calc_var(batch, ori_latent_mean=None, cond_latent_mean=None, total_ori_var=None, total_cond_var=None):
            (x, x_name), (x_cond, x_cond_name) = batch
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_latent = self.net.encode(x_cond, cond=True, normalize=False)
            x_var = ((x_latent - ori_latent_mean) ** 2).mean(axis=[0], keepdim=True)
            total_ori_var = x_var if total_ori_var is None else x_var + total_ori_var

            x_cond_var = ((x_cond_latent - cond_latent_mean) ** 2).mean(axis=[0], keepdim=True)
            total_cond_var = x_cond_var if total_cond_var is None else x_cond_var + total_cond_var
            return total_ori_var, total_cond_var

        print(f"start calculating latent mean")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            batch_count += 1
            total_ori_mean, total_cond_mean = calc_mean(train_batch, total_ori_mean, total_cond_mean)

        ori_latent_mean = total_ori_mean / batch_count
        self.net.ori_latent_mean = ori_latent_mean

        cond_latent_mean = total_cond_mean / batch_count
        self.net.cond_latent_mean = cond_latent_mean

        print(f"start calculating latent std")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            batch_count += 1
            total_ori_var, total_cond_var = calc_var(train_batch,
                                                     ori_latent_mean=ori_latent_mean,
                                                     cond_latent_mean=cond_latent_mean,
                                                     total_ori_var=total_ori_var,
                                                     total_cond_var=total_cond_var)
            # break

        ori_latent_var = total_ori_var / batch_count
        cond_latent_var = total_cond_var / batch_count

        self.net.ori_latent_std = torch.sqrt(ori_latent_var)
        self.net.cond_latent_std = torch.sqrt(cond_latent_var)
        print(self.net.ori_latent_mean)
        print(self.net.ori_latent_std)
        print(self.net.cond_latent_mean)
        print(self.net.cond_latent_std)

    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        (x, x_name), (x_cond, x_cond_name) = batch
        x = x.to(self.config.training.device[0])
        x_cond = x_cond.to(self.config.training.device[0])

        loss, additional_info = net(x, x_cond)
        if write:
            self.writer.add_scalar(f'loss/{stage}', loss, step)
            if additional_info.__contains__('recloss_noise'):
                self.writer.add_scalar(f'recloss_noise/{stage}', additional_info['recloss_noise'], step)
            if additional_info.__contains__('recloss_xy'):
                self.writer.add_scalar(f'recloss_xy/{stage}', additional_info['recloss_xy'], step)
        return loss

    @torch.no_grad()
    def sample(self, net, batch, sample_path, stage='train'):
        sample_path = make_dir(os.path.join(sample_path, f'{stage}_sample'))
        reverse_sample_path = make_dir(os.path.join(sample_path, 'reverse_sample'))
        forward_sample_path = make_dir(os.path.join(sample_path, 'forward_sample'))
        reverse_one_step_path = make_dir(os.path.join(sample_path, 'reverse_one_step_samples'))

        (x0, x0_name), (xT, xT_name) = batch

        batch_size = x0.shape[0] if x0.shape[0] < 4 else 4

        x0 = x0[:batch_size].to(self.config.training.device[0])
        xT = xT[:batch_size].to(self.config.training.device[0])

        grid_size = 4
        
        sample_x0 = net.sample_backward(xT, clip_denoised=self.config.testing.clip_denoised)         # last-step samples
        sample_xT = net.sample_forward(x0, clip_denoised=self.config.testing.clip_denoised)         # last-step samples
        
        # midstep_x0, sample_x0 = net.sample_backward(xT, clip_denoised=self.config.testing.clip_denoised, sample_mid_step=True)         # mid-step samples
        # midstep_xT, sample_xT = net.sample_forward(x0, clip_denoised=self.config.testing.clip_denoised, sample_mid_step=True)         # mid-step samples
        # sample_x0 = sample_x0[0]
        # sample_xT = sample_xT[0]
        # self.save_images(midstep_x0, reverse_sample_path, grid_size, gif_interval=2, save_interval=20,
        #                  writer_tag=f'{stage}_sample' if stage != 'test' else None)
        # self.save_images(midstep_xT, forward_sample_path, grid_size, gif_interval=2, save_interval=20,
        #                  writer_tag=f'{stage}_sample' if stage != 'test' else None)
        
        image_grid = get_image_grid(sample_x0.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'skip_sample_x0.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_skip_sample_x0', image_grid, self.global_step, dataformats='HWC')
            
        image_grid = get_image_grid(sample_xT.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'skip_sample_xT.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_skip_sample_xT', image_grid, self.global_step, dataformats='HWC')

        image_grid = get_image_grid(xT.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'condition.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_condition', image_grid, self.global_step, dataformats='HWC')

        image_grid = get_image_grid(x0.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        im = Image.fromarray(image_grid)
        im.save(os.path.join(sample_path, 'ground_truth.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_ground_truth', image_grid, self.global_step, dataformats='HWC')
        
        try:
            if wandb.run is not None:
                inp = torch.vstack((xT, x0))
                res1 = torch.vstack((xT, sample_x0))
                res2 = torch.vstack((x0, sample_xT))
                res_img = torch.vstack((inp, res1, res2))
                res_grid = get_image_grid(res_img.to('cpu'), grid_size*2, to_normal=self.config.data.dataset_config.to_normal)
                res_grid = wandb.Image(res_grid, caption=f"Line1:GT; Line2:Sample Backward; Line3:Sample Forward")
                wandb.log({f"{stage}_step:{self.global_step}": res_grid})
                del inp, res1, res2, res_img, res_grid
        except:
            pass

    @torch.no_grad()
    def sample_to_eval(self, net, test_loader, sample_path):
        xT_path = make_dir(os.path.join(sample_path, 'xT'))
        x0_path = make_dir(os.path.join(sample_path, 'x0'))
        result_path = make_dir(os.path.join(sample_path, f"{self.config.model.BB.params.sample_step}_eta={self.config.model.BB.params.eta}_var={self.config.model.BB.params.max_var}"))
        x0_pred_path = make_dir(os.path.join(result_path, 'x0_pred'))
        xT_pred_path = make_dir(os.path.join(result_path, 'xT_pred'))

        print(f'Result store at: {result_path}')
        print(f'Number of images: {len(test_loader.dataset)}')

        pbar = tqdm(test_loader, total=len(test_loader), smoothing=0.01)
        
        to_normal = self.config.data.dataset_config.to_normal
        sample_num = self.config.testing.sample_num
        
        for k, test_batch in enumerate(pbar):
            (x0, x0_name), (xT, xT_name) = test_batch
            x0 = x0.to(self.config.training.device[0])
            xT = xT.to(self.config.training.device[0])

            batch_size = x0.shape[0]
            
            for j in range(sample_num):
                
                sample_x0 = net.sample_backward(xT, clip_denoised=False)       
                sample_xT = net.sample_forward(x0, clip_denoised=False)        
                
                # sample = net.sample_vqgan(x)
                for i in range(batch_size):
                    gt_xT = xT[i].detach().clone()
                    gt_x0 = x0[i]
                    pred_x0 = sample_x0[i]
                    pred_xT = sample_xT[i]
                    if j == 0:
                        save_single_image(gt_x0, x0_path, f'{x0_name[i]}.png', to_normal=to_normal)
                        save_single_image(gt_xT, xT_path, f'{xT_name[i]}.png', to_normal=to_normal)
                    if sample_num > 1:
                        x0_pred_path_i = make_dir(os.path.join(x0_pred_path, x0_name[i]))
                        xT_pred_path_i = make_dir(os.path.join(xT_pred_path, xT_name[i]))
                        save_single_image(pred_x0, x0_pred_path_i, f'output_{j}.png', to_normal=to_normal)
                        save_single_image(pred_xT, xT_pred_path_i, f'output_{j}.png', to_normal=to_normal)
                    else:
                        save_single_image(pred_x0, x0_pred_path, f'{x0_name[i]}.png', to_normal=to_normal)
                        save_single_image(pred_xT, xT_pred_path, f'{xT_name[i]}.png', to_normal=to_normal)