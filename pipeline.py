
import os
import gc
import sys
import numpy as np
import math
import random
import time
import PIL
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from collections import OrderedDict
from einops import rearrange
from diffusers.utils.torch_utils import randn_tensor

from model.scheduler import PyramidFlowMatchEulerDiscreteScheduler
from model.vae import CausalVideoVAE
from model.mmdit import MMDiT, SD3TextEncoderWithMask


def get_raymap_from_camera_parameters_batchversion(trans2d_matrix, trans3d_matrix, depth_shape, vae_downsample=1):
    raymap_list = []
    for _trans3d_matrix, _trans2d_matrix in zip(trans3d_matrix, trans2d_matrix):
        raymap = get_raymap_from_camera_parameters(
            _trans2d_matrix,
            _trans3d_matrix,
            depth_shape,
            vae_downsample,
        )
        raymap_list.append(raymap)
    return torch.stack(raymap_list, dim=0)

def get_raymap_from_camera_parameters(trans2d_matrix, trans3d_matrix, depth_shape, vae_downsample=1):

    H, W = depth_shape

    def get_raymap_from_trans2d(trans2d_matrix, H, W):
        fu = trans2d_matrix[:, 0, 0].unsqueeze(-1).unsqueeze(-1)
        fv = trans2d_matrix[:, 1, 1].unsqueeze(-1).unsqueeze(-1)
        cu = trans2d_matrix[:, 0, 2].unsqueeze(-1).unsqueeze(-1)
        cv = trans2d_matrix[:, 1, 2].unsqueeze(-1).unsqueeze(-1)

        u, v = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
        u    = u.unsqueeze(0).repeat(trans2d_matrix.shape[0], 1, 1).to(trans2d_matrix.device)
        v    = v.unsqueeze(0).repeat(trans2d_matrix.shape[0], 1, 1).to(trans2d_matrix.device)
        z_cam = torch.ones_like(u).to(trans2d_matrix.device)
        x_cam = (u - cu) / fu
        y_cam = (v - cv) / fv
        addition_dim = torch.ones_like(u).to(trans2d_matrix.device)

        return torch.stack((x_cam, y_cam, z_cam, addition_dim), dim=-1)
    
    ray_d = get_raymap_from_trans2d(trans2d_matrix, H, W).to(trans3d_matrix)
    ray_d = rearrange(ray_d,   't h w c -> t c h w')
    
    _trans3d_matrix = trans3d_matrix.clone()
    _trans3d_matrix[:, :3, 3] = 0.
    ray_d       = F.avg_pool2d(ray_d, kernel_size=vae_downsample, stride=vae_downsample)
    T, _, ray_d_h, ray_d_w = ray_d.shape
    ray_d       = rearrange(ray_d,   't c h w -> t c (h w)')
    ray_d_world = torch.bmm(_trans3d_matrix, ray_d)
    ray_d_world = rearrange(ray_d_world, 't c (h w) -> t c h w', h=ray_d_h, w=ray_d_w)
    ray_d_world = ray_d_world[:, :3]
    ray_d_world = ray_d_world / ray_d_world.norm(dim=1, keepdim=True) # TODO : check it
    ray_o_world = torch.ones_like(ray_d_world).to(ray_d_world.device) * trans3d_matrix[:, :3, 3].unsqueeze(-1).unsqueeze(-1)
    ray_world   = torch.cat([ray_d_world, ray_o_world], dim=1) 
    return ray_world

def raymap_to_trans_matrix(
    raymap, # [b, c=6, t, h, w]
    trans3d_scale_factor=1.0,
    append_first_reference=False,
    from_relative_to_absolute=False,
    vae_downsample=8,
):
    b, _, t, h, w = raymap.shape

    ref_ray       = raymap[:, :3].mean(dim=[-1, -2]).unsqueeze(-1).unsqueeze(-1)
    ref_ray       = ref_ray / ref_ray.norm(dim=1, keepdim=True)
    projection    = (raymap[:, :3] * ref_ray).sum(dim=1, keepdim=True)
    raymap[:, :3] = raymap[:, :3] / projection
    
    ray_o = rearrange(raymap[:, 3: ], "b c t h w -> b t h w c") / trans3d_scale_factor
    ray_d = rearrange(raymap[:,  :3], "b c t h w -> b t h w c")  # [T, H, W, C]
    ray_o = torch.sign(ray_o) * (ray_o.abs() ** 2)

    # Compute location and directions
    location       = ray_o.reshape(b, t, -1, 3).mean(dim=-2) # [b, t, c]
    image_location = (ray_o + ray_d).reshape(b, t, -1, 3).mean(dim=-2) # [b, t, c]
    Focal          = torch.norm(image_location - location, dim=-1) # [b, t]
    Z_Dir          = image_location - location  # [b, t, c]

    # Compute the width (W) and field of view (FoV_x)
    W_Left  = ray_d[:, :, :,   :1, :].reshape(b, t, -1, 3).mean(dim=-2) # [b, t, c]
    W_Right = ray_d[:, :, :, -1: , :].reshape(b, t, -1, 3).mean(dim=-2) # [b, t, c]
    W       = W_Right - W_Left
    W_real  = (
        torch.norm(torch.cross(W, Z_Dir), dim=-1)
        / (raymap.shape[-1] - 1)
        * raymap.shape[-1]
    ) # [b, t]
    Fov_x   = torch.arctan(W_real / (2 * Focal))

    # Compute the height (H) and field of view (FoV_y)
    H_Up    = ray_d[:, :,   :1, :, :].reshape(b, t, -1, 3).mean(dim=-2)
    H_Down  = ray_d[:, :, -1: , :, :].reshape(b, t, -1, 3).mean(dim=-2)
    H       = H_Up - H_Down
    H_real  = (
        torch.norm(torch.cross(H, Z_Dir), dim=-1)
        / (raymap.shape[-2] - 1)
        * raymap.shape[-2]
    )
    Fov_y   = torch.arctan(H_real / (2 * Focal))

    # Compute X, Y, and Z directions for the camera
    X_Dir = W_Right - W_Left
    Y_Dir = torch.cross(Z_Dir, X_Dir) # [b, t, c]
    X_Dir = torch.cross(Y_Dir, Z_Dir)  # [b, t, c]

    X_Dir /= torch.norm(X_Dir, dim=-1, keepdim=True)
    Y_Dir /= torch.norm(Y_Dir, dim=-1, keepdim=True)
    Z_Dir /= torch.norm(Z_Dir, dim=-1, keepdim=True)

    # Create the camera-to-world (camera_pose) transformation matrix
    camera_pose              = torch.zeros((b, t, 4, 4))
    camera_pose[:, :, :3, 0] = X_Dir
    camera_pose[:, :, :3, 1] = Y_Dir
    camera_pose[:, :, :3, 2] = Z_Dir
    camera_pose[:, :, :3, 3] = location
    camera_pose[:, :,  3, 3] = 1.0

    # Create the intrinsic matrix
    intri_rescale_factor  = (w / W_real + h / H_real) / 2 * vae_downsample
    intrinsic             = torch.zeros((b, t, 4, 4))
    intrinsic[:, :, 0, 0] = Focal      * intri_rescale_factor
    intrinsic[:, :, 1, 1] = Focal      * intri_rescale_factor
    intrinsic[:, :, 0, 2] = w / 2 * vae_downsample
    intrinsic[:, :, 1, 2] = h / 2 * vae_downsample
    intrinsic[:, :, 2, 2] = 1.0
    intrinsic[:, :, 3, 3] = 1.0

    if append_first_reference:
        camera_pose_ref = torch.eye(4, 4).unsqueeze(0).unsqueeze(0).repeat(b, 1, 1, 1).to(camera_pose)
        camera_pose     = torch.cat([camera_pose_ref, camera_pose], dim=1)
        intrinsic_ref   = intrinsic[:, :1]
        intrinsic       = torch.cat([intrinsic_ref, intrinsic], dim=1) # t dim concat
    
    camera_pose = camera_pose.to(raymap)
    intrinsic   = intrinsic.to(raymap)

    if from_relative_to_absolute:
        for i in range(t):
            camera_pose[:, i+1] = torch.bmm(camera_pose[:, i], camera_pose[:, i+1])

    return camera_pose, intrinsic


class InferencePipeline:
    """
    A consolidated pipeline for performing inference with the DeepVerse model.
    This class handles model loading, architecture customization, and the video generation process.
    """

    def __init__(
        self, 
        model_cfg: Dict, 
        device: str = "cuda", 
        torch_dtype: torch.dtype = torch.bfloat16
    ):

        super().__init__()
        """
        Initializes the pipeline.
        """
        self.device     = torch.device(device)
        self.dtype      = torch_dtype
        self.model_cfg  = model_cfg
        self.downsample = 8

        self.model, self.vae, self.scheduler, self.text_encoder = self._create_models()
        
        self.model.eval().to(self.device, dtype=self.dtype)
        self.vae.eval().to(self.device, dtype=self.dtype)
        self.text_encoder.eval().to(self.device, dtype=self.dtype)

        self.vae_shift_factor       = 0.1490
        self.vae_scale_factor       = 1 / 1.8415
        self.vae_video_shift_factor = -0.2343
        self.vae_video_scale_factor = 1 / 3.0986

        self.text_embeds = torch.load(self.model_cfg['text_embeds_path'], map_location='cpu')
        self.raymap_mean = torch.tensor([-0.0016, -0.0010, 0.9015, 0.0313, -0.0538, 0.2079]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.raymap_std  = torch.tensor([ 0.3333,  0.2567, 0.0927, 0.4338,  0.1746, 0.5802]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    def _create_models(self):
        """Instantiates the DiT, VAE, and Scheduler models."""
        dit_config = self.model_cfg['dit_config']
        dit = MMDiT.from_pretrained(
            dit_config['model_path'], 
            torch_dtype=self.dtype, 
            use_mixed_training=False,
            # use_flash_attn=True,
        )

        vae_config = self.model_cfg['vae_config']
        vae        = CausalVideoVAE.from_pretrained(**vae_config)
        vae.enable_tiling()

        scheduler_config = self.model_cfg['scheduler_config']
        scheduler        = PyramidFlowMatchEulerDiscreteScheduler(**scheduler_config)

        text_encoder_config = self.model_cfg['text_encoder_config']
        text_encoder        = SD3TextEncoderWithMask(**text_encoder_config)

        return dit, vae, scheduler, text_encoder

    @torch.no_grad()
    def get_pyramid_latent(self, x, stage_num):
        vae_latent_list = []
        vae_latent_list.append(x)

        temp, height, width = x.shape[-3], x.shape[-2], x.shape[-1]
        for _ in range(stage_num):
            height //= 2
            width  //= 2
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = torch.nn.functional.interpolate(x, size=(height, width), mode='bilinear')
            x = rearrange(x, '(b t) c h w -> b c t h w', t=temp)
            vae_latent_list.append(x)

        vae_latent_list = list(reversed(vae_latent_list))
        return vae_latent_list

    @torch.no_grad()
    def get_history_vae_latent(
        self, 
        history_rgb,        # b c t h w
        history_disparity,  # b c t h w
        history_raymap,     # b c t h w
    ):
        cur_device = history_rgb.device
        video      = self.vae.encode(      history_rgb.to(self.vae.device, dtype=self.vae.dtype)).latent_dist.sample().to(cur_device) 
        disparity  = self.vae.encode(history_disparity.to(self.vae.device, dtype=self.vae.dtype)).latent_dist.sample().to(cur_device) 

        video[:, :,  :1]     = (video[:, :,  :1] - self.vae_shift_factor)     * self.vae_scale_factor
        disparity[:, :,  :1] = (disparity[:, :,  :1] - self.vae_shift_factor) * self.vae_scale_factor
        
        self.raymap_mean = self.raymap_mean.to(self.device)
        self.raymap_std  = self.raymap_std.to( self.device)
        history_raymap[:, :3] = history_raymap[:, :3] / history_raymap[:, :3].norm(dim=1, keepdim=True) 
        history_raymap        = (history_raymap - self.raymap_mean) / self.raymap_std
       
        video = torch.cat([video, disparity, history_raymap], dim=1)
        return video
    
    def generate(self, batch_dict: Dict) -> Dict:
    
        actual_frame = (self.model_cfg['max_temporal_length'] - 1) * self.model_cfg['vae_downsample'] + 1
        actual_unit  = self.model_cfg['max_temporal_length']

        num_input_image = 25
        num_input_unit  = 4

        start_frame = 0
        start_unit  = 0

        motion_prompt_total = batch_dict["prompt"]
        while ((len(motion_prompt_total) - actual_unit) % (actual_unit - num_input_unit) != 0) or (len(motion_prompt_total) < self.model_cfg['max_temporal_length']):
            motion_prompt_total = np.concatenate([motion_prompt_total, motion_prompt_total[-1:]])

        total_iters = math.floor((len(motion_prompt_total) - actual_unit) / (actual_unit - num_input_unit) + 1)
        
        images_list, disparitys_list, trans3d_list, trans2d_list = [], [], [], []
        motion_prompt_list = []

        input_image, input_disparity, input_raymap, input_history = None, None, None, None
        scale_factor = 1.0

        ###########################################################################
        #                                inference                                #
        ###########################################################################
        for now_iter in range(total_iters):
            
            if input_image is None:
                input_image = [batch_dict["img"]]

            motion_prompt = np.concatenate([motion_prompt_total[0:1], motion_prompt_total[start_unit + 1 : start_unit + actual_unit]])

            self.raymap_mean = self.raymap_mean.to(self.device)
            self.raymap_std  = self.raymap_std.to( self.device)
            if input_raymap is not None:
                input_raymap = (input_raymap - self.raymap_mean) / self.raymap_std

            images, disparitys, trans3d, trans2d = self.generate_i2v(
                motion_prompt,
                (batch_dict['prompt_type'] == 'action'), 
                input_image, input_disparity, input_raymap, input_history,
                temp = self.model_cfg['max_temporal_length'],
                num_inference_steps = self.model_cfg.get('num_inference_steps', 10),
                guidance_scale = 4.0, video_guidance_scale = 3.5,
                use_linear_guidance = False, alpha = 1.0, min_guidance_scale = 1.1,
            )
            disparitys           = disparitys.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1, 1) * 0.5 + 0.5
            disparitys           = torch.clamp(disparitys, 0, 1) ** 2
            disparitys           = disparitys / scale_factor / 0.95
            trans3d[:, :, :3, 3] = trans3d[:, :, :3, 3] * scale_factor

            start_frame = start_frame + actual_frame - num_input_image
            start_unit  = start_unit  + actual_unit  - num_input_unit

            if now_iter == 0:
                images_list.append(        images)
                disparitys_list.append(disparitys)
                motion_prompt_list.append(motion_prompt)
                trans3d_list.append(trans3d)
                trans2d_list.append(trans2d)

            else:
                images_list.append(        images[:, :, num_input_image:])
                disparitys_list.append(disparitys[:, :, num_input_image:])
                motion_prompt_list.append(motion_prompt[num_input_unit:])
                trans3d_pre                = trans3d_list[-1][:, -num_input_unit]
                for idx_trans3d in range(trans3d.shape[1]):
                    trans3d[:, idx_trans3d] = torch.bmm(trans3d_pre, trans3d[:, idx_trans3d])

                trans3d_list.append(trans3d[:, num_input_unit:]) # [b t 4 4]
                trans2d_list.append(trans2d[:, num_input_unit:])

            #### for the next iter
            input_images = rearrange(images, 'b c t h w -> b t h w c')[0, -num_input_image:]
            input_image  = []
            for img_idx in range(num_input_image):
                img = (torch.clamp(input_images[img_idx] * 0.5 + 0.5, 0, 1).to(torch.float32).cpu().numpy() * 255).astype(np.uint8)
                img = Image.fromarray(img)
                input_image.append(img)

            input_disparity = disparitys[:, :, -num_input_image:] # [0, 1]
            if not self.model_cfg.get('no_need_depth', False):
                scale_factor    = 1 / input_disparity[:, :, 0].max()
                input_disparity = input_disparity * scale_factor * 0.95
                input_disparity = torch.sqrt(input_disparity)
                input_disparity = input_disparity * 2 - 1

            cur_trans3d     = torch.cat(trans3d_list, dim=1)[:, -num_input_unit:] # [b t 4 4]
            ref_trans3d     = cur_trans3d[:, 0]
            ref_trans3d_inv = torch.inverse(ref_trans3d) # [b 1 4 4]
            for idx_trans3d in range(cur_trans3d.shape[1]):
                cur_trans3d[:, idx_trans3d] = torch.bmm(ref_trans3d_inv, cur_trans3d[:, idx_trans3d])
            for idx_trans3d in range(cur_trans3d.shape[1]-1, 0, -1):
                cur_trans3d[:, idx_trans3d] = torch.bmm(torch.inverse(cur_trans3d[:, idx_trans3d-1]), cur_trans3d[:, idx_trans3d])

            cur_trans3d[:, :, :3, 3] /= scale_factor
            cur_trans3d[:, :, :3, 3]  = torch.sign(cur_trans3d[:, :, :3, 3]) * torch.sqrt(cur_trans3d[:, :, :3, 3].abs())
            input_raymap    = get_raymap_from_camera_parameters_batchversion( 
                trans2d_list[-1][:, -num_input_unit:],
                cur_trans3d.to(input_disparity),
                input_disparity.shape[-2:],
                vae_downsample=8,
            )
            input_raymap    = rearrange(input_raymap, 'b t c h w -> b c t h w')
            
            cur_images     = torch.cat(images_list, dim=2)[    :, :, ::self.model_cfg['vae_downsample']]
            cur_disparitys = torch.cat(disparitys_list, dim=2)[:, :, ::self.model_cfg['vae_downsample']]
            cur_trans3d    = torch.cat(trans3d_list, dim=1)
            cur_trans2d    = torch.cat(trans2d_list, dim=1)
            ref_trans3d    = cur_trans3d[:, -num_input_unit]
            ref_trans3d_inv = torch.inverse(ref_trans3d) # [b 1 4 4]
            for idx_trans3d in range(0, cur_trans3d.shape[1]):
                cur_trans3d[:, idx_trans3d] = torch.bmm(ref_trans3d_inv, cur_trans3d[:, idx_trans3d])

            c2w = cur_trans3d.squeeze(0)
            t   = c2w.shape[0]

            last_cam_pos     = c2w[-1, :3, 3] 
            last_cam_forward = c2w[-1, :3, 2]
            cam_positions    = c2w[ :, :3, 3]
            distances        = torch.norm(cam_positions[:-1] - last_cam_pos, dim=1)  
            _, closest_pos_indices = torch.topk(-distances, k=5)
            closest_cam_forwards   = c2w[closest_pos_indices, :3, 2] 

            dots          = torch.sum(closest_cam_forwards * last_cam_forward, dim=1) 
            angles        = torch.acos(torch.clamp(dots, -1.0, 1.0)) 
            min_angle_idx = torch.argmin(angles)

            sample_history_idx = closest_pos_indices[min_angle_idx].item()
            cur_image     = cur_images[    :, :, sample_history_idx:sample_history_idx+1]
            cur_disparity = cur_disparitys[:, :, sample_history_idx:sample_history_idx+1]
            cur_trans3d   = cur_trans3d[   :,    sample_history_idx:sample_history_idx+1]
            cur_trans2d   = cur_trans2d[   :,    sample_history_idx:sample_history_idx+1]

            cur_disparity = cur_disparity * scale_factor * 0.95
            cur_disparity = torch.sqrt(cur_disparity)
            cur_disparity = torch.clamp(cur_disparity * 2 - 1, -1, 1) 

            cur_trans3d[:, :, :3, 3] /= scale_factor
            cur_trans3d[:, :, :3, 3]  = torch.sign(cur_trans3d[:, :, :3, 3]) * torch.sqrt(cur_trans3d[:, :, :3, 3].abs())

            cur_raymap    = get_raymap_from_camera_parameters_batchversion(
                cur_trans2d, cur_trans3d, cur_disparity.shape[-2:],
                vae_downsample=self.model_cfg['vae_downsample'],
            )
            cur_raymap    = rearrange(cur_raymap,    'b t c h w -> b c t h w')
            input_history = self.get_history_vae_latent( cur_image, cur_disparity, cur_raymap)

        images     = torch.cat(images_list, dim=2)
        disparitys = torch.cat(disparitys_list, dim=2)
        trans3d    = torch.cat(trans3d_list, dim=1)
        trans2d    = torch.cat(trans2d_list, dim=1)

        return {
            'pred_img': images,
            'pred_disparity': disparitys,
            'motion_prompt_list': motion_prompt_list,
            'trans3d': trans3d,
            'trans2d': trans2d,
        }

    def prepare_latents(self, batch_size, num_channels_latents, temp, height, width, dtype, device, generator):
        shape   = (batch_size, num_channels_latents, int(temp), int(height) // self.downsample, int(width) // self.downsample)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def sample_block_noise(self, bs, ch, temp, height, width):
        gamma = self.scheduler.config.gamma
        dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(4), torch.eye(4) * (1 + gamma) - torch.ones(4, 4) * gamma)
        block_number = bs * ch * temp * (height // 2) * (width // 2)
        noise = torch.stack([dist.sample() for _ in range(block_number)])
        noise = rearrange(noise, '(b c t h w) (p q) -> b c t (h p) (w q)',b=bs,c=ch,t=temp,h=height//2,w=width//2,p=2,q=2)
        return noise

    @torch.no_grad()
    def generate_one_unit(self, latents, input_history, past_conditions, 
                          prompt_embeds, prompt_attention_mask, pooled_prompt_embeds,
                          num_inference_steps, height, width, temp, device, dtype, generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                          is_first_frame: bool = False):

        stages = self.model_cfg['stages']
        intermed_latents = []
        
        for i_s in range(len(stages)):
            self.scheduler.set_timesteps(num_inference_steps[i_s], i_s, device=device)
            timesteps = self.scheduler.timesteps

            if i_s > 0:
                height *= 2; width *= 2
                latents = rearrange(latents, 'b c t h w -> (b t) c h w')
                latents = F.interpolate(latents, size=(height, width), mode='nearest')
                latents = rearrange(latents, '(b t) c h w -> b c t h w', t=temp)
                ori_sigma = 1 - self.scheduler.ori_start_sigmas[i_s]
                gamma = self.scheduler.config.gamma
                alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)

                bs, ch, temp, height, width = latents.shape
                noise = self.sample_block_noise(bs, ch, temp, height, width)
                noise = noise.to(device=device, dtype=dtype)
                latents = alpha * latents + beta * noise
            
            for idx, t in enumerate(timesteps):
                if input_history == None:
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                else:
                    latent_model_input = torch.cat([latents] * 3)
                    
                timestep = t.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)
                latent_model_input = past_conditions[i_s] + [latent_model_input]

                if self.model_cfg.get('no_need_depth', False):
                    for x in latent_model_input:
                        x[:, 16:] *= 0

                if input_history is not None:
                    history_len      = (input_history.size(-1) / self.model_cfg['history_downsample_ratio'] / 2) * (input_history.size(-2) / self.model_cfg['history_downsample_ratio'] / 2)
                    pos_history_mask = torch.ones( input_history.size(0), int(history_len))
                    neg_history_mask = torch.zeros(input_history.size(0), int(history_len))
                    history_mask     = torch.cat([neg_history_mask, neg_history_mask, pos_history_mask]).to(timestep)

                with torch.cuda.amp.autocast(enabled=True, dtype=self.dtype):
                    noise_pred = self.model(
                        sample=[latent_model_input],
                        timestep_ratio=timestep.to(self.device),
                        encoder_hidden_states=prompt_embeds.to(self.device),
                        encoder_attention_mask=prompt_attention_mask.to(self.device),
                        pooled_projections=pooled_prompt_embeds.to(self.device),

                        history=torch.cat([input_history] * 3)                              if input_history is not None else None,
                        history_downsample_ratio=self.model_cfg['history_downsample_ratio'] if input_history is not None else None,
                        history_mask=history_mask                                           if input_history is not None else None,
                    )

                noise_pred = noise_pred[0]

                # perform guidance
                if input_history == None:
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        if is_first_frame:
                            noise_pred = noise_pred_uncond + self.guidance_scale       * (noise_pred_text - noise_pred_uncond)
                        else:
                            noise_pred = noise_pred_uncond + self.video_guidance_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    noise_pred_uncond, noise_pred_text, noise_pred_text_history = noise_pred.chunk(3)
                    noise_pred = noise_pred_uncond \
                            + self.video_guidance_scale * (noise_pred_text - noise_pred_uncond) \
                            + self.model_cfg['history_guidance_scale'] * (noise_pred_text_history - noise_pred_text)

                latents = self.scheduler.step(
                    model_output=noise_pred,
                    timestep=timestep,
                    sample=latents,
                    generator=generator,
                ).prev_sample

            intermed_latents.append(latents)

        return intermed_latents

    @torch.no_grad()
    def generate_i2v(self, motion_prompt = None, use_motion_prompt = True, input_image: PIL.Image = None, input_disparity = None, input_raymap = None, input_history = None,
                     temp: int = 1, num_inference_steps: Optional[Union[int, List[int]]] = 28, guidance_scale: float = 7.0, video_guidance_scale: float = 4.0, min_guidance_scale: float = 2.0,
                     use_linear_guidance: bool = False, alpha: float = 0.5, num_images_per_prompt: Optional[int] = 1, generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                     output_type: Optional[str] = "pil", save_memory: bool = True, ):
        firstframe_mask  = (input_disparity is None)

        device = self.device 
        dtype  = self.dtype
        
        width  = input_image[0].width
        height = input_image[0].height

        assert temp % self.model_cfg['frame_per_unit'] == 0, "The frames should be divided by frame_per unit"

        if isinstance(num_inference_steps, int):
            num_inference_steps = [num_inference_steps] * len(self.model_cfg['stages'])

        if use_linear_guidance:
            max_guidance_scale = guidance_scale
            guidance_scale_list = [max(max_guidance_scale - alpha * t_, min_guidance_scale) for t_ in range(temp+1)]

        self._guidance_scale       = guidance_scale
        self._video_guidance_scale = video_guidance_scale
        
        latents = self.prepare_latents(num_images_per_prompt, self.model.in_channels, temp + firstframe_mask, height, width, dtype, device, generator)
        temp, height, width = latents.shape[-3], latents.shape[-2], latents.shape[-1]

        latents = rearrange(latents, 'b c t h w -> (b t) c h w')
        for _ in range(len(self.model_cfg['stages'])-1):
            height //= 2;width //= 2
            latents = F.interpolate(latents, size=(height, width), mode='bilinear') * 2

        latents = rearrange(latents, '(b t) c h w -> b c t h w', t=temp)

        num_units = temp // self.model_cfg['frame_per_unit']
        stages    = self.model_cfg['stages']

        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        input_image_tensor = torch.cat([image_transform(input_img).unsqueeze(0).unsqueeze(2) for input_img in input_image], dim=2)
        input_image_latent = self.vae.encode(input_image_tensor.to(self.device, dtype=self.dtype)).latent_dist.sample()
        input_image_latent[:, :,  :1] = (input_image_latent[:, :,  :1] - self.vae_shift_factor)       * self.vae_scale_factor
        input_image_latent[:, :, 1: ] = (input_image_latent[:, :, 1: ] - self.vae_video_shift_factor) * self.vae_video_scale_factor

        if input_disparity is not None:
            input_disparity_latent = self.vae.encode(input_disparity.to(self.device, dtype=self.dtype)).latent_dist.sample()
            input_disparity_latent[:, :,  :1] = (input_disparity_latent[:, :,  :1] - self.vae_shift_factor)       * self.vae_scale_factor
            input_disparity_latent[:, :, 1: ] = (input_disparity_latent[:, :, 1: ] - self.vae_video_shift_factor) * self.vae_video_scale_factor
            
        input_image_latent = torch.cat([
            input_image_latent, 
            torch.zeros_like(input_image_latent)                                       if input_disparity is None else input_disparity_latent,
            torch.zeros_like(input_image_latent[:, :self.model_cfg['raymap_dim'], :1]) if input_raymap    is None else input_raymap,
        ], dim=1).to(self.dtype)

        generated_latents_list = [input_image_latent]  
        last_generated_latents =  input_image_latent

        start_unit_index = 1 if firstframe_mask else (len(input_image) - 1) // 8 + 1
        for unit_index in tqdm(range(start_unit_index, num_units)):
            gc.collect()
            torch.cuda.empty_cache()
            
            if use_linear_guidance:
                self._guidance_scale       = guidance_scale_list[unit_index]
                self._video_guidance_scale = guidance_scale_list[unit_index]

            unit_motion_prompt = motion_prompt[unit_index - firstframe_mask]

            if use_motion_prompt:
                prompt_embeds         = self.text_embeds[unit_motion_prompt]['prompt_embeds']
                pooled_prompt_embeds  = self.text_embeds[unit_motion_prompt]['pooled_prompt_embeds'] 
                prompt_attention_mask = self.text_embeds[unit_motion_prompt]['prompt_attention_mask']
            else:
                prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = self.text_encoder(unit_motion_prompt, self.device)
                
            negative_prompt_embeds         = self.text_embeds['empty']['prompt_embeds'].to(prompt_embeds.device)
            negative_pooled_prompt_embeds  = self.text_embeds['empty']['pooled_prompt_embeds'].to(prompt_embeds.device)
            negative_prompt_attention_mask = self.text_embeds['empty']['prompt_attention_mask'].to(prompt_embeds.device)

            if input_history == None:
                if self.do_classifier_free_guidance:
                    prompt_embeds         = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    pooled_prompt_embeds  = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                    prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
            else:
                prompt_embeds         = torch.cat([negative_prompt_embeds, prompt_embeds, prompt_embeds], dim=0)
                pooled_prompt_embeds  = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask, prompt_attention_mask], dim=0)

            # prepare the condition latents
            past_condition_latents = []
            clean_latents_list = self.get_pyramid_latent(
                torch.cat(generated_latents_list, dim=2), 
                len(stages) - 1
            )

            for i_s in range(len(stages)):
                last_cond_latent = clean_latents_list[i_s][:,:,-self.model_cfg['frame_per_unit']:]

                if input_history == None:
                    stage_input = [torch.cat([last_cond_latent] * 2) if self.do_classifier_free_guidance else last_cond_latent]
                else:
                    stage_input = [torch.cat([last_cond_latent] * 3)]

                # pad the past clean latents
                cur_unit_num = unit_index
                cur_stage    = i_s
                cur_unit_ptx = 1

                while cur_unit_ptx < cur_unit_num - firstframe_mask:
                    cur_stage = max(cur_stage - 1, 0)
                    if cur_stage == 0:
                        break
                    cur_unit_ptx += 1
                    cond_latents = clean_latents_list[cur_stage][:, :, -(cur_unit_ptx * self.model_cfg['frame_per_unit']) : -((cur_unit_ptx - 1) * self.model_cfg['frame_per_unit'])]
                    if input_history == None:
                        stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)
                    else:
                        stage_input.append(torch.cat([cond_latents] * 3))

                if cur_stage == 0 and cur_unit_ptx < cur_unit_num - firstframe_mask:
                    cond_latents = clean_latents_list[0][:, :, firstframe_mask : -(cur_unit_ptx * self.model_cfg['frame_per_unit'])]
                    if input_history == None:
                        stage_input.append(torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents)
                    else:
                        stage_input.append(torch.cat([cond_latents] * 3))

                stage_input = list(reversed(stage_input))
                past_condition_latents.append(stage_input)

            intermed_latents = self.generate_one_unit(
                latents[:, :, unit_index * self.model_cfg['frame_per_unit'] : (unit_index+1) * self.model_cfg['frame_per_unit']],
                input_history,
                past_condition_latents,
                prompt_embeds,
                prompt_attention_mask,
                pooled_prompt_embeds,
                
                num_inference_steps,
                height,
                width,
                self.model_cfg['frame_per_unit'],
                device,
                dtype,
                generator,
                is_first_frame=False,
            )

            generated_latents_list.append(intermed_latents[-1])
            last_generated_latents = intermed_latents
        
        if firstframe_mask:
            generated_latents_list = generated_latents_list[1:]

        generated_latents = torch.cat(generated_latents_list, dim=2)

        generated_latents_image, generated_latents_disparity = torch.chunk(generated_latents[:, :-self.model_cfg['raymap_dim']], 2, dim=1)
        generated_raymap                                     = generated_latents[:, -self.model_cfg['raymap_dim']:]

        self.raymap_mean = self.raymap_mean.to(self.device)
        self.raymap_std  = self.raymap_std.to( self.device)
        generated_raymap = generated_raymap * self.raymap_std + self.raymap_mean

        generated_trans3d, generated_trans2d = raymap_to_trans_matrix(generated_raymap[:, :, 1:], append_first_reference=True, from_relative_to_absolute=True) 
        
        image     = self.decode_latent(generated_latents_image,     save_memory=save_memory) 
        disparity = self.decode_latent(generated_latents_disparity, save_memory=save_memory) 
        if self.model_cfg.get('no_need_depth', False):
            disparity = torch.zeros_like(disparity).to(disparity)

        return image, disparity, generated_trans3d, generated_trans2d


    def decode_latent(self, latents, save_memory=True, return_pil=False):

        if latents.shape[2] == 1:
            latents = (latents / self.vae_scale_factor) + self.vae_shift_factor
        else:
            latents[:, :, :1] = (latents[:, :, :1] / self.vae_scale_factor) + self.vae_shift_factor
            latents[:, :, 1:] = (latents[:, :, 1:] / self.vae_video_scale_factor) + self.vae_video_shift_factor

        if save_memory:
            # reducing the tile size and temporal chunk window size
            image = self.vae.decode(latents, temporal_chunk=True, window_size=1, tile_sample_min_size=256).sample
        else:
            image = self.vae.decode(latents, temporal_chunk=True, window_size=2, tile_sample_min_size=512).sample

        if not return_pil:
            return image # tensor [b, c, t, h, w] [-1, 1]

        image = image.mul(127.5).add(127.5).clamp(0, 255).byte()
        image = rearrange(image, "B C T H W -> (B T) H W C")
        image = image.cpu().numpy()
        image = self.numpy_to_pil(image)
        
        return image

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def video_guidance_scale(self):
        return self._video_guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 0
        