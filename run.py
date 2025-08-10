import torch
import numpy as np
import imageio
import os
import fire
import re
import time

from PIL import Image
from pipeline import InferencePipeline
from transformers.trainer_utils import set_seed


def create_model_config():

    model_cfg = {
        
        'dit_config': {
            'model_path': './ckpts/transformer',  
        },
        
        'vae_config': {
            'pretrained_model_name_or_path': './ckpts/causal_video_vae', 
            'interpolate': False,
        },

        'scheduler_config': {
            'num_train_timesteps': 1000,
            'gamma': 0.3333,
            'stage_range': [0, 1/3, 2/3, 1],
        },

        'text_encoder_config': {
            'model_path': './ckpts',
            'torch_dtype': torch.float32,
        },

        'raymap_dim': 6,
        'max_temporal_length': 8,
        'frame_per_unit': 1,
        'stages': [1, 2, 4],
        'num_inference_steps': 5,
        'history_guidance_scale': 6.0,
        'history_downsample_ratio': 2,

        'text_embeds_path': './assets/text_embeds_len77.pt',
        'vae_downsample': 8,
        'use_motion_prompt': True,
        'no_need_depth': False,
    }
    return model_cfg


def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    valid_mask = (depthmap > 0.0)
    return X_cam, valid_mask

def depth_edge(depth: torch.Tensor, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: torch.Tensor = None):
    import torch.nn.functional as F

    shape = depth.shape
    depth = depth.reshape(-1, 1, *shape[-2:])
    if mask is not None:
        mask = mask.reshape(-1, 1, *shape[-2:])

    if mask is None:
        diff = (F.max_pool2d(depth, kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(-depth, kernel_size, stride=1, padding=kernel_size // 2))
    else:
        diff = (F.max_pool2d(torch.where(mask, depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(torch.where(mask, -depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2))

    edge = torch.zeros_like(depth, dtype=torch.bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= (diff / depth).nan_to_num_() > rtol
    edge = edge.reshape(*shape)
    return edge

def depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose, **kw):
    X_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)

    X_world = X_cam
    if camera_pose is not None:
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3,  3]
        X_world = np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]

    return X_world, valid_mask

def interpolate_cameras(c2w_list, K_list, k=9):
    from scipy.spatial.transform import Rotation, Slerp

    interpolated_c2w = []
    interpolated_K   = []

    for i in range(len(c2w_list) - 1):
        c2w_A, K_A = c2w_list[i], K_list[i]
        c2w_B, K_B = c2w_list[i+1], K_list[i+1]

        R_A, t_A = c2w_A[:3, :3], c2w_A[:3, 3]
        R_B, t_B = c2w_B[:3, :3], c2w_B[:3, 3]

        quat_A = Rotation.from_matrix(R_A).as_quat()
        quat_B = Rotation.from_matrix(R_B).as_quat()
        slerp  = Slerp([0, 1], Rotation.from_quat([quat_A, quat_B]))
        times  = np.linspace(0, 1, k)
        rots   = slerp(times)

        t_interp = [(1 - alpha)*t_A + alpha*t_B for alpha in times]
        K_interp = [(1 - alpha)*K_A + alpha*K_B for alpha in times]
        factor   = 1

        for j in range(k):
            if i > 0 and j == 0: continue
            c2w_new         = np.eye(4)
            c2w_new[:3, :3] = rots[j].as_matrix()
            c2w_new[:3, 3]  = t_interp[j] * factor
            interpolated_c2w.append(c2w_new)
            interpolated_K.append(K_interp[j])

    return interpolated_c2w, interpolated_K

def save_ply_file(points, mask, image, output_file, trans=np.eye(3), downsample=10):
    from plyfile import PlyData, PlyElement

    h, w, _  = points.shape
    image    = image[:h, :w]
    points   = points[:h, :w]

    if mask is not None:
        points = points[mask].reshape(-1, 3)
        colors = image[mask].reshape(-1, 3)
    else:
        points = points.reshape(-1, 3) 
        colors = image.reshape(-1, 3) 
    points = (trans @ points.T).T
    
    invalid = np.isnan(points).any(axis=-1) + np.isinf(points).any(axis=-1)
    points  = points[~invalid]
    colors  = colors[~invalid]

    invalid = (points > 20).any(axis=-1) # forward to far
    points  = points[~invalid]
    colors  = colors[~invalid]

    choose = np.random.permutation(points.shape[0])[:(points.shape[0] // downsample)]
    points = points[choose]
    colors = colors[choose]

    vertices = []
    for p, c in zip(points, colors):
        vertex = (p[0], p[1], p[2], int(c[0]), int(c[1]), int(c[2]))
        vertices.append(vertex)

    vertex_dtype = np.dtype([
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    vertex_array = np.array(vertices, dtype=vertex_dtype)

    ply_element = PlyElement.describe(vertex_array, 'vertex')
    PlyData([ply_element], text=True).write(output_file)
    print(f'[info] save ply at {output_file}, have {len(vertices)} points.')


def add_controler_on_image(merge, prompt):

    def past_image(background, overlay, position):
        background.paste(overlay, position, overlay)
        return background

    def trans_color(overlay, to_rgb=np.array([244, 234, 42])):
        x                 = np.array(overlay)
        mask              = (x[:, :, -1] > 0)
        x[:, :, :3][mask] = to_rgb
        overlay           = Image.fromarray(x)
        return overlay
        
    controler_icon_path = './assets/icons'
    icon_size     = 29 # 232 / 8
    
    clock         = Image.open(os.path.join(controler_icon_path,         'clock.png')).convert("RGBA").resize((icon_size, icon_size))
    counterclock  = Image.open(os.path.join(controler_icon_path,  'counterclock.png')).convert("RGBA").resize((icon_size, icon_size))

    backward      = Image.open(os.path.join(controler_icon_path,      'backward.png')).convert("RGBA").resize((icon_size, icon_size))
    forward       = Image.open(os.path.join(controler_icon_path,       'forward.png')).convert("RGBA").resize((icon_size, icon_size))
    leftforward   = Image.open(os.path.join(controler_icon_path,   'leftforward.png')).convert("RGBA").resize((icon_size, icon_size))
    leftbackward  = Image.open(os.path.join(controler_icon_path,  'leftbackward.png')).convert("RGBA").resize((icon_size, icon_size))
    rightbackward = Image.open(os.path.join(controler_icon_path, 'rightbackward.png')).convert("RGBA").resize((icon_size, icon_size))
    rightforward  = Image.open(os.path.join(controler_icon_path,  'rightforward.png')).convert("RGBA").resize((icon_size, icon_size))
    left          = Image.open(os.path.join(controler_icon_path,          'left.png')).convert("RGBA").resize((icon_size, icon_size))
    right         = Image.open(os.path.join(controler_icon_path,         'right.png')).convert("RGBA").resize((icon_size, icon_size))

    
    counterclock  =  trans_color(counterclock) if 'counterclockwise' in prompt else counterclock
    clock         =         trans_color(clock) if ' clockwise'       in prompt else clock

    backward      =      trans_color(backward) if 'backward'         in prompt else backward
    backward      =      trans_color(backward) if 'rear left'        in prompt else backward
    backward      =      trans_color(backward) if 'rear right'       in prompt else backward
    
    forward       =       trans_color(forward) if 'forward'          in prompt else forward
    forward       =       trans_color(forward) if 'front left'       in prompt else forward
    forward       =       trans_color(forward) if 'front right'      in prompt else forward

    left          =          trans_color(left) if 'the left'         in prompt else left
    left          =          trans_color(left) if 'front left'       in prompt else left
    left          =          trans_color(left) if 'rear left'        in prompt else left

    right         =         trans_color(right) if 'the right'        in prompt else right
    right         =         trans_color(right) if 'front right'      in prompt else right
    right         =         trans_color(right) if 'rear right'       in prompt else right

    W, H = merge.size
    W    = W // 3

    merge = past_image(merge,       forward, (W//2 - 2*icon_size, H - 2*icon_size))
    merge = past_image(merge,      backward, (W//2 - 2*icon_size, H -   icon_size))
    merge = past_image(merge,          left, (W//2 - 3*icon_size, H -   icon_size))
    merge = past_image(merge,         right, (W//2 -   icon_size, H -   icon_size))

    merge = past_image(merge,  counterclock, (W//2              , H - icon_size//2 - icon_size))
    merge = past_image(merge,         clock, (W//2 +   icon_size, H - icon_size//2 - icon_size))

    return merge

def prepare_input_data(image_path, video_length, height, width, prompt_type, prompt):    
    first_frame = Image.open(image_path).convert("RGB")
    original_width, original_height = first_frame.size
    target_ratio = width / height
    if original_width / original_height > target_ratio:
        new_width = int(original_height * target_ratio)
        left      = (original_width - new_width) // 2
        top       = 0
        right     = left + new_width
        bottom    = original_height
    else:
        new_height = int(original_width / target_ratio)
        left       = 0
        top        = (original_height - new_height) // 2
        right      = original_width
        bottom     = top + new_height
        
    first_frame_cropped = first_frame.crop((left, top, right, bottom))
    first_frame         = first_frame_cropped.resize((width, height))

    if prompt_type == 'action':
        pattern = r'^\((?:[a-z][A-Z]{2}|[A-Z]{2})(?:\)\((?:[a-z][A-Z]{2}|[A-Z]{2}))*\)$'
        assert re.fullmatch(pattern, prompt), 'input prompt is not valid'
        matches = re.findall(r'\((.*?)\)', prompt)
        motion_prompts = ['empty',] # first frame
        trans_prompts  = {
            'S':  'Stay where you are.',
            'L':  'Move to the left.',
            'rL': 'Move to the rear left.',
            'B':  'Move backward.',
            'rR': 'Move to the rear right.',
            'R':  'Move to the right.',
            'fR': 'Move to the front right.',
            'F':  'Move forward.',
            'fL': 'Move to the front left.',
        }

        rot_prompts = {
            'N': 'The perspective hasn\'t changed.',
            'L': 'Rotate the perspective counterclockwise.',
            'R': 'Rotate the perspective clockwise.',
        }
        for m in matches:
            motion_prompts.append(trans_prompts[m[:-1]] + ' ' + rot_prompts[m[-1:]])

    else:
        motion_prompts = [prompt] * 10

    batch_dict = {
        'img': first_frame,
        'prompt': np.array(motion_prompts),
        'prompt_type': prompt_type,
    }
    return batch_dict


def save_video(output, output_path, fps=24, add_controler=False, add_depth=False):
    import matplotlib

    def colorize_depth(depth, min_depth, max_depth, cmap="Spectral"):
        cm = matplotlib.colormaps[cmap]
        depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
        depth_colored = cm(depth, bytes=False)[..., 0:3]  # value from 0 to 1
        return depth_colored

    if os.path.dirname(output_path) and not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    video_tensor = output['pred_img'].squeeze(0)  # -> (C, T, H, W)
    video_tensor = (video_tensor.permute(1, 2, 3, 0) + 1) / 2.0
    video_np     = (video_tensor.to(torch.float32).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)

    if add_depth:
        disparity = output["pred_disparity"].squeeze(0)
        disparity = disparity.mean(dim=0, keepdim=False).to(torch.float32).cpu()#.numpy()
        video_np_  = np.repeat(video_np, repeats=2, axis=2)
        for i, disparity_ in enumerate(disparity):
            mask  = (1. / disparity_) < torch.inf
            min_d = torch.quantile(disparity_[mask], 0.01)
            max_d = torch.quantile(disparity_[mask], 0.99)
            disparity_ = (disparity_ - min_d) / (max_d - min_d)
            disparity_ = torch.clamp(disparity_, 0.0, 1.0)
            disparity_ = 1 - disparity_
            disparity_ = colorize_depth(disparity_.numpy(), 0, 1)
            disparity_ = (disparity_ * 255).astype(np.uint8)

            video_np_[i] = np.concatenate([video_np[i], disparity_], axis=1)
        video_np = video_np_

    frames = [Image.fromarray(frame) for frame in video_np]

    if add_controler:
        for i in range(len(frames)):
            frame = frames[i]
            frame = add_controler_on_image(frame, np.concatenate(output['motion_prompt_list'])[int((i - 1)// 8 + 1)])
            frames[i] = frame
    
    try:
        imageio.mimsave(output_path, frames, fps=fps, quality=8, codec='libx264')
    except Exception as e:
        gif_path = os.path.splitext(output_path)[0] + ".gif"
        imageio.mimsave(gif_path, frames, fps=fps)

def save_ply(output, output_path):
    video_tensor = output['pred_img'].squeeze(0)
    video_tensor = (video_tensor.permute(1, 2, 3, 0) + 1) / 2.0
    video_np     = (video_tensor.to(torch.float32).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    frames       = [frame for frame in video_np]

    disparity    = output['pred_disparity'].squeeze(0)
    disparity    = disparity.mean(dim=0, keepdim=False).to(torch.float32).cpu()
    # pred_depth   = 1. / (disparity + 1e-10)
    pred_depth   = 1. / disparity 

    pred_trans3ds = output["trans3d"].squeeze(0).to(torch.float32).cpu().numpy()
    pred_trans2ds = output["trans2d"].squeeze(0).to(torch.float32).cpu().numpy()
    pred_trans3ds, pred_trans2ds = interpolate_cameras(pred_trans3ds, pred_trans2ds)

    for i, (frame, depth, pred_trans3d, pred_trans2d) in enumerate(zip(frames, pred_depth, pred_trans3ds, pred_trans2ds)):
        if i % 8 != 0: continue # reduce file number
        mask = depth < torch.inf
        edge_mask = depth_edge(depth, atol=0.1).numpy()
        conf      = ~edge_mask | mask.numpy()
        pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(depth.numpy(), pred_trans2d, pred_trans3d)
        conf              = conf | valid_mask
        save_ply_file(pts3d, conf, frame, output_path.replace('.ply', f'_frame{i}.ply'))

def main(
    input_image,
    model_path,
    prompt_type='text',
    prompt='',
    seed=666,
    no_need_depth=False,
    add_controler=False, add_depth=False, add_ply=False,
):
    set_seed(seed)

    # ===================== 1. load config ========================
    model_cfg = create_model_config()
    model_cfg['no_need_depth'] = no_need_depth
    model_cfg['use_motion_prompt'] = (prompt_type == 'action')
    model_cfg['dit_config']['model_path'] = os.path.join(model_path, 'transformer')
    model_cfg['vae_config']['pretrained_model_name_or_path'] = os.path.join(model_path, 'causal_video_vae')
    model_cfg['text_encoder_config']['model_path'] = model_path

    OUTPUT_VIDEO_PATH = "output/generated_video.mp4"
    VIDEO_LENGTH = 57
    VIDEO_HEIGHT = 384
    VIDEO_WIDTH  = 512

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    # ===================== 2. init pipeline ====================
    pipeline = InferencePipeline(
        model_cfg=model_cfg,
        device=DEVICE,
        torch_dtype=DTYPE,
    )

    # ===================== 3. prepare input ====================
    batch_dict = prepare_input_data(input_image, VIDEO_LENGTH, VIDEO_HEIGHT, VIDEO_WIDTH, prompt_type, prompt)

    # ===================== 4. generate =========================
    with torch.no_grad():
        st = time.time()
        output = pipeline.generate(batch_dict)
        ed = time.time()

    # ===================== 5. save output =======================
    save_video(output, OUTPUT_VIDEO_PATH, fps=20, add_controler=(add_controler and (prompt_type == 'action')), add_depth=(add_depth and (model_cfg['no_need_depth'] == False)))

    if add_ply and (not model_cfg['no_need_depth']): 
        save_ply(output, OUTPUT_VIDEO_PATH.replace('.mp4', '.ply'))

    print(f'[info] save result at {OUTPUT_VIDEO_PATH}')


if __name__ == "__main__":
    fire.Fire(main)