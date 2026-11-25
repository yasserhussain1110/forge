from math import pi
import torch

cos = None
sin = None
N = 0

def do_rope(tensor, rope_theta, head_dim):
    global cos, sin, N
    if N != tensor.size()[0]:
        angles = get_angles(len(tensor), rope_theta, head_dim)
        angles = torch.cat([angles, angles], dim=1)
        cos = angles.cos()
        cos = cos[:, None, :]
        sin = angles.sin()
        sin = sin[:, None, :]
        cos = cos.to(tensor.device)
        sin = sin.to(tensor.device)
        N = tensor.size()[0]
    left_half = tensor[..., :tensor.shape[-1] // 2]
    right_half = tensor[..., tensor.shape[-1] // 2:]
    swapped = torch.concat([-right_half, left_half], dim=2)
    return tensor * cos + swapped * sin

def adjust_speeds(speeds):
    wavelens = 2 * pi / speeds
    smooth = (8192 / wavelens - 1) / (4 - 1)
    slow = speeds / 32
    blend = (1 - smooth) * (speeds / 32) + smooth * speeds
    inner = torch.where(wavelens > 8192, slow, blend)
    result = torch.where(wavelens < 2048, speeds, inner)
    return result

def get_speeds(rope_theta, head_dim):
    i = torch.arange(head_dim / 2)
    speeds = 1 / rope_theta ** (2 * i / head_dim)
    return speeds

def get_angles(T, rope_theta, head_dim):
    token_nums = torch.arange(T)
    token_nums = token_nums[:,None]
    speeds = get_speeds(rope_theta, head_dim)
    return token_nums * adjust_speeds(speeds)