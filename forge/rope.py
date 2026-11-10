import math
import torch

def compute_rope_inv_freq(rope_theta, head_dim, device=None):
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float, device=device) / head_dim
    inv_freq = 1.0 / (rope_theta ** exponents)
    return inv_freq

def apply_llama3_scaling(inv_freq, factor, low_freq_factor, high_freq_factor, original_max_position_embeddings):
    low_freq_wavelen = original_max_position_embeddings / low_freq_factor
    high_freq_wavelen = original_max_position_embeddings / high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    inv_freq_scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (original_max_position_embeddings / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = ((1 - smooth_factor) * inv_freq_scaled / factor + smooth_factor * inv_freq_scaled)
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_scaled)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)
    return q_rotated.to(q.dtype), k_rotated.to(k.dtype)
