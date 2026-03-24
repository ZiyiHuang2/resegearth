import torch
from segearth_r2.model.mipha.model.multimodal_encoder.builder import build_vision_tower

vt = build_vision_tower(
    "/home/wangcj/huangziyi/SegEarth-R2/pretrained_model/CLIP/siglip2-so400m-patch14-384"
)
vt.eval()

x = torch.randn(2, 3, 384, 384)
with torch.no_grad():
    y = vt(x)

print(y.shape)