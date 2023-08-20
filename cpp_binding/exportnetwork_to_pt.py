import torch
import torchvision
import sys
sys.path.append("/home/lamlam/code/deep_learned_visual_features/src/model")
from unet import UNet 

model = UNet(3,1,16)
file_path = "/home/lamlam/data/networks/network_multiseason_layer16.pth"
torch.save(model.state_dict(), file_path)
model.eval()
example = torch.rand(1, 3, 384, 512)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("/home/lamlam/data/cpp_data/multiseason.pt")
