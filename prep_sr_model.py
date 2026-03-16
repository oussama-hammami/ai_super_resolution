from FSRCNN_Pytorch.utils.common import *
from FSRCNN_Pytorch.model import FSRCNN
from config import device, ckpt_path
import torch

# -----------------------------------------------------------
# model config
# -----------------------------------------------------------

scale = 4
sigma = 0.3 if scale == 2 else 0.2


# -----------------------------------------------------------
# test 
# -----------------------------------------------------------

def prepare_sr_model()->FSRCNN:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FSRCNN(scale, device)
    model.load_weights(ckpt_path)
    print(f"model loaded!")
    return model

def run_inferance(model, lr_image):
        lr_image = lr_image[:, :, ::-1].copy()  # BGR -> RGB
        lr_image = np.transpose(lr_image, (2, 0, 1))
        lr_image = torch.from_numpy(lr_image)
        lr_image = gaussian_blur(lr_image, sigma=sigma)
        lr_image = rgb2ycbcr(lr_image)
        lr_image = norm01(lr_image)
        lr_image = torch.unsqueeze(lr_image, dim=0).to(device)
        print(lr_image.shape)
        sr_image = model.predict(lr_image)[0]
        sr_image = denorm01(sr_image)
        sr_image = sr_image.type(torch.uint8)
        sr_image = ycbcr2rgb(sr_image)  # YCbCr -> RGB
        sr_image = sr_image.cpu().detach().numpy()
        sr_image = np.transpose(sr_image, (1, 2, 0))
        sr_image = sr_image[:, :, ::-1].copy()  # RGB -> BGR for cv2
        print(sr_image.shape, flush = True)
        return sr_image

