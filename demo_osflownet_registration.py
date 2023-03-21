import sys
sys.path.append('core')
import argparse
import os
import numpy as np
import torch
from PIL import Image
from osflownet.net_model import OSFlowNet
from utils.utils import InputPadder
import os.path as osp
from utils import flow_viz

DEVICE = torch.device("cuda:0")


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    if len(img.shape) == 2:
        img = np.tile(img[..., None], (1, 1, 3))
    else:
        img = img[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)



def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = -flo
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    import matplotlib.pyplot as plt
    plt.imshow(img_flo / 255.0)
    plt.show()



def demo(args):
    import time
    tic = time.time()
    model = torch.nn.DataParallel(OSFlowNet(args), device_ids=args.gpus)
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        name_opt = 'im_opt.tif' # warp_opt, warp_opt11.png
        name_sar = 'im_sar.tif'

        file_opt = osp.join(args.path, name_opt)
        file_sar = osp.join(args.path, name_sar)

        image1 = load_image(file_opt)
        image2 = load_image(file_sar)

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_s, flow_up = model(image1, image2, iters=12, test_mode=True)  # iters=20

        toc = time.time()
        tt1 = toc-tic
        print('without finetune time:')
        print(tt1)

        viz(image1, flow_up)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='checkpoints/osflownet_model.pth',
                        help="restore checkpoint")
    parser.add_argument('--path', default='ims/', help="dataset for evaluation") # demo-frames2
    parser.add_argument('--small', action='store_true', default=True, help='use small model') # False, True
    parser.add_argument('--mixed_precision', action='store_true', default=False, help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_false', default=False,
                        help='use efficent correlation implementation')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='enables CUDA training')
    args = parser.parse_args()
    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    args = parser.parse_args()
    demo(args)
