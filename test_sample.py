import argparse
import cv2
import importlib
import numpy as np
import os
import sys

import torch

from utils.common import scandir, tensor2img, calculate_psnr_ssim
from utils.model_opr import load_model, load_model_filter_list


def read_image_to_tensor(ipath):
    img = cv2.imread(ipath, cv2.IMREAD_COLOR)
    img = np.transpose(img[:, :, ::-1], (2, 0, 1)).astype(np.float32) / 255.0
    img = torch.from_numpy(img).float().unsqueeze(0)

    return img


def lcm(ab):
    a, b = ab[0], ab[1]
    for i in range(min(a, b), 0, -1):
        if a % i == 0 and b % i == 0:
            return a * b // i


if __name__ == '__main__':
    print('---------- Start ----------')

    ### load arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--gt', type=str, default=None)
    parser.add_argument('--noise_level', type=int, default=None)
    parser.add_argument('--sf', action='store_true')
    args = parser.parse_args()
    if not args.noise_level:
        assert args.input is not None, 'Please assign the input value!'
        print('[Input Path]', args.input)
    else:
        assert args.gt is not None, 'Please assign the gt value for denoising!'
        print('[GT Path]', args.gt)
        np.random.seed(seed=0)

    if args.output:
        os.makedirs(args.output, exist_ok=True)


    ### check config
    config_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
    config_files = [
            os.path.splitext(os.path.basename(v))[0] for v in scandir(config_root)
            if v.endswith('.py')
    ]
    config_file = os.path.basename(args.config).split('.')[0]
    assert config_file in config_files, 'Illegal config!'
    module_reg = importlib.import_module(f'configs.{config_file}')
    config = getattr(module_reg, 'Config', None)


    ### build model
    if not args.sf:
        from utils.modules.edt import Network
    else:
        from utils.modules.edtsf import Network

    # # For multi-task model, only build one branch for testing.
    # config.MODEL.SCALES = [2]  # 2, 3, 4
    # config.MODEL.NOISE_LEVELS = []  # 15, 25, 50
    # config.MODEL.RAIN_LEVELS = []  # 'L', 'H'
    # config.VAL.SCALES = config.MODEL.SCALES  # 2, 3, 4
    # config.VAL.NOISE_LEVELS = config.MODEL.NOISE_LEVELS  # 15, 25, 50
    # config.VAL.RAIN_LEVELS = config.MODEL.RAIN_LEVELS  # 'L', 'H'

    model = Network(config)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    # For multi-task model, only load one branch using filter.
    # Optional values of filter_list: 'x2', 'x3', 'x4', 'g15', 'g25', 'g50', 'dr_L', 'dr_H'.
    load_model_filter_list(model, args.model, filter_list=[])
    # load_model_filter_list(model, args.model, filter_list=['x3', 'x4'])


    ### load images
    legal_types = ['.png', '.jpg', '.tif']
    if args.input:
        ipath_l = []
        for f in sorted(os.listdir(args.input)):
            if os.path.splitext(f)[1] in legal_types:
                ipath_l.append(os.path.join(args.input, f))
    if args.gt:
        gpath_l = []
        for f in sorted(os.listdir(args.gt)):
            if os.path.splitext(f)[1] in legal_types:
                gpath_l.append(os.path.join(args.gt, f))
    if args.noise_level:
        ipath_l = gpath_l  # produce lq by adding noise to gt


    ### inference
    model.eval()
    with torch.no_grad():
        scales = []  # upsampling sclaes
        to_ys = []  # convert to y channel or not
        bds = []  # crop boundary pixels
        for s in config.VAL.SCALES:
            scales.append(s)
            to_ys.append(True)
            bds.append(s)
        for nl in config.VAL.NOISE_LEVELS:
            scales.append(1)
            to_ys.append(False)
            bds.append(0)
        for rl in config.VAL.RAIN_LEVELS:
            scales.append(1)
            to_ys.append(True)
            bds.append(0)

        all_tasks = config.VAL.SCALES + config.VAL.NOISE_LEVELS + config.VAL.RAIN_LEVELS
        psnr_l = [[] for _ in all_tasks]
        ssim_l = [[] for _ in all_tasks]

        # for batch=1
        for idx, lq_path in enumerate(ipath_l):
            img_name = lq_path.split('/')[-1]
            lq_img = cv2.imread(lq_path, cv2.IMREAD_COLOR).astype(np.float32)
            if args.noise_level:
                noise = np.random.normal(loc=0.0, scale=args.noise_level, size=lq_img.shape)
                lq_img = lq_img + noise
            lq_img = np.transpose(lq_img[:, :, ::-1], (2, 0, 1)) / 255.0
            lq_img = torch.from_numpy(lq_img).float().unsqueeze(0)
            lq_imgs = [lq_img.to(device)]

            lqs = []
            h_olds = []
            w_olds = []
            for s, lq_img in zip(scales, lq_imgs):
                _, _, h_old, w_old = lq_img.size()
                if s > 1:
                    window_size = lcm(config.MODEL.WINDOW_SIZE)
                else:
                    window_size = lcm(config.MODEL.WINDOW_SIZE) * 2 ** config.MODEL.DEPTH
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                lq_img = torch.cat([lq_img, torch.flip(lq_img, [2])], 2)[:, :, :h_old + h_pad, :]
                lq_img = torch.cat([lq_img, torch.flip(lq_img, [3])], 3)[:, :, :, :w_old + w_pad]
                lqs.append(lq_img)
                h_olds.append(h_old)
                w_olds.append(w_old)

            preds = model(lqs)
            outputs = []
            for pred, s, h_old, w_old in zip(preds, scales, h_olds, w_olds):
                outputs.append(tensor2img(pred[..., :h_old * s, : w_old * s]))

            if args.output:
                cv2.imwrite(os.path.join(args.output, img_name), outputs[0])

            if args.gt:
                gts = [cv2.imread(gpath_l[idx]), cv2.IMREAD_COLOR]
                for i, (to_y, bd, output, gt) in enumerate(zip(to_ys, bds, outputs, gts)):
                    psnr, ssim = calculate_psnr_ssim(output, gt, to_y=to_y, bd=bd)
                    psnr_l[i].append(psnr)
                    ssim_l[i].append(ssim)

        if args.gt:
            avg_psnr, avg_ssim = [], []
            for psnr, ssim in zip(psnr_l, ssim_l):
                avg_psnr.append(sum(psnr) / len(psnr))
                avg_ssim.append(sum(ssim) / len(ssim))
            for i, (ta, psnr, ssim) in enumerate(zip(all_tasks, avg_psnr, avg_ssim)):
                print(f'[Val Result - {ta}] PSNR: {psnr:.4f}, SSIM: {ssim:.4f}')

