import glob, os, sys
sys.path.append('Pytorch_Retinaface')

import torch.backends.cudnn as cudnn
from tqdm import tqdm
import cv2
import torch
import argparse

from generate_pairs import Estimator3D
from model.networks import CFRNet
from torchvision.transforms import ToTensor


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=f'cuda:{device}')
    
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def normalize(img):
    return (img-0.5)*2

def main():
    parser = argparse.ArgumentParser(description='CFR-GAN Inference')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    args = parser.parse_args()

    # Initialize model
    model = CFRNet()
    if not args.cpu and torch.cuda.is_available():
        model = model.cuda()
        torch.backends.cudnn.benchmark = True
    else:
        model = model.cpu()
    
    # Load model weights
    model = load_model(model, args.model, args.cpu)
    model.eval()

    # Load and preprocess input image
    img = cv2.imread(args.input)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = ToTensor()(img).unsqueeze(0)
    
    if not args.cpu and torch.cuda.is_available():
        img = img.cuda()

    # Run inference
    with torch.no_grad():
        output = model(img)
    
    # Save output
    output = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    output = (output * 255).astype('uint8')
    cv2.imwrite(args.output, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    main()

