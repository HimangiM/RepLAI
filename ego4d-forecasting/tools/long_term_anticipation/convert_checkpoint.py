import torch
import sys
import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_green(msg):
    print(bcolors.OKGREEN + msg + bcolors.ENDC)

def print_yellow(msg):
    print(bcolors.WARNING + msg + bcolors.ENDC)

if __name__ == '__main__':
    ckp = torch.load(sys.argv[1], map_location='cpu')['model_av']
    ckp_video = {k.replace('module.video_model.', ''): ckp[k] for k in ckp if k.startswith('module.video_model.')}
    for k in ckp:
        if k.startswith('module.backbone.video.'):
            ckp_video[k.replace('module.backbone.video.', '')] = ckp[k]
            print_green(f"Converted {k} -> {k.replace('module.backbone.video.', '')}")
        else:
            print_yellow(f"Ignored {k}")


    fn, ext = os.path.splitext(sys.argv[1])
    dst = f"{fn}-ego{ext}"
    torch.save({'state_dict': ckp_video}, dst)
    print(f"Converted checkpoint\n{dst}")
