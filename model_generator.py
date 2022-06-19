import torch
from mobile_former.model import MobileFormer
from mobile_former.bridge import Mobile2Former, Former2Mobile
from mobile_former.mobile import Mobile, MobileDown
from mobile_former.former import Former

from thop import profile
from ptflops import get_model_complexity_info
from pytorch_model_summary import summary


def mobile_former_508(num_class, pre_train=False, state_dir=None):
    cfg = {
        'name': 'mf508',
        'token': 6,  # tokens and embed_dim
        'embed': 192,
        'stem': 24,
        'bneck': {'e': 48, 'o': 24, 's': 1},
        'body': [
            {'inp': 24, 'exp': 144, 'out': 40, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 40, 'exp': 120, 'out': 40, 'se': None, 'stride': 1, 'heads': 2},

            {'inp': 40, 'exp': 240, 'out': 72, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 72, 'exp': 216, 'out': 72, 'se': None, 'stride': 1, 'heads': 2},

            {'inp': 72, 'exp': 432, 'out': 128, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 128, 'exp': 512, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 128, 'exp': 768, 'out': 176, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 176, 'exp': 1056, 'out': 176, 'se': None, 'stride': 1, 'heads': 2},

            {'inp': 176, 'exp': 1056, 'out': 240, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 240, 'exp': 1440, 'out': 240, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 240, 'exp': 1440, 'out': 240, 'se': None, 'stride': 1, 'heads': 2},
        ],
        'fc1': 1920,  # hid_layer
        'fc2': num_class   # num_classes
    }
    model = MobileFormer(cfg)
    if pre_train:
        print('Model loading...')
        model = torch.load(state_dir)
        print('Model loaded.')
    else:
        print('Model initialized.')
    return model


def mobile_former_294(num_class, pre_train=False, state_dir=None):
    cfg = {
        'name': 'mf294',
        'token': 6,  # tokens
        'embed': 192,  # embed_dim
        'stem': 16,
        # stage1
        'bneck': {'e': 32, 'o': 16, 's': 1},  # exp out stride
        'body': [
            # stage2
            {'inp': 16, 'exp': 96, 'out': 24, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 24, 'exp': 96, 'out': 24, 'se': None, 'stride': 1, 'heads': 2},
            # stage3
            {'inp': 24, 'exp': 144, 'out': 48, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 48, 'exp': 192, 'out': 48, 'se': None, 'stride': 1, 'heads': 2},
            # stage4
            {'inp': 48, 'exp': 288, 'out': 96, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 96, 'exp': 384, 'out': 96, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 96, 'exp': 576, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 128, 'exp': 768, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
            # stage5
            {'inp': 128, 'exp': 768, 'out': 192, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 192, 'exp': 1152, 'out': 192, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 192, 'exp': 1152, 'out': 192, 'se': None, 'stride': 1, 'heads': 2},
        ],
        'fc1': 1920,  # hid_layer
        'fc2': num_class   # num_classes
    }
    model = MobileFormer(cfg)
    if pre_train:
        print('Model loading...')
        model = torch.load(state_dir)
        print('Model loaded.')
    else:
        print('Model initialized.')
    return model

def mobile_former_151(num_class, pre_train=False, state_dir=None):
    cfg = {
        'name': 'mf151',
        'token': 6,  # tokens
        'embed': 192,  # embed_dim
        'stem': 12,
        # stage1
        'bneck': {'e': 24, 'o': 12, 's': 1},  # exp out stride
        'body': [
            # stage2
            {'inp': 12, 'exp': 72, 'out': 16, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 16, 'exp': 48, 'out': 16, 'se': None, 'stride': 1, 'heads': 2},
            # stage3
            {'inp': 16, 'exp': 96, 'out': 32, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 32, 'exp': 96, 'out': 32, 'se': None, 'stride': 1, 'heads': 2},
            # stage4
            {'inp': 32, 'exp': 192, 'out': 64, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 64, 'exp': 256, 'out': 64, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 64, 'exp': 384, 'out': 88, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 88, 'exp': 528, 'out': 88, 'se': None, 'stride': 1, 'heads': 2},
            # stage5
            {'inp': 88, 'exp': 528, 'out': 128, 'se': None, 'stride': 2, 'heads': 2},
            {'inp': 128, 'exp': 768, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
            {'inp': 128, 'exp': 768, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
        ],
        'fc1': 1280,  # hid_layer
        'fc2': num_class   # num_classes`
    }
    model = MobileFormer(cfg)
    if pre_train:
        print('Model loading...')
        model = torch.load(state_dir)
        print('Model loaded.')
    else:
        print('Model initialized.')
    return model


if __name__ == "__main__":
    model = mobile_former_151(100, pre_train=True, state_dir='./saved_model/mobile_former_151.pt')
    inputs = torch.randn((1, 3, 224, 224)).cuda()
    # # 第一种方法
    # flops, params = profile(model, (inputs,))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    # 第二种方法：每一个block的参数量和计算量都有
    flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)
    # # 第三种方法：每一个block的参数量和计算量都有。更简略
    # print(summary(model, inputs, show_input=False, show_hierarchical=False))
