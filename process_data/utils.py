import math
import torch
import random
import numpy as np


def cutmix(input, target, beta):
    # 随机获得小方块长宽概率
    lam = np.random.beta(beta, beta)
    b = input.size()[0]
    rand_index = torch.randperm(b).cuda()
    target_a = target
    target_b = target[rand_index]
    bx1, by1, bx2, by2 = rand_box(input.size(), lam)
    # 把图片随机裁剪后的小方块换成打乱后其他图片的小方块
    input[:, :, bx1:bx2, by1:by2] = input[rand_index, :, bx1:bx2, by1:by2]
    # 没换的区域的占比
    lam = 1 - ((bx2 - bx1) * (by2 - by1) / (input.size()[-1] * input.size()[-2]))
    return input, target_a, target_b, lam


def cutmix_criterion(criterion, output, target_a, target_b, lam):
    return lam * criterion(output, target_a) + (1. - lam) * criterion(output, target_b)


# 计算随机裁剪小方块的四角坐标
def rand_box(size, lam):
    _, _, h, w = size
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(w * cut_rat)
    cut_h = np.int(h * cut_rat)
    # 在图片上随机取一点作为cut的中心点
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    # 随机点坐标-随机裁剪的宽
    bx1 = np.clip(cx - cut_w // 2, 0, w)
    by1 = np.clip(cy - cut_h // 2, 0, h)
    bx2 = np.clip(cx + cut_w // 2, 0, w)
    by2 = np.clip(cy + cut_h // 2, 0, h)
    # 裁剪的四角坐标
    return bx1, by1, bx2, by2


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        # 一半概率不擦除
        if random.uniform(0, 1) > self.probability:
            return img
        for attempt in range(100):
            # 计算图片面积
            # c h w
            area = img.size()[1] * img.size()[2]
            # 比率范围
            target_area = random.uniform(self.sl, self.sh) * area
            # 宽高比
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            # 要模糊的高宽
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size()[2] and h < img.size()[1]:
                # 要模糊的左上角坐标
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img
        return img


# 每次都不一样
if __name__ == "__main__":
    print(np.random.beta(1,1))
