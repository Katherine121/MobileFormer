import os
import time

import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from process_data import autoaugment
from model_generator import *

torch.set_printoptions(profile="full")


{19: 'cattle', 29: 'dinosaur', 0: 'apple', 11: 'boy', 1: 'aquarium_fish', 86: 'telephone', 90: 'train', 28: 'cup', 23: 'cloud', 31: 'elephant', 39: 'keyboard', 96: 'willow_tree', 82: 'sunflower', 17: 'castle', 71: 'sea', 8: 'bicycle', 97: 'wolf', 80: 'squirrel', 74: 'shrew', 59: 'pine_tree', 70: 'rose', 87: 'television', 84: 'table', 64: 'possum', 52: 'oak_tree', 42: 'leopard', 47: 'maple_tree', 65: 'rabbit', 21: 'chimpanzee', 22: 'clock', 81: 'streetcar', 24: 'cockroach', 78: 'snake', 45: 'lobster', 49: 'mountain', 56: 'palm_tree', 76: 'skyscraper', 89: 'tractor', 73: 'shark', 14: 'butterfly', 9: 'bottle', 6: 'bee', 20: 'chair', 98: 'woman', 36: 'hamster', 55: 'otter', 72: 'seal', 43: 'lion', 51: 'mushroom', 35: 'girl', 83: 'sweet_pepper', 33: 'forest', 27: 'crocodile', 53: 'orange', 92: 'tulip', 50: 'mouse', 15: 'camel', 18: 'caterpillar', 46: 'man', 75: 'skunk', 38: 'kangaroo', 66: 'raccoon', 77: 'snail', 69: 'rocket', 95: 'whale', 99: 'worm', 93: 'turtle', 4: 'beaver', 61: 'plate', 94: 'wardrobe', 68: 'road', 34: 'fox', 32: 'flatfish', 88: 'tiger', 67: 'ray', 30: 'dolphin', 62: 'poppy', 63: 'porcupine', 40: 'lamp', 26: 'crab', 48: 'motorcycle', 79: 'spider', 85: 'tank', 54: 'orchid', 44: 'lizard', 7: 'beetle', 12: 'bridge', 2: 'baby', 41: 'lawn_mower', 37: 'house', 13: 'bus', 25: 'couch', 10: 'bowl', 57: 'pear', 5: 'bed', 60: 'plain', 91: 'trout', 3: 'bear', 58: 'pickup_truck', 16: 'can'}
{19: '11-large_omnivores_and_herbivores', 29: '15-reptiles', 0: '4-fruit_and_vegetables', 11: '14-people', 1: '1-fish', 86: '5-household_electrical_devices', 90: '18-vehicles_1', 28: '3-food_containers', 23: '10-large_natural_outdoor_scenes', 31: '11-large_omnivores_and_herbivores', 39: '5-household_electrical_devices', 96: '17-trees', 82: '2-flowers', 17: '9-large_man-made_outdoor_things', 71: '10-large_natural_outdoor_scenes', 8: '18-vehicles_1', 97: '8-large_carnivores', 80: '16-small_mammals', 74: '16-small_mammals', 59: '17-trees', 70: '2-flowers', 87: '5-household_electrical_devices', 84: '6-household_furniture', 64: '12-medium_mammals', 52: '17-trees', 42: '8-large_carnivores', 47: '17-trees', 65: '16-small_mammals', 21: '11-large_omnivores_and_herbivores', 22: '5-household_electrical_devices', 81: '19-vehicles_2', 24: '7-insects', 78: '15-reptiles', 45: '13-non-insect_invertebrates', 49: '10-large_natural_outdoor_scenes', 56: '17-trees', 76: '9-large_man-made_outdoor_things', 89: '19-vehicles_2', 73: '1-fish', 14: '7-insects', 9: '3-food_containers', 6: '7-insects', 20: '6-household_furniture', 98: '14-people', 36: '16-small_mammals', 55: '0-aquatic_mammals', 72: '0-aquatic_mammals', 43: '8-large_carnivores', 51: '4-fruit_and_vegetables', 35: '14-people', 83: '4-fruit_and_vegetables', 33: '10-large_natural_outdoor_scenes', 27: '15-reptiles', 53: '4-fruit_and_vegetables', 92: '2-flowers', 50: '16-small_mammals', 15: '11-large_omnivores_and_herbivores', 18: '7-insects', 46: '14-people', 75: '12-medium_mammals', 38: '11-large_omnivores_and_herbivores', 66: '12-medium_mammals', 77: '13-non-insect_invertebrates', 69: '19-vehicles_2', 95: '0-aquatic_mammals', 99: '13-non-insect_invertebrates', 93: '15-reptiles', 4: '0-aquatic_mammals', 61: '3-food_containers', 94: '6-household_furniture', 68: '9-large_man-made_outdoor_things', 34: '12-medium_mammals', 32: '1-fish', 88: '8-large_carnivores', 67: '1-fish', 30: '0-aquatic_mammals', 62: '2-flowers', 63: '12-medium_mammals', 40: '5-household_electrical_devices', 26: '13-non-insect_invertebrates', 48: '18-vehicles_1', 79: '13-non-insect_invertebrates', 85: '19-vehicles_2', 54: '2-flowers', 44: '15-reptiles', 7: '7-insects', 12: '9-large_man-made_outdoor_things', 2: '14-people', 41: '19-vehicles_2', 37: '9-large_man-made_outdoor_things', 13: '18-vehicles_1', 25: '6-household_furniture', 10: '3-food_containers', 57: '4-fruit_and_vegetables', 5: '6-household_furniture', 60: '10-large_natural_outdoor_scenes', 91: '1-fish', 3: '8-large_carnivores', 58: '18-vehicles_1', 16: '3-food_containers'}


if __name__ == "__main__":
    # 加载pt模型
    model = mobile_former_151(100, pre_train=True, state_dir="./saved_model/mobile_former_151.pt")
    model.eval()

    num_correct = 0
    num_samples = 0
    total_correct = 0
    total_samples = 0

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_aug = transforms.Compose([
        autoaugment.CIFAR10Policy(),
        transforms.Resize(224),
        transforms.ToTensor(),
        # 接收tensor
        transforms.RandomErasing(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    cifar_val = torchvision.datasets.CIFAR100('./dataset/', train=False, download=True, transform=transform)
    # cifar_val_aug = torchvision.datasets.CIFAR100('./dataset/', train=False, download=True, transform=transform_aug)
    # cifar_val += cifar_val_aug

    loader_val = DataLoader(cifar_val, batch_size=16, shuffle=True, pin_memory=True)
    print(len(cifar_val))

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    with torch.no_grad():
        for x, y in loader_val:
            x = x.to(torch.device('cuda'), dtype=torch.float32)
            y = y.to(torch.device('cuda'), dtype=torch.long)
            scores = model(x)
            # _,是batch_size*概率，preds是batch_size*最大概率的列号
            _, preds = scores.max(1)
            num_correct = (preds == y).sum()
            num_samples = preds.size(0)
            total_correct += num_correct
            total_samples += num_samples
            print(float(total_correct) / total_samples)

        acc = float(total_correct) / total_samples
    print(acc)

    image_PIL = Image.open('./test_img/orange.jpg')
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image_tensor = transform(image_PIL)
    # 以下语句等效于 image_tensor = torch.unsqueeze(image_tensor, 0)
    image_tensor.unsqueeze_(0)
    print(image_tensor.shape)
    image_tensor = image_tensor.cuda()

    starttime = time.time()
    out = model(image_tensor)
    endtime = time.time()
    print(int(round((endtime - starttime) * 1000)))
    print(out.shape)
    # 得到预测结果，并且从大到小排序
    _, preds = out.max(1)
    print(preds)