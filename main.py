import os
import torch.nn as nn
import torchvision.datasets as dset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import sklearn.model_selection as ms

from model_generator import *
from process_data import autoaugment
from process_data.mixup import mixup_data, mixup_criterion


def check_accuracy(loader, device, dtype, model):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        t = 0
        for x, y in loader:
            x = x.to(device, dtype=dtype)
            y = y.to(device, dtype=torch.long)
            scores = model(x)
            # _,是batch_size*概率，preds是batch_size*最大概率的列号
            _, preds = scores.max(1)
            num_correct = (preds == y).sum()
            num_samples = preds.size(0)
            total_correct += num_correct
            total_samples += num_samples

            # 每100个iteration打印一次测试集准确率
            if t % 100 == 0:
                print('预测正确的图片数目' + str(num_correct))
                print('总共的图片数目' + str(num_samples))
            t += 1
        acc = float(total_correct) / total_samples
    return acc


def train(
        loader_train=None, loader_val=None,
        device=None, dtype=None,
        model=None,
        criterion=nn.CrossEntropyLoss(),
        scheduler=None, optimizer=None,
        epochs=450, save_epochs=3
):
    acc = 0
    accs = [0]
    losses = []

    model = model.to(device)

    for e in range(epochs):
        model.train()
        total_loss = 0
        for t, (x, y) in enumerate(loader_train):
            x = x.to(device=device, dtype=dtype, non_blocking=True)
            y = y.to(device=device, dtype=torch.long, non_blocking=True)

            # 原x+混x，原t，混t，原混比
            inputs, targets_a, targets_b, lam = mixup_data(x, y, 1)
            # 原x+混x->原y+混y
            outputs = model(inputs)

            # 原y+混y和原t，混t求损失：lam越大，小方块越小，被识别成真图片的概率越大
            # 2
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss_value = np.array(loss.item())
            total_loss += loss_value

            # 1
            optimizer.zero_grad()
            # 3
            loss.backward()
            # 4
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # 200个iteration打印一下训练集损失
            if t % 200 == 0:
                print("Iteration:" + str(t) + ', average Loss = ' + str(loss_value))

        total_loss /= t
        losses.append(total_loss)

        acc = check_accuracy(loader_val, device, dtype, model)
        accs.append(np.array(acc))

        # 每个epoch记录一次测试集准确率和所有batch的平均训练损失
        print("Epoch:" + str(e) + ', Val acc = ' + str(acc) + ', average Loss = ' + str(total_loss))
        # 将每个epoch的平均损失写入文件
        with open("./saved_model/avgloss.txt", "a") as file1:
            file1.write(str(total_loss) + '\n')
        file1.close()
        # 将每个epoch的测试集准确率写入文件
        with open("./saved_model/testacc.txt", "a") as file2:
            file2.write(str(acc) + '\n')
        file2.close()

        # 如果到了保存的epoch或者是训练完成的最后一个epoch
        if (e % save_epochs == 0 and e != 0) or e == epochs - 1 or acc >= 0.74:
            model.eval()
            # 保存模型参数
            torch.save(model.state_dict(), './saved_model/mobile_former_151.pth')
            # 保存模型结构
            torch.save(model, './saved_model/mobile_former_151.pt')
            # 保存jit模型
            trace_model = torch.jit.trace(model, torch.Tensor(1, 3, 224, 224).cuda())
            torch.jit.save(trace_model, './saved_model/mobile_former_jit.pt')
    return acc


def run(
        loader_train=None, loader_val=None,
        device=None, dtype=None,
        model=None,
        criterion=nn.CrossEntropyLoss(),
        epoch=450, lr=0.0009, wd=0.10,
        save_epochs=3,
        MULTI_GPU=True,
):
    epochs = epoch
    model_ = model
    learning_rate = lr
    weight_decay = wd
    print('Training under lr: ' + str(lr) + ' , wd: ' + str(wd) + ' for ', str(epochs), ' epochs.')

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wd)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3)

    if MULTI_GPU:
        optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
        lr_scheduler = nn.DataParallel(lr_scheduler, device_ids=device_ids)

    args = {
        'loader_train': loader_train, 'loader_val': loader_val,
        'device': device, 'dtype': dtype,
        'model': model_,
        'criterion': criterion,
        'scheduler': lr_scheduler.module, 'optimizer': optimizer.module,
        'epochs': epochs,
        'save_epochs': save_epochs,
    }
    print('#############################     Training...     #############################')
    val_acc = train(**args)
    # 最后一个epoch的最后一次测试集准确率
    print('Training for ' + str(epochs) + ' epochs, learning rate: ', learning_rate, ', weight decay: ',
          weight_decay, ', Val acc: ', val_acc)


if __name__ == '__main__':
    print('############################### Dataset loading ###############################')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_aug = transforms.Compose([
        autoaugment.ImageNetPolicy(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # 接收tensor
        transforms.RandomErasing(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    image_train = dset.ImageFolder(root='/datasets/ILSVRC2012/train/', transform=transform_aug)
    loader_train = DataLoader(image_train, batch_size=512, shuffle=True)
    print(len(image_train))

    image_val = dset.ImageFolder(root='./val/', transform=transform)
    loader_val = DataLoader(image_val, batch_size=256, shuffle=False)
    print(len(image_val))

    print('###############################  Dataset loaded  ##############################')

    print('############################### multi GPU loading ###############################')

    # 检测机器是否有多张显卡
    if torch.cuda.device_count() > 1:
        MULTI_GPU = True
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
        device_ids = [0, 1]
    else:
        MULTI_GPU = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('############################### multi GPU loaded ###############################')

    print('############################### model loading ###############################')

    model = mobile_former_151(1000)
    if MULTI_GPU:
        model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)

    print('############################### model loaded ###############################')

    args = {
        'loader_train': loader_train, 'loader_val': loader_val,
        'device': device, 'dtype': torch.float32,
        'model': model,
        'criterion': nn.CrossEntropyLoss(),
        'epoch': 450, 'lr': 0.0009, 'wd': 0.10,
        'save_epochs': 3,
        'MULTI_GPU': True,
    }
    run(**args)
