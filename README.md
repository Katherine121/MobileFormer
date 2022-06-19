# MobileFormer

## Table of Branches

main - pretrain branch is mobileformer pretrained code  
rembridge - rembridge branch is mobileformer compression code  
tune - tune branch is mobileformer tuned code using structural regularization  
distillation - distillation branch is mobileformer distilled code using Resnet152 model  
rk3399 - rk3399 branch is mobileformer deployed code on Rockchip RK3399ProD  

## Project Structure

(rk3399 branch is as the criterion)  

```angular2html
Pytorch-implementation-of-Mobile-Former  
|-- mobile_former   model structure  
|   |-- bridge.py  mobile->former and former->mobile  
|   |-- former.py  transformer block  
|   |-- mobile.py  mobile block  
|   |-- model.py   the whole model structure  
|-- process_data   data augment oprations  
|   |-- autoaugment.py     autoaugment policy  
|   |-- ops.py     oprations of autoaugment used  
|   |-- utils.py   cutmix policy  
|-- draw.py     draw train loss and test acc pictures  
|-- main.py     train and test mobileformer  
|-- model_generator.py     mobileformer params config(mf151, mf294, mf508)  
|-- test.py     test acc  
|-- to_onnx.py     transform .pt to .onnx  
|-- to_rknn.py     transformer .onnx to .rknn  
|-- valid_onnx.py    validate .onnx model acc  
```

## Train and Test

Using Autoaugment, RandomErasing and cutmix to realize data augment  
Realize mobileformer by mf151, mf294, mf508 in model_generator.py  

```angular2html
test main.py
```

## Inference

https://arxiv.org/pdf/2108.05895.pdf  
