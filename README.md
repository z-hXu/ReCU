# ReCU: Reviving the Dead Weights in Binary Neural Networks ([Paper Link](http://arxiv.org/abs/2103.12369)) ![]( https://visitor-badge.glitch.me/badge?page_id=bnn_recu).
Pytorch implementation of ReCU in ICCV 2021.

## Tips

Any problem, please contact the first author (Email: ianhsu@stu.xmu.edu.cn). 

## Dependencies
* Python 3.7
* Pytorch 1.1.0

## Citation
If you find ReCU useful in your research, please consider citing:
```
@inproceedings{xu2021recu,
  title={ReCU: Reviving the Dead Weights in Binary Neural Networks},
  author={Xu, Zihan and Lin, Mingbao and Liu, Jianzhuang and Chen, Jie and Shao, Ling and Gao, Yue and Tian, Yonghong and Ji, Rongrong},
  booktitle={Proceedings of the International Conference on Computer Vision (ICCV)},
  pages={5198--5208},
  year={2021}
}
```

## Training on CIFAR-10
```bash
python -u main.py \
--gpus 0 \
--model resnet18_1w1a (or resnet20_1w1a or vgg_small_1w1a) \
--results_dir ./result \
--data_path [DATA_PATH] \
--dataset cifar10 \
--epochs 600 \
--lr 0.1 \
-b 256 \
-bt 128 \
--lr_type cos \
--warm_up \
--weight_decay 5e-4 \
--tau_min 0.85 \
--tau_max 0.99 \
```

### Optional arguments
```
optinal arguments:
    --gpus                    Specify gpus, e.g., 0, 1  
    --seed                    Fix random seeds (Code efficiency will be slightly affected)
                              set to 0 to disable
                              default: 0
    --model / -a              Choose model   
                              default: resnet18_1w1a   
                              options: resnet20_1w1a / vgg_small_1w1a       
    --results_dir             Path to save directory  
    --save                    Path to save folder    
    --data_path               Path to dataset    
    --evaluate / -e           Evaluate  
    --dataset                 Choose dataset
                              default: cifar10
                              options: cifar100 / tinyimagenet / imagenet  
    --epochs                  Number of training epochs
                              default: 600  
    --lr                      Initial learning rate
                              default: 0.1  
    --batch_size / -b         Batch size
                              default: 256   
    --batch_size_test / -bt   Evaluating batch size
                              default: 128  
    --momentum                Momentum
                              default: 0.9  
    --workers                 Data loading workers
                              default: 8  
    --print_freq              Print frequency 
                              default: 100  
    --time_estimate           Estimate finish time of the program
                              set to 0 to disable
                              default: 1     
    --lr_type                 Type of learning rate scheduler
                              default: cos (CosineAnnealingLR)
                              options: step (MultiStepLR)  
    --lr_decay_step           If choose MultiStepLR, set milestones.
                              e.g., 30 60 90      
    --warm_up                 Use warm up  
    --weight_decay            Weight decay
                              default: 5e-4  
    --tau_min                 Minimum of param τ in ReCU(x)
                              default: 0.85 
    --tau_max                 Maximum of param τ in ReCU(x)
                              default: 0.99  
    --resume                  Reload last checkpoint if the training is terminated by accident.
```

### Results on CIFAR-10. 
|Quantized model Link                                                                                  | batch_size | batch_size_test | epochs| training method | Top-1 |
|:----------------------------------------------------------------------------------------------------:|:----------:|:---------------:|:-----:|:---------------:|:-----:|
|[resnet18_1w1a](https://drive.google.com/drive/folders/1g8dHSWKgVfETNj-5oXNWiTI7hiwOgmRS?usp=sharing) |    256     |       128       | 600   |     vanilla     |  92.8 |
|[resnet18_1w1a](https://drive.google.com/drive/folders/1k9znZGMcvGe8QfJNcMhytQYy8OhARsnF?usp=sharing) |    256     |       128       | 600   |     finetune    |  93.2 |
|[resnet20_1w1a](https://drive.google.com/drive/folders/1ikmlm2H5ZjsZYiUvb3qFh4QxSvN6C3ZJ?usp=sharing) |    256     |       128       | 600   |     vanilla     |  87.5 |
|[resnet20_1w1a](https://drive.google.com/drive/folders/1X3DspRZPKum-dH4Z52M5jt5zNaXv-o1X?usp=sharing) |    256     |       128       | 600   |     finetune    |  88.0 |
|[vgg_small_1w1a](https://drive.google.com/drive/folders/1bskc10Hb8RkNp-Btd9xak2aU4PNPacEo?usp=sharing)|    256     |       128       | 600   |     vanilla     |  92.2 | 
|[vgg_small_1w1a](https://drive.google.com/drive/folders/18Im9WcxHC-Q5Rr7QGCHvixLDw9r03Rjn?usp=sharing)|    256     |       128       | 600   |     finetune    |  93.3 | 

To ensure the reproducibility, please refer to our training details provided in the links for our quantized models.

To verify the performance of our quantized models on CIFAR-10, please use the following command:
```bash 
python -u main.py \
--gpus 0 \
-e [best_model_path] \
--model resnet18_1w1a (resnet20_1w1a or vgg_small_1w1a) \
--data_path [DATA_PATH] \
--dataset cifar10 \
-bt 128 \
```
## Training on ImageNet
```bash
python -u main.py \
--gpus 0,1 \
--model resnet18_1w1a (or resnet34_1w1a) \
--results_dir ./result \
--data_path [DATA_PATH] \
--dataset imagenet \
--epochs 200 \
--lr 0.1 \
-b 512 \
-bt 256 \
--lr_type cos \
--warm_up \
--weight_decay 1e-4 \
--tau_min 0.85 \
--tau_max 0.99 \
```
Other arguments are the same as those on CIFAR-10.

### Optional arguments
```
optinal arguments:
    --model / -a              Choose model   
                              default: resnet18_1w1a   
                              options: resnet34_1w1a  
```
We provide two types of dataloaders for ImageNet by [nvidia-dali](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html) and [Pytorch](https://pytorch.org/docs/stable/data.html) respectively. We empirically find that the dataloader by nvidia-dali can offer higher training efficiency than Pytorch (14min vs 28min on 2 Tesla V100 for one epoch when training ResNet-18), but the model accuracy would be affected. The reported experimental results are on the basis of Pytorch. If interested, you can try dataloader by nvidia-dali via adding the optional argument ```--use_dali``` to obtain a shorter training time.  

Nvidia-dali package
```bash
# for CUDA 10
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100
# for CUDA 11
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110
```

### Results on ImageNet

|Quantized model Link                                                                                  | batch_size | batch_size_test | epochs| use_dali| training method | Top-1 | Top-5 | 
|:----------------------------------------------------------------------------------------------------:|:----------:|:---------------:|:-----:|:-------:|:---------------:|:-----:|:-----:|
| [resnet18_1w1a](https://drive.google.com/drive/folders/1RM1QAf8fDO-DR_1woLqNzF_rHrKJWLSF?usp=sharing)|    512     |       256       |  200  |   ✘    |     vanilla     | 60.98 | 82.57 |
| [resnet18_1w1a](https://drive.google.com/drive/folders/1fv98WNR503iFRwIHrhFUnQT_bQSwpWFN?usp=sharing)|    512     |       256       |  200  |   ✘    |     finetune    | 61.20 | 82.93 |
| [resnet34_1w1a](https://drive.google.com/drive/folders/1688Juur4lYFWqZMgwvlJwT3DcqeFHg_q?usp=sharing)|    512     |       256       |  200  |   ✘    |     vanilla     | 65.10 | 85.78 |
| [resnet34_1w1a](https://drive.google.com/drive/folders/1CmApN8sgGuM3zkQbCfsAUqv0w5flHk-p?usp=sharing)|    512     |       256       |  200  |   ✘    |     finetune    | 65.25 | 85.98 |

To ensure the reproducibility, please refer to our training details provided in the links for our quantized models. \


To verify the performance of our quantized models on ImageNet, please use the following command:
```bash
python -u main.py \
--gpu 0 \
-e [best_model_path] \
--model resnet18_1w1a (or resnet34_1w1a)\
--dataset imagenet \
--data_path [DATA_PATH] \
-bt 256 \
```

## Comparison with SOTAs

We test our ReCU using the same ResNet-18 structure and training setttings as [ReActNet](https://github.com/liuzechun/ReActNet), and obtain higher top-1 accuracy.

| Methods | Top-1 acc | Quantized model link |
|:-------:|:---------:|:--------------------:|
|ReActNet |  65.9     | [ReActNet (Bi-Real based)](https://github.com/liuzechun/ReActNet#models) |
| ReCU    |  66.4     | [ResNet-18](https://drive.google.com/drive/folders/1vukw5yU0gLQlERmI9_dE4R4V1eg59mEI?usp=sharing)        |


To verify the performance of our quantized models with ReActNet-like structure on ImageNet, please use the following command:
```bash
cd imagenet_two-stage && python -u evaluate.py \
python -u main.py \
--gpus 0 \
-e [best_model_path] \
--model resnet18_1w1a \
--data_path [DATA_PATH] \
--dataset imagenet \
-bt 256 \
```
