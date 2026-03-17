# FeatherFace_MM

## install
1. git clone https://github.com/dohun-mat/FeatherFace_MM

## data
1. We also provide the organized dataset we used as in the above directory structure.
Link: from [google cloud](https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS) or [baidu cloud](https://pan.baidu.com/s/1jIp9t30oYivrAvrgUgIoLQ) Password: ruck

2. Organise the dataset directory as follows:
```Shell
  ./data/widerface/
    train/
      images/
      label.txt
    val/
      images/
      wider_val.txt
```
ps: wider_val.txt only include val file names but not label information.

## train
1. Before training, you can check network configuration (e.g. batch_size, min_sizes and steps etc..) in
   ```data/config.py and train.py```

2. Train the model using WIDER FACE:
  ```Shell
   CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node=1 train.py --network eresnet
  ```


## 평가(두가지 방법 중 하나만)
1. Generate txt file
```Shell
원본 이미지만 사용하고 싶을때(빠른 대신 성능 낮음)
CUDA_VISIBLE_DEVICES=1 python test_widerface.py --trained_model ./weights/eresnet_Final.pth --network eresnet

multi-전략을 사용하고 싶을때(느린 대신 성능 높음)
CUDA_VISIBLE_DEVICES=1 python test_widerface_multi_scale.py --trained_model ./weights/eresnet_Final.pth --network eresnet --test_scales 0.5 1.0 1.5 2.0 --do_flip

```
2. Evaluate txt results. Demo come from [Here](https://github.com/wondervictor/WiderFace-Evaluation)  
```Shell
cd ./widerface_evaluate
python evaluation.py -p ./widerface_txt -g ./eval_tools/ground_truth
```
3. You can also use widerface official Matlab evaluate demo in [Here](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html)  


