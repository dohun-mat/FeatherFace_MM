# FeatherFace_MM

## install
1. git clone https://github.com/dohun-mat/FeatherFace

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
download the pre-trained weights file MobilenetV1X0.25_pretrain.tar for the backbone from this link. [google cloud](https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1) 
```Shell
  ./weights/
      mobilenetV1X0.25_pretrain.tar
```
1. Before training, you can check network configuration (e.g. batch_size, min_sizes and steps etc..) in
   ```data/config.py and train.py```

2. Train the model using WIDER FACE:
  ```Shell
   CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py --network mobile0.25
  ```


## 평가
1. Generate txt file
```Shell
python test_widerface.py --trained_model ./weights/mobilenet0.25_Final.pth --network mobile0.25 --origin_size True
```
2. Evaluate txt results. Demo come from [Here](https://github.com/wondervictor/WiderFace-Evaluation)  
```Shell
cd ./widerface_evaluate
python evaluation.py -p ./widerface_txt -g ./eval_tools/ground_truth
```
3. You can also use widerface official Matlab evaluate demo in [Here](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html)  


