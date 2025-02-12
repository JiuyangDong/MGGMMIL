# MGGMMIL
The official implementation of "Multi-Granularity Graph-Mamba Multi-Instance Learning for Unlabeled Autoffuorescence Whole-Slide Image Classiffcation"

## training & validation & testing
```shell
python train_and_test.py --pretrain ResNet18_ImageNet --baseline_model AFMIL --data_dir your_data_dir \
    --wsi_type AF --gpu_id 0 --L 384 --lr 3e-5 --layer 4 --KNN 4+8+12 --mil_head GABMIL
```
