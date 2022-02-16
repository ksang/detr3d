# Object DGCNN & DETR3D

This repo contains the implementations of Object DGCNN (https://arxiv.org/abs/2110.06923) and DETR3D (https://arxiv.org/abs/2110.06922). Our implementations are built on top of MMdetection3D.  

### Prerequisite

```
   CUDA_HOME=/usr/local/cuda conda env create -f environment.yml
   conda activate detr3d

   CUDA_HOME=/usr/local/cuda LLVM_CONFIG=/usr/bin/llvm-config-7 pip install -r requirements.txt
```
NOTE: mmdetection3D and mmcv_full might need install from complile, also numpy may need to be upgraded.

### Data
1. Follow the mmdet3d to process the data.

```
   python tools/create_data.py nuscenes \
         --root-path data/nuscenes/ \
         --out-dir data/nuscenes/ \
         --extra-tag nuscenes \
         --version v1.0-mini
```
### Train
1. Downloads the [pretrained backbone weights](https://drive.google.com/drive/folders/1h5bDg7Oh9hKvkFL-dRhu5-ahrEp2lRNN?usp=sharing) to pretrained/ 

2. For example, to train Object-DGCNN with pillar on 8 GPUs, please use

`tools/dist_train.sh projects/configs/obj_dgcnn/pillar.py 8`

### Evaluation using pretrained models
1. Download the weights accordingly.  

|  Backbone   | mAP | NDS | Download |
| :---------: | :----: |:----: | :------: |
|[DETR3D, ResNet101 w/ DCN](./projects/configs/detr3d/detr3d_res101_gridmask.py)|34.7|42.2|[model](https://drive.google.com/file/d/1YWX-jIS6fxG5_JKUBNVcZtsPtShdjE4O/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1uvrf42seV4XbWtir-2XjrdGUZ2Qbykid/view?usp=sharing)|
|[above, + CBGS](./projects/configs/detr3d/detr3d_res101_gridmask_cbgs.py)|34.9|43.4|[model](https://drive.google.com/file/d/1sXPFiA18K9OMh48wkk9dF1MxvBDUCj2t/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1NJNggvFGqA423usKanqbsZVE_CzF4ltT/view?usp=sharing)|
|[DETR3D, VoVNet on trainval, evaluation on test set](./projects/configs/detr3d/detr3d_vovnet_gridmask_det_final_trainval_cbgs.py)| 41.2 | 47.9 |[model](https://drive.google.com/file/d/1d5FaqoBdUH6dQC3hBKEZLcqbvWK0p9Zv/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1ONEMm_2W9MZAutjQk1UzaqRywz5PMk3p/view?usp=sharing)|

|  Backbone   | mAP | NDS | Download |
| :---------: | :----: |:----: | :------: |
|[Object DGCNN, pillar](./projects/configs/obj_dgcnn/pillar.py)|53.2|62.8|[model](https://drive.google.com/file/d/1nd6-PPgdb2b2Bi3W8XPsXPIo2aXn5SO8/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1A98dWp7SBOdMpo1fHtirwfARvpE38KOn/view?usp=sharing)|
|[Object DGCNN, voxel](./projects/configs/obj_dgcnn/voxel.py)|58.6|66.0|[model](https://drive.google.com/file/d/1zwUue39W0cAP6lrPxC1Dbq_gqWoSiJUX/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1pjRMW2ffYdtL_vOYGFcyg4xJImbT7M2p/view?usp=sharing)|


2. To test, use  
`tools/dist_test.sh projects/configs/obj_dgcnn/pillar_cosine.py /path/to/ckpt 8 --eval=bbox`

```
python tools/test.py \
   projects/configs/detr3d/detr3d_vovnet_gridmask_det_final_trainval_cbgs.py \
   pretrained/detr3d_vovnet_trainval.pth \
   --out output_result.pkl

```

If you find this repo useful for your research, please consider citing the papers

```
@inproceedings{
   obj-dgcnn,
   title={Object DGCNN: 3D Object Detection using Dynamic Graphs},
   author={Wang, Yue and Solomon, Justin M.},
   booktitle={2021 Conference on Neural Information Processing Systems ({NeurIPS})},
   year={2021}
}
```

```
@inproceedings{
   detr3d,
   title={DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries},
   author={Wang, Yue and Guizilini, Vitor and Zhang, Tianyuan and Wang, Yilun and Zhao, Hang and and Solomon, Justin M.},
   booktitle={The Conference on Robot Learning ({CoRL})},
   year={2021}
}
```
