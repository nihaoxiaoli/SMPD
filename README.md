# SMPD 
Stabilizing Multispectral Pedestrian Detection with Evidential Hybrid Fusion

# Comparison Results
You can directly click on the [link](https://pan.baidu.com/s/1KwlIrZLVU3Mp_2VOPJRMBg?pwd=1ggv) to download our results for drawing the FPPI-MR curve.

# Demo
## 1. Preparation
This code is based on the repository of [faster rcnn](https://github.com/jwyang/faster-rcnn.pytorch/blob/pytorch-1.0/README.md) (pytorch-1.0 branch). You can refer to the environment of that repository. 

or 

You can follow the steps below to configure the environment.

### Prerequisites
* Pytorch 1.0 or higher
* CUDA 8.0 or higher

### Compilation

Compile the cuda dependencies using following simple commands:

```
cd lib
python setup.py build develop
```

It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Align and ROI_Crop. 

**As pointed out in this [issue](https://github.com/jwyang/faster-rcnn.pytorch/issues/16), if you encounter some error during the compilation, you might miss to export the CUDA paths to your environment.**

## 2. Train
### Pretrained Model

We used pretrained models VGG in our experiments. You can download model from:

* VGG16: [Baidu Netdisk](https://pan.baidu.com/s/1LRRVp4XrzryxDV0_libnjA?pwd=hi8z)

Download it and put it into the data/pretrained_model/.

### Data Preparation
Download data Kaist, and add the path of data to the variable ```self._pic_path``` in the file ```kaist.py```.

### Train Model
To train the SMPD with vgg16 on kaist, simply run:
```
CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
                   --dataset kaist --net vgg16 \
                   --lr 0.005 --lr_decay_step 2 \
                   --epochs 4 --cuda
```

The model parameters of each epoch will be automatically saved.

## 3. Evaluation
Add the path of model parameters to the variable ```load_name``` in the file ```test_net.py```.

It can be the model that is automatically saved in the path ```models/vgg16/kaist``` after training. (The performance of the third epoch is usually the best. Model performance often has some randomness, which may be better than the results reported in the paper or slightly inferior, but the overall error range is not too large.)

or 

you can directly download the [model](https://pan.baidu.com/s/1w7AJ7AjPlGeaGUsui0FqaA?pwd=068v) used in our paper.

To test on kaist, simply run:
```
python test_net.py --dataset kaist --net vgg16 --cuda --reasonable
```

# Acknowledgements
This pipeline is largely built on [faster-rcnn](https://github.com/jwyang/faster-rcnn.pytorch/blob/pytorch-1.0/README.md) and [AR-CNN
](https://github.com/luzhang16/AR-CNN). Thank for these great works. Meanwhile, if you encounter any problems during the configuration process, you can check [issue](https://github.com/jwyang/faster-rcnn.pytorch/issues) to see if you can find answers there.