# Query6DoF: Learning Sparse Queries as Implicit Shape Prior for Category-Level 6DoF Pose Estimation
This is the PyTorch implemention of ICCV'23 paper Query6DoF: Learning Sparse Queries as Implicit Shape Prior for Category-Level 6DoF Pose Estimation

# Requirements
- Linux (tested on Ubuntu 16.04)
- Python 3.8
- CUDA 11.1
- PyTorch 1.10.2
  
# Installation
~~~
conda create -n query6dof python=3.8

conda activate query6dof

pip install torch==1.10.2+cu111 -f  https://download.pytorch.org/whl/cu111/torch_stable.html

pip install opencv-python mmengine numpy tqdm

cd Pointnet2/pointnet2

python setup.py install
~~~

# Dataset
Download camera_train, camera_eval, real_test, real_train, ground-truth annotations and mesh models provided by [NOCS](https://github.com/hughw19/NOCS_CVPR2019).
Then process these files following [SPD](https://github.com/mentian/object-deformnet). And download segmentation results from Mask R-CNN, and predictions of NOCS from [SPD](https://github.com/mentian/object-deformnet).
The dataset is organized as follows:
~~~

── data
    ├── CAMERA
    ├── gts
    ├── obj_models
    ├── Real
    └── results
            └── mrcnn_results   
── results
      └── nocs_results
~~~

# Evaluation
Please download our pretrain model [here](https://drive.google.com/file/d/11DKVV6NCgecKoe6Pu9OIXWyiROXhuW3J/view?usp=drive_link) and put it in 'runs/CAMERA+Real/run/model' dictionory. 

Then, you can make an evaluation for REAL275 using following command.
~~~
python tools/valid.py --cfg config/run_eval_real.py --gpus 0
~~~
Then, you can make an evaluation for CAMERA25 using following command.
~~~
python tools/valid.py --cfg config/run_eval_camera.py --gpus 0
~~~

# Train
'tools/train.py' is the main file for training. You can train using the following command.
~~~
python tools/train.py --cfg config/run.py --gpus 0,1,2,3
~~~
You can modity the training config.

# Ackownledgment
The dataset is provided by [NOCS](https://github.com/hughw19/NOCS_CVPR2019). Our code is developed based on [Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch) and [SPD](https://github.com/mentian/object-deformnet)