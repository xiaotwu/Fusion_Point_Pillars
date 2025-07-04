# Fusion Point Pillars
Fusion Point Pillars is an enhanced version of the classic PointPillars framework, incorporating multi-modal or multi-level feature fusion to significantly improve perception and representation in 3D object detection tasks.

### Datasets
Please download the datasets from the following links and place them in the `datasets/kitti` directory:

- calib
- image_2
- label_2
- velodyne

`Notice: Only training sets are required for training. The test sets are not needed. All the datasets and tookits can be downloaded from [here](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).`

### Project Structure
The project structure is organized as follows:
```
Fusion_Point_Pillars/
├── datasets/kitti/
│   ├── calib/
│   ├── image_2/
│   ├── label_2/
│   └── velodyne/
|   └── devkit_object/  # Optional, for validation and testing
|       |── data/object/label_2
|       |── results
|       └── ...
├── models/
|   ├── backbone.py
│   ├── fusion_pointpillars.py
│   └── image_encoder.py
|   └── pointpillars.py
|   └── ssd_head.py
├── utils/
│   ├── anchor_generator.py
│   ├── box_utils.py
│   ├── calibration.py
│   ├── kitti_dataset.py
│   ├── lidar_utils.py
│   ├── projection.py
│   └── target_assigner.py
|── config.py
├── labels.py
├── train.py
├── requirements.txt
└── test.py
└── ...
```

### How to Run
To run the training process, execute the following command in your terminal:
```bash
# Clone the repository if you haven't already
git clone https://github.com/xiaotwu/Fusion_Point_Pillars.git

# Ensure you are in the Fusion_Point_Pillars directory
cd Fusion_Point_Pillars

# Create a conda environment (optional but recommended)
conda create -n fpp python=3.12

# Activate the conda environment
conda activate fpp

# Install the required packages
pip install -U pip
pip install -r requirements.txt

# Run the training script
python train.py
```

`Notice: This project is trained on a Nvidia RTX 5070 Ti GPU. The newer GPU architure and CUDA 12.9 may require Pytorch Preview (Nightly) version to get the full support. If you are using an older GPU, please use the stable version of Pytorch.`

### Validation and Testing
We decide to use Object Development Kit to validate and test the model. The tookit can be downloaded from [here](https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip) and placed in project root directory.