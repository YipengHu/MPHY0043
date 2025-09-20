# Intraoperative Motion Classification

This tutorial is adapted from [Point cloud classification with PointNet](https://keras.io/examples/vision/pointnet/) with TensorFlow and Keras.

Here, we use Intraoperative motion data from an ultrasound-guided prostate intervention, for a motion detection problem. The clinical background can be found in the [MPHY0030 SSM tutorial](https://github.com/YipengHu/MPHY0030/tree/main/tutorials/statistical_motion_model).

## Environment
Use the [mphy0043-tf env](../../docs/dev_tools.md):
```bash
micromamba activate mphy0043-tf
```

## Data
Download the simulated motion data by running: 
```bash
python download_data.py
```

## Run
The run the motion classification script:
```bash
python motion_classification.py
```


<img src="../../docs/media/motion.jpg" alt="alt text"/>
