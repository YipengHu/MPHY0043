# 3D Medical Image Segmentation

This is a tutorial adapted from [Spleen 3D segmentation with MONAI](https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/spleen_segmentation_3d.ipynb).

This tutorial shows how to integrate MONAI into an existing PyTorch medical DL program.

And easily use below features:

- Transforms for dictionary format data.
- Load Nifti image with metadata.
- Add channel dim to the data if no channel dimension.
- Scale medical image intensity with expected range.
- Crop out a batch of balanced images based on positive / negative label ratio.
- Cache IO and transforms to accelerate training and validation.
- 3D UNet model, Dice loss function, Mean Dice metric for 3D segmentation task.
- Sliding window inference method.
- Deterministic training for reproducibility.
- The Spleen dataset can be downloaded from http://medicaldecathlon.com/.

Target: Spleen
Modality: CT
Size: 61 3D volumes (41 Training + 20 Testing)
Source: Memorial Sloan Kettering Cancer Center
Challenge: Large ranging foreground size


Start the tutorial within the [mphy0043-pt env](../../docs/dev_tools.md):
```bash
micromamba activate mphy0043-pt
jupyter notebook
```
Then, select the `spleen_segmentation_3d.ipynb` notebook file.


<img src="../../docs/media/spleen.png" alt="alt text"/>
