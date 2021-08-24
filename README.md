# MPHY0043: Artificial Intelligence for Surgery and Intervention
UCL Module | [MPBE](https://www.ucl.ac.uk/medical-physics-biomedical-engineering/) | UCL Moodle Page
>Term 1 (Autumn), Academic Year 2021-22 


## Contacts
|Name                 | Email                       | Role                    |
|---------------------|-----------------------------|-------------------------|
|Yipeng Hu            | <yipeng.hu@ucl.ac.uk>       | Module Lead             |
|Zachary Baum         | <zachary.baum.19@ucl.ac.uk> | Head Teaching Assistant |
|XXXX XXX             | <bxx.xxxxx@ucl.ac.uk>       | Teaching Assistant      |


## 1. Programming and development

### Python, numerical computing 
All practical tutorials, group work and coursework projects in this module are based on Python, with a number of common libraries, including NumPy, SciPy and Matplotlib. For a refresher or relevant materials in medical image analysis, please have a look at the [UCL Module MPHY0030 - Programming Foundations in Medical Image Analysis](https://weisslab.cs.ucl.ac.uk/YipengHu/mphy0030).

### Machine learning, deep learning
This module uses two deep learning libraries, [TensorFlow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/).

Guide and tutorial materials for the deep learning libraries are widely available, for example, from the [UCL Module COMP0090 - Introduction to Deep Learning](https://github.com/YipengHu/COMP0090), with relevant materials designed for medical applications in the [UCL Module MPHY0041 - Machine Learning in Medical Imaging](https://weisslab.cs.ucl.ac.uk/YipengHu/mphy0041).  

[MONAI](https://monai.io/) is also used, with many dedicated deep learning functionalities designed for medical applications.

### Development tools
[Jupyter Notebook](https://jupyter.org/) and [Anaconda/Conda](https://www.anaconda.com/products/individual) are frequently used in most tutorials and will be required for the assessed group work and coursework. Follow the [instructions]() to set them up on your machine.  

Although not required, it is ecouraged to use [Git](https://git-scm.com/) with this repository. Tutorials for its basic uses are also widely available, e.g. [xxx]().


## 2. Tutorials
| tools | envs | learning type | applications | remarks |
>Go to individual tutorial directory and read the _readme.md_ file to get started. 

### Surgical Data Regression
[Tutorial directory](tutorials/regression)  
Classical machine learning methods using linear algebra and optimisation functions in NumPy, TensorFlow and PyTorch  
Devlopement tools: Jupyter Notebook (via Anaconda/Colab) 

### Surgical Gesture and Skill Assessment
[Tutorial directory](tutorials/gesture)  
TensorFlow Keras, Supervised classification, "off-the-shelf" networks, JIGSAWS  
Devlopement tools: Anaconda with TensorFlow

### 3D Medical Image Segmentation
[Tutorial directory](tutorials/segmentation)  
PyTorch, Segmentation, MONAI U-Net, (optionally nnUNet), clinical imaging data  
Devlopement tools: Anaconda with Pytorch 

### Image Registration
[Tutorial directory](tutorials/registration)  
PyTorch, Unsupervised registration, MONAI, MedNist dataset  
Devlopement tools: Jupyter Notebook (via Anaconda/Colab) 

### Workflow Recognition
[Tutorial directory](tutorials/workflow)  
TensorFlow Keras, Supervised classification, "off-the-shelf" networks, [endoscopic video data](https://www.synapse.org/#!Synapse:syn25147789/wiki/608848)  
Devlopement tools: Anaconda with TensorFlow


## 3. Reading list
A collection of books and research papers, applying artificial intelligence to surgery and intervention, is provided in the [Reading List](docs/reading.md).
