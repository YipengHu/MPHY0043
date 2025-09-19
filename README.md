# MPHY0043: Artificial Intelligence for Surgery and Intervention
[UCL Module](https://www.ucl.ac.uk/module-catalogue/modules/artificial-intelligence-for-surgery-and-intervention-MPHY0043) | [MPBE](https://www.ucl.ac.uk/medical-physics-biomedical-engineering/) | [UCL Moodle Page](https://moodle.ucl.ac.uk/)
>Term 1 (Autumn)


## Contacts
|Name           | Email                         | Role        |
|---------------|-------------------------------|-------------|
|Yipeng Hu      | <yipeng.hu@ucl.ac.uk>         | Module Lead |
|Athena Reissis | <athena.reissis.21@ucl.ac.uk> | Tutor       |
|Weixi Yi       | <weixi.yi.22@ucl.ac.uk>       | Tutor       |


## 1. Programming and development

### Python, numerical computing 
All practical tutorials, group work and coursework projects in this module are based on Python, with a number of common libraries, including NumPy, SciPy and Matplotlib. For a refresher or relevant materials in medical image analysis, please have a look at the [UCL Module MPHY0030 - Programming Foundations in Medical Image Analysis](https://github.com/YipengHu/MPHY0030).

### Machine learning, deep learning
This module uses two deep learning libraries, [TensorFlow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/).

Guide and tutorial materials for the deep learning libraries are widely available, for example, from the [UCL Module COMP0197 - Applied Deep Learning](https://github.com/YipengHu/COMP0197), with relevant materials designed for medical applications in the [UCL Module MPHY0041 - Machine Learning in Medical Imaging](https://github.com/YipengHu/MPHY0041).  

[MONAI](https://monai.io/) is also used, with many dedicated deep learning functionalities designed for medical applications.

### Development tools
[Jupyter Notebook](https://jupyter.org/) and [Anaconda/Conda](https://www.anaconda.com/) are used in certain tutorials and may be helpful for the assessed group work and coursework. Follow the [Development Tools](docs/dev_tools.md) to set them up on your machine.  

Although not required, it is encouraged to use [Git](https://git-scm.com/) with this repository. Tutorials for its basic uses are also widely available, e.g. [Work with Git](https://github.com/YipengHu/MPHY0030/blob/main/docs/dev_env_git.md).


## 2. Tutorials
| tools | envs | learning type | applications | remarks |
>Go to individual tutorial sub-directories and read the _readme.md_ file to get started. 

### Surgical Data Regression
[Tutorial directory](tutorials/regression)  
_Keywords_: Classical machine learning, linear algebra, optimisation, NumPy, TensorFlow and PyTorch  
_Devlopement tools_: Jupyter Notebook (via Anaconda) 

### Surgical Gesture and Skill
[Tutorial directory](tutorials/gesture)  
_Keywords_: supervised classification, PyTorch, 3D CNN, JIGSAWS  
_Devlopement tools_: Anaconda with PyTorch

### 3D Medical Image Segmentation
[Tutorial directory](tutorials/segmentation)  
_Keywords_: PyTorch, segmentation, MONAI U-Net, clinical imaging data  
_Devlopement tools_: Jupyter Notebook (via Anaconda)  

### Image Registration
[Tutorial directory](tutorials/registration)  
_Keywords_: PyTorch, Unsupervised registration, MONAI, MedNist dataset  
_Devlopement tools_: Jupyter Notebook (via Anaconda)  

### Intraoperative Motion Classification
[Tutorial directory](tutorials/pointset)  
_Keywords_: TensorFlow, Keras, PointNet, simulated dataset  
_Devlopement tools_: Anaconda with TensorFlow  

### Vision and Workflow
[Tutorial directory](tutorials/scopic)  
_Keywords_: TensorFlow Keras, Supervised classification, "off-the-shelf" networks, [endoscopic video data](https://www.synapse.org/#!Synapse:syn25147789/wiki/608848)  
_Devlopement tools_: Anaconda with TensorFlow 


## 3. Reading list
A collection of books and research papers, applying artificial intelligence to surgery and intervention, is provided in the [Reading List](docs/reading.md).
