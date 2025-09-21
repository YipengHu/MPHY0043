# Development Tools

## Micromamba

### i. Install on different operating systems
Micromamba can be used on Linux (inc. Crostini on ChromeOS, WSL on Windows), MacOS and Windows. Please follow the [official installation instructions](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) for individual machines.

[Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main), [Mamba](https://mamba.readthedocs.io/) and potentially other package management tools can also be used to set up the following development environments.

### ii. Create development environments
After installing Micromamba, one needs to set up the environment with additional libraries. The simplest way to install the useful packages is to use the Anaconda Prompt for Windows, or in a terminal window for macOS or Linux. 

#### Create a `mphy0043-pt` environment for PyTorch and MONAI:
```bash
micromamba create --name mphy0043-pt python=3.11
micromamba activate mphy0043-pt 
pip install "monai[nibabel, gdown, ignite]"  # monai includes PyTorcch
pip install notebook matplotlib av pillow
```
Deactivate the environment before switching/creating a new one:
```bash
micromamba deactivate 
```

#### Create a `mphy0043-tf` environment for TensorFlow and Keras:
```bash
micromamba create -n mphy0043-tf python=3.11
micromamba activate mphy0043-tf
pip install tensorflow==2.17.0  # >=2.16.1 for Keras 3.0 
pip install notebook matplotlib 
```

> Installation of TensorFlow and PyTorch can be OS-dependent, especially for GPU-enbaled versions. Please refer to their official documentations for individual machines.

### iii. Run Python scripts or use Jupyter Notebook
With the created development environment, Python scripts or Jupyter Notebook can be started, e.g.:
```bash
micromamba activate mphy0043-pt
jupyter notebook
```
Select to open the notebook files with _.ipynb_ extension, using the browser-based interface.


## Microsoft Visual Studio Code
VSCode is the [supported IDE](https://github.com/YipengHu/MPHY0030/blob/main/docs/dev_env_python.md) in this module. [Get started using the official guide](https://code.visualstudio.com/docs).


## Google Colab
The links to upload the notebooks to [Colab](https://research.google.com/colaboratory/) may be provided but not technically supported in this module. Stable internet connection is required. 
