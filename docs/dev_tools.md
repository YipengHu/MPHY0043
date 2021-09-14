# Development Tools

For tutorials that are marked to use either *Anaconda* or *Jupyter Notebook*, using [Anaconda](https://docs.anaconda.com/anaconda/) is recommended.


## I. Anaconda

### Install on different operating systems
Anaconda can be used on Linux (inc. on ChromeOS), Windows and MacOS. Please follow the [official installation instructions](https://docs.anaconda.com/anaconda/install/) for individual machines.

### Create a conda environment
After installing Anaconda, one needs to set up the environment with additional libraries. The simplest way to install the useful packages is to use the Anaconda Prompt, by creating a new `mphy0043` environmen:
```bash
conda create --name mphy0043 tensorflow pytorch notebook matplotlib 
```
In the activated `mphy0043`, install other useful libraries:
```bash
conda activate mphy0043 
conda install torchvision -c pytorch  # PyTorch library for computer vision 
pip install "monai[nibabel, gdown, ignite]"  # MONAI and its optional dependencies 
pip install av  # PyAV for reading video files
conda deactivate  # `mphy0043` is ready and can be deactivated
```

> Installation of TensorFlow and PyTorch can be OS-dependent, especially for GPU-enbaled versions. Please refer to the official guidelines if any issue on individual machines.


## II. Jupyter Notebook

### Use notebook with Anaconda
After installing Anaconda and setting up the conda environment, there are different ways to start the Jupyter Notebook in `mphy0043`, for example:

- Start [Anaconda Navigator](https://docs.anaconda.com/anaconda/navigator/index.html);
- Select the created `mphy0043` in the drop-down menu "Applications on ...";
- Launch the Jupyter Notebook in the below tab; 
- Select to open the notebook files with _.ipynb_ extension, using the browser-based interface, .

### Google Colab
The links to upload the notebooks to [Colab](https://research.google.com/colaboratory/) are provided but not technically supported in this module. Stable internet connection is required.


## III. Advanced users
For those experienced, direct use of [miniconda](https://docs.conda.io/en/latest/miniconda.html) and [Jupyter Notebook](https://jupyter.org/) should also work for those materials that are marked with Anaconda and Jupyter Notebook, respectively. However, technical support may not be available from this module.  