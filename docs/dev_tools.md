# Development Tools

For tutorials that are marked to use either *Anaconda* or *Jupyter Notebook*, using [Anaconda](https://docs.anaconda.com/anaconda/) is recommended.


## I. Anaconda

### i. Install on different operating systems
Anaconda can be used on Linux (inc. on ChromeOS), Windows and MacOS. Please follow the [official installation instructions](https://docs.anaconda.com/anaconda/install/) for individual machines.

### ii. Create a conda environment
After installing Anaconda, one needs to set up the environment with additional libraries. The simplest way to install the useful packages is to use the Anaconda Prompt for Windows, or in a terminal window for macOS or Linux. Create a new `mphy0043` environment:
```bash
conda create --name mphy0043 tensorflow==2.9  
```
In the activated `mphy0043`, install other useful libraries:
```bash
conda activate mphy0043 
pip install notebook matplotlib av "monai[nibabel, gdown, ignite]"  # monai includes PyTorch
```

> Installation of TensorFlow and PyTorch can be OS-dependent, especially for GPU-enbaled versions. Please refer to their official documentations if any issue on individual machines.

### iii. Run Python scripts
Some tutorials are written in Python scripts, which can be run at the Anaconda Prompt for Windows or in a terminal window for macOS or Linux with the activated `mphy0043`, by typing the command line commands, e.g.:
```bash
python video_classification.py
```


## II. Jupyter Notebook

### i. Use notebook with Anaconda
After installing Anaconda and setting up the conda environment, there are different ways to start the Jupyter Notebook in `mphy0043`, for example:

- Start [Anaconda Navigator](https://docs.anaconda.com/anaconda/navigator/index.html);
- Select the created `mphy0043` in the drop-down menu "Applications on ...";
- Launch the Jupyter Notebook in the below tab; 
- Select to open the notebook files with _.ipynb_ extension, using the browser-based interface.

### ii. Google Colab
The links to upload the notebooks to [Colab](https://research.google.com/colaboratory/) may be provided but not technically supported in this module. Stable internet connection is required. 


## III. Advanced users
For those experienced, direct use of [miniconda](https://docs.conda.io/en/latest/miniconda.html) and [Jupyter Notebook](https://jupyter.org/) should also work for those materials that are marked with Anaconda and Jupyter Notebook, respectively. However, technical support may not be available from this module.  
