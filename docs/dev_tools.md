# Development Tools

For tutorials that are marked to use either *Anaconda* or *Jupyter Notebook*, using [Anaconda](https://docs.anaconda.com/anaconda/) is recommended.


## Anaconda

### Install on different operating systems
Anaconda can be used on Linux (inc. on ChromeOS), Windows and MacOS. Please follow the [official installation instructions](https://docs.anaconda.com/anaconda/install/) for individual machines.

### Set up a conda environment
After installing Anaconda, one needs to set up the environment with additional libraries. The simplest way to install the useful packages is to use the Anaconda Prompt, by creating a new `mphy0043` environment:
    ```bash
    conda create --name mphy0043 pytorch tensorflow notebook
    ```
    Activate `mphy0043`: 
    ```bash
    conda activate mphy0043
    ```
    Install other libraries:
    ```bash
    pip install monai
    ```
    The `mphy0043` now is ready for use and deactivate and exit.
    ```bash
    conda deactivate
    ```

## Jupyter Notebook

### Use notebook with Anaconda
After installing Anaconda and setting up the conda environment, there are different ways to start the Jupyter Notebook in `mphy0043`: 

- Start [Anaconda Navigator](https://docs.anaconda.com/anaconda/navigator/index.html);
- Select the created `mphy0043` in the drop-down menu "Applications on ...";
- Launch the Jupyter Notebook in the below tab; 
- Using the started browser-based interface, select the notebook files with _.ipynb_ extension.

### Google Colab
The links to upload the notebooks to [Colab](https://research.google.com/colaboratory/) are provided but not technically supported in this module. Stable internet connection is required.


## Adavanced users
For those experienced, direct use of [miniconda](https://docs.conda.io/en/latest/miniconda.html) and [Jupyter Notebook](https://jupyter.org/) should also work for those materials that are marked with Anaconda and Jupyter Notebook, respectively. However, technical support may not be available from this module.  