# Easy installation

The simplest way to install the autopipy package is to first install a deeplabcut conda environment and then install autopipy within it.

You can find the steps to install deeplabcut on their [github repository](https://github.com/DeepLabCut/DeepLabCut).

If you are not doing inference with deeplabcut, you don't need tensorflow with GPU support.

```{python}
conda activate DLC-GPU 
cd ~/repo
git clone https://github.com/kevin-allen/autopipy.git
pip install -e ~/repo/autopipy
```

To run code, you can use `jupyter notebook` or `jupyter lab`. I prefer using `jupyter lab` and you will need to add it to your DLC environment.

```
conda install -c conda-forge jupyterlab
```

Then start the jupyter lab server.

```
cd ~/repo
jupyter lab
```
