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
If you have problem with the proxy and you are log in with ssh, try to log on with a desktop session on the computer or try to set the proxy in the file `.condarc`.

Then start the jupyter lab server.

```
cd ~/repo
jupyter lab
```

# Running on a remote server

If your computer does not have a good GPU, you can install DLC and autopipy on the computer with the GPU. If you start the jupyter lab server, you should be able to access it from a different computer using its IP address.

```
http://a230-pc73:8888/lab
```
