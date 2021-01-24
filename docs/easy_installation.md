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

First ssh into the computer with the GPU, activate the right conda environment and launch jupyter lab.

```
ssh a230-pc73 -X 
conda activate DLC-GPU
cd ~/repo
jupyter lab --no-browser
```

Open a second terminal on your local computer.
This command forwards the remote port 8888 onto our local machine’s port 8888.

``` 
ssh -N -L 8888:localhost:8888 kevin@a230-pc73
```

Then, point your browser to 127.0.0.1:8888. You will be asked for a tocken or password.
Use the token that was printed when you started jupyter lab.
