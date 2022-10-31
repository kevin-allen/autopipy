# Easy installation

The simplest way to install the autopipy package is to first install a deeplabcut conda environment and then install autopipy within it.

## Install anaconda

## Install the DeepLabCut environment

You can find the steps to install [deeplabcut on their github repository](https://github.com/DeepLabCut/DeepLabCut).

I usually remove the [gui] from the DEEPLABCUT.yaml file and added seaborn.


## Install autopipy in your DeepLabCut environment
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

If you have problem with the proxy and you are log in with ssh, it is probably because the HTTP_PROXY and HTTPS_PROXY environment variables are not set in your shell. You have a few options: 

*  log on with a desktop session on the computer
*  Set the proxy in the file `.condarc` or try to set the HTTP_PROXY and HTTPS_PROXY manually
*  Set the environment variables manually in a terminal.

For instance, you could run this in the terminal
```
export HTTP_PROXY=http://www-int2.inet.dkfz-heidelberg.de:80
export HTTPS_PROXY=http://www-int2.inet.dkfz-heidelberg.de:80
export http_proxy=http://www-int2.inet.dkfz-heidelberg.de:80
export https_proxy=http://www-int2.inet.dkfz-heidelberg.de:80
```


Then start the jupyter lab server.

```
cd ~/repo
jupyter lab
```
 
 If you have problem loading dlc, try with dlc light. You will need to set a variable in your shell environment.
 
 ```
 os.environ['DLClight'] = 'True'
 ```

