# Development of python package

* [Example of python package] (https://github.com/pypa/sampleproject)

Most of the work needed to get going is to prepare your computer and virtuel environment for using deeplabcut and tensorflow. 
This will not be needed when using the package with anaconda but I wanted a non-anaconda environment working.

## Install the right python version

One principal consideration was to be able to use deeplabcut with tensorflow-GPU. 

The version of tensorflow available from pypi works with tensorflow 1.15, which is avaialbe with python3.7. So that chose the python version for me: **3.7**

```
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.7
sudo apt-get install python3.7-dev
python3.7 --version
sudo apt install python3.7-venv
```

## Set up a virtual environment from scratch for development

We will use `venv` to create the virtual environments. You might have to install it on your system.

```
sudo apt-get install python3.7-venv
```

```
cd ~
mkdir python_virtual_environments
cd python_virtual_environments
python3.7 -m venv autopi37
```

**autopi37** is my environment to use autopipy when developing the package.

To activate the environment, use source

```
source ~/python_virtual_environments/autopi37/bin/activate
```

## Install a few needed packages

```
python3.7 -m pip install --upgrade pip
pip install deeplabcut
pip install tensorflow-GPU==1.15
```

## Installing NVIDIA CUDA to use the GPU

Be **carefull**, you can do some damage to your Ubuntu installation.

If you want to use tensorflow-gpu, you need to install the cuda libraries on your computer.

You need a version of cuda that is compatible with the version of tensorflow you are running.

For tensoflow==1.15, I had to use cuda 10.1

I followed the instructions (here)[https://medium.com/@stephengregory_69986/installing-cuda-10-1-on-ubuntu-20-04-e562a5e724a0].

I had to force an overwrite when installing to get rid of an error.

```
sudo apt-get -o Dpkg::Options::="--force-overwrite" install --fix-broken
```

```
nvcc --version
```

Add the following to your ~/.bashrc file.

```
if [ -d "/usr/local/cuda-10.1/bin/" ]; then
    export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi
```

**Reboot your computer**


## Check if you can use tensorflow and your GPU

```
source ~/python_virtual_environments/autopi37/bin/activate
python -c "import tensorflow as tf; tf.config.experimental.list_physical_devices('GPU');"
```
If you get error messages like, you will have to install the missing packages.

```
2021-01-18 17:06:34.023491: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcudart.so.10.0'; dlerror: libcudart.so.10.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda-10.1/lib64:/home/kevin/catkin_ws/devel/lib:/opt/ros/noetic/lib
```

I was missing the following packages on my Ubuntu 20.04 installation.

```
sudo apt-get install cuda-cudart-10-0
sudo apt-get install cuda-cublas-10-0
sudo apt-get install cuda-cufft-10-0
sudo apt-get install cuda-curand-10-0
sudo apt-get install cuda-cusolver-10-0
sudo apt-get install cuda-cusparse-10-0
```


## Installing wxPython

I could not install wxPython from a wheel in the pypi repository. I had to download the source, build the wheel and install it. 

I presume that the wheel is probably python-version-specific, so build it from an virtual environment with the same python version as you intend to use.

You can find more information [here](https://wxpython.org/blog/2017-08-17-builds-for-linux-with-pip/index.html).

```
deactivate
python3.7 -m venv xwPy_build
cd xwPy_build/
source bin/activate
pip install -U pip
pip install -U six wheel setuptools
pip install python-config
pip download wxPython
pip wheel wxPython-4.1.1.tar.gz
```

Now you can install the wheel in any environment with the same python version.

```
source ~/python_virtual_environments/autopi37/bin/activate
pip install wxPython-4.1.1-cp37-cp37m-linux_x86_64.whl
python -c "import wx; a=wx.App(); wx.Frame(None,title='hello world').Show(); a.MainLoop();"
```

## Saving the environment requirements

You can save the requirement in the repository.

```
pip freeze > ~/repo/autopipy/requirements.txt
```

If you download the repository on a new computer, you can install the requirements with pip.

```
python -m pip install -r ~/repo/autopipy/requirements.txt
```

## Manually add your package to the PYTHONPATH variable

If you are developing a package, you probably want to include the directory into the shell variable PYTHONPATH.
You can set this in your `~/.bashrc`

```
export PYTHONPATH="${PYTHONPATH}:/home/kevin/repo/autopipy/src/autopipy/"
```

You can then import a module with

```
python -c "import detectArenaCoordinates"
```

## Upload to testpypi

Set a new version for the package in `setup.py` as you can only use a version once.

```
cd ~/repo/autopipy
rm -fr autopipy_kevinallen.egg-info build dist
python3 setup.py sdist bdist_wheel
python3 -m twine upload --repository testpypi dist/* --verbose
```

To install autopipy

```
pip install -i https://test.pypi.org/simple/ autopipy-kevinallen==0.0.2
```