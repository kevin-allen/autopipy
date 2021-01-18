# Development of python package


* [Example of python package] (https://github.com/pypa/sampleproject)


## Set up a virtual environment from scratch for development

We will use venv to start with. We need to install it on the system.

```
sudo apt-get install python3-venv
```

```
cd ~
mkdir python_virtual_environments
cd python_virtual_environments
python3 -m venv tutorial-env
```

To activate the environment, use source

```
source ~/python_virtual_environments/autopi-env/bin/activate
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
import detectArenaCoordinates
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
