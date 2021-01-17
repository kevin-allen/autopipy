# Development of python package


* 
* [Example of python package] (https://github.com/pypa/sampleproject)

## Upload to testpypi

Set a new version for the package in `setup.py` as you can only use a version once.

```
cd ~/repo/autopipy
rm -fr autopipy_kevinallen.egg-info build dist
python3 setup.py sdist bdist_wheel
python3 -m twine upload --repository testpypi dist/* --verbose
pip install -i https://test.pypi.org/simple/ autopipy-kevinallen==0.0.2
```
