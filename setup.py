import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autopipy-kevinallen", # Replace with your own username
    version="0.0.3",
    author="Kevin Allen",
    author_email="allen@uni-heidelberg.de",
    description="A python package to analyse data from the AutoPI task",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kevin-allen/autopipy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='>=3.6',
)
