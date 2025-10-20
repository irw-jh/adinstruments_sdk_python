# -*- coding: utf-8 -*-

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="adi-reader",
    version="0.0.14",
    author="Jim Hokanson, irw-jh",
    author_email="",
    description="Reading LabChart recorded data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/irw-jh/adinstruments_sdk_python",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'cffi'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires='~=3.6',
    include_package_data=True,
)