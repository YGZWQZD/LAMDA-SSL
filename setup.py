# -*- coding: utf-8 -*-

"""
Author     : Lin-Han Jia, Lan-Zhe Guo and Zhi Zhou
Description: LAMDA-SSL is an useful toolkit for semi-supervised learning.
"""

import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='LAMDA-SSL',
    version="1.0.2",
    author="Lin-Han Jia, Lan-Zhe Guo, Zhi Zhou, Yu-Feng Li",
    license='MIT',
    author_email="1129198222@qq.com",
    description="LAMDA-SSL is an useful toolkit for semi-supervised learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YGZWQZD/LAMDA-SSL",
    packages=setuptools.find_packages(exclude=["Test","Unused"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=['Semi-Supervised Learning', 'Machine Learning', 'Deep Learning', 'Toolkit'],
    install_requires=['scikit-learn','torchtext',
                      'torchvision','torch-geometric','Pillow',
                      'numpy','scipy','pandas','matplotlib'],
    python_requires='>=3.7',
)