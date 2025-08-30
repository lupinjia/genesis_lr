from setuptools import find_packages
from distutils.core import setup

setup(
    name='genesis_lr',
    version='0.1.0',
    author='Yasen Jia',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='jason_1120202397@163.com',
    description='Genesis environments for Legged Robots',
    install_requires=['genesis-world',
                      'torch',
                      'matplotlib',
                      'torchvision',
                      'numpy',
                      'tensorboard',
                      'xlsxwriter',
                      'pandas',
                      'wandb>=0.21.2']
)