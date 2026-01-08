from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='torchpso',
    version='0.0.1',    
    description='A fast implementation of Particle Swarm Optimization & variants using PyTorch, with a framework for inverse problems',
    url='https://github.com/GiovanniBillo/torchswarm-FSTPSO',
    author='Giovanni Billo',
    license='MIT',
    install_requires=['torch'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.6",
)
