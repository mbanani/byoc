# Setup 

This repository makes use of several external libraries. 
We highly recommend installing them within a virtual environment such as Anaconda. 

The script below will help you set up the environment; the `--yes` flag allows conda to install
without requesting your input for each package.

```bash 
conda create --name byoc python=3.8 --yes
conda activate byoc

# install pytorch
conda install pytorch=1.9.0 torchvision=0.10.0 cudatoolkit=10.2 -c pytorch --yes

# pytorch3d 0.5.0
conda install -c fvcore -c iopath -c conda-forge fvcore iopath --yes
conda install -c bottler nvidiacub --yes
python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.5.0"

# MinkowskiEngine 0.5.4
# ensure GCC > 7.4
conda install openblas-devel -c anaconda --yes
python -m pip install ninja
python -m pip install -U MinkowskiEngine --install-option="--blas=openblas" -v --no-deps

# Set of other usefull packages 
python -m pip install install hydra-core pytorch-lightning
python -m pip install install opencv-python open3d 

# The following is not essential to run the code, but good if you want to contribute
# or just keep clean repositories. 
# .pre-commit-config.yaml file is in the repo.
python -m pip install pre-commit
pre-commit install 
```


