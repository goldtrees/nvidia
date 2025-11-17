# System Requirements
* OS : Ubuntu 24.04
* GPU : NVIDIA® GeForce RTX™ 4070 Laptop GPU
* CUDA Toolkit : 12.5
* cuDNN : 9.3
* Tensorflow : 2.18.0

# Pre-installation

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions

### verify CUDA-capable GPU
```
lspci | grep -i nvidia
01:00.0 VGA compatible controller: NVIDIA Corporation Device 28a1 (rev a1)
01:00.1 Audio device: NVIDIA Corporation Device 22be (rev a1)
```
* NVDIA Driver (https://www.nvidia.com/Download/index.aspx?lang=en-us)

* supported version of linux
```
uname -m && cat /etc/*release
```

### verify gcc installed
```
gcc --version
gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

* kernel header (https://docs.nvidia.com/cuda/archive/11.8.0/cuda-installation-guide-linux/index.html)

```
uname -r
6.2.0-26-generic
```
### nvidia driver check

* CUDA 11 and Later Defaults to Minor Version Compatibility

https://docs.nvidia.com/deploy/cuda-compatibility/#default-to-minor-version

Table 1. Example CUDA Toolkit 11.x Minimum Required Driver Versions (Refer to CUDA Release Notes)
CUDA Toolkit	Linux x86_64 Minimum Required Driver Version	Windows Minimum Required Driver Version
CUDA 12.x	>=525.60.13	>=527.41
CUDA 11.x	>= 450.80.02*	>=452.39*


### ~~nvidia dirver install (autoinstall)~~

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#driver-installation

```
sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall

# sudo apt install nvidia-driver

sudo reboot
```

* uninstall previous nvidia dirver
```
sudo apt remove --purge nvidia-*
sudo apt autoremove --purge
sudo apt clean
sudo reboot
```

* nvidia driver version
```
nvidia-smi

Tue Aug 29 19:51:58 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
| N/A   41C    P4    N/A /  25W |      6MiB /  6141MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      2067      G   /usr/lib/xorg/Xorg                  4MiB |
+-----------------------------------------------------------------------------+
```

# CUDA Toolkit download and install

### Tensorflow GPU support CUDA, cuDNN version

https://www.tensorflow.org/install/source?hl=ko#gpu

Version	Python version	Compiler	Build tools	cuDNN	CUDA
> tensorflow-2.13.0	3.8-3.11	Clang 16.0.0	Bazel 5.3.0	8.6	11.8

## CUDA Toolkit 11.8 (Runfile Installation) :smile:
https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#id8

https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local

* disable nouveau driver

```
sudo vi /etc/modprobe.d/blacklist-nouveau.conf
blacklist nouveau
options nouveau modeset=0


cat /etc/modprobe.d/blacklist-nouveau.conf
```
```
sudo update-initramfs -u
sudo reboot 
```

```
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

-----------------------
= Summary =
-----------------------

Driver:   Not Selected
Toolkit:  Installed in /usr/local/cuda-11.8/

Please make sure that
 -   PATH includes /usr/local/cuda-11.8/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-11.8/lib64, or, add /usr/local/cuda-11.8/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-11.8/bin
***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 520.00 is required for CUDA 11.8 functionality to work.
To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
    sudo <CudaInstaller>.run --silent --driver

Logfile is /var/log/cuda-installer.log

```
```
sudo nvidia-xconfig
sudo reboot
```

* check toolkit version
```
nvcc -V

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
```

## CUDA Tookit env 

```
sudo sh -c "echo 'export PATH=$PATH:/usr/local/cuda-11.8/bin' >> /etc/profile"
sudo sh -c "echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64' >> /etc/profile"
sudo sh -c "echo 'export CUDADIR=/usr/local/cuda-11.8' >> /etc/profile"

source /etc/profile
``` 

# cuDNN download and install

https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-deb

### cuDNN 8.6.0

cuDNN v8.6.0 (October 3rd, 2022), for CUDA 11.x (https://developer.nvidia.com/rdp/cudnn-archive#a-collapse860-118)

Download Local Installer for Ubuntu22.04 x86_64 (Deb) -> https://developer.nvidia.com/compute/cudnn/secure/8.6.0/local_installers/11.8/cudnn-local-repo-ubuntu2204-8.6.0.163_1.0-1_amd64.deb

```
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.6.0.163_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update

sudo apt-get install libcudnn8=8.6.0.163-1+cuda11.8
sudo apt-get install libcudnn8-dev=8.6.0.163-1+cuda11.8
sudo apt-get install libcudnn8-samples=8.6.0.163-1+cuda11.8
```

## verifying the install

```
sudo apt-get install libfreeimage3 libfreeimage-dev

cp -r /usr/src/cudnn_samples_v8/ $HOME
cd  $HOME/cudnn_samples_v8/mnistCUDNN
make clean && make
./mnistCUDNN


Executing: mnistCUDNN
cudnnGetVersion() : 8904 , CUDNN_VERSION from cudnn.h : 8904 (8.9.4)
Host compiler version : GCC 11.4.0
...
...
...
Result of classification: 1 3 5

Test passed!

```

# install tensorflow

Version	Python version	Compiler	Build tools	cuDNN	CUDA
tensorflow-2.13.0	3.8-3.11	Clang 16.0.0	Bazel 5.3.0	8.6	11.8

https://www.tensorflow.org/install/pip

```
sudo apt install python3-pip
pip3 install tensorflow

python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.13.*

/*
sudo mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> sudo $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
sudo sh -c "echo 'export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
*/
```
* verify install
```
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

````