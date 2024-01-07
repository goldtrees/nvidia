# Pre-installation

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions

### verify CUDA-capable GPU
```
lspci | grep -i nvidia
01:00.0 VGA compatible controller: NVIDIA Corporation Device 28a1 (rev a1)
01:00.1 Audio device: NVIDIA Corporation Device 22be (rev a1)
```

* ASUS Vivobook Pro 15 OLED (K6502) (https://www.asus.com/kr/laptops/for-creators/vivobook/asus-vivobook-pro-15-oled-k6502/techspec/)

* NVIDIA® GeForce RTX™ 4050 Laptop GPU
6GB GDDR6

* Linux x64 display dirver version : 535.104 (2023.8.22) (https://www.nvidia.com/Download/driverResults.aspx/210649/en-us/)


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


### nvidia dirver install (autoinstall)

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

cat /etc/modprobe.d/blacklist-nouveau.conf
blacklist nouveau
options nouveau modeset=0
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

### ~~CUDA Toolkit 11.8 (Package Installation)~~ :disappointed:

https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local

```
sudo apt-get install zlib1g

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
sudo apt-get install nvidia-gds

sudo reboot
```

### ~~CUDA Toolkit 12.2 (latest version)~~

https://developer.nvidia.com/cuda-downloadsTensorRT 8.6.1?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local

```
sudo apt-get install zlib1g

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.2.1/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.1-535.86.10-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.1-535.86.10-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
sudo apt-get install nvidia-gds

sudo reboot
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

### ~~cuDNN 8.9.4~~

Download local installer for Ubuntu22.04 x86_64 (Deb) -> https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.4/local_installers/12.x/cudnn-local-repo-ubuntu2204-8.9.4.25_1.0-1_amd64.deb

```
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.4.25_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update

sudo apt-get install libcudnn8=8.9.4.25-1+cuda12.2
sudo apt-get install libcudnn8-dev=8.9.4.25-1+cuda12.2
sudo apt-get install libcudnn8-samples=8.9.4.25-1+cuda12.2
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

# install pytorch

https://pytorch.org/get-started/locally/

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## check cuDNN symbolic link
```
ldconfig -N -v $(sed 's/:/ /' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn

	libcudnn_cnn_train.so.8 -> libcudnn_cnn_train.so.8.6.0
	libcudnn_adv_infer.so.8 -> libcudnn_adv_infer.so.8.6.0
	libcudnn_ops_train.so.8 -> libcudnn_ops_train.so.8.6.0
	libcudnn.so.8 -> libcudnn.so.8.6.0
	libcudnn_adv_train.so.8 -> libcudnn_adv_train.so.8.6.0
	libcudnn_ops_infer.so.8 -> libcudnn_ops_infer.so.8.6.0
	libcudnn_cnn_infer.so.8 -> libcudnn_cnn_infer.so.8.6.0

```
## removing CUDA toolkit and driver

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#removing-cuda-toolkit-and-driver

* remove CUDA Toolkit
```
sudo apt-get --purge remove "*cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" \
 "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*" "*nvvm*"
```

* remove NVIDIA drivers
```
sudo apt-get --purge remove "*nvidia*" "libxnvctrl*"
sudo apt-get autoremove
```

### ~~change ubuntu kernel version~~ 

https://packages.ubuntu.com/jammy/amd64/linux-image-5.15.0-25-generic/download

```
wget http://mirrors.kernel.org/ubuntu/pool/main/l/linux-signed/linux-image-5.15.0-25-generic_5.15.0-25.25_amd64.deb

sudo apt install linux-modules-5.15.0-25-generic
sudo dpkg -i linux-image-5.15.0-25-generic_5.15.0-25.25_amd64.deb
```

* enable GRUB menu

````
grep TIMEOUT /etc/default/grub

GRUB_TIMEOUT_STYLE=hidden
GRUB_TIMEOUT=0

sudo vi /etc/default/grub

GRUB_TIMEOUT_STYLE=menu
GRUB_TIMEOUT=5


sudo update-grub

sudo reboot
`````` 

### ~~Wireless Driver~~

https://www.intel.co.kr/content/www/kr/ko/support/articles/000005511/wireless.html

```
sudo tar -xvf iwlwifi-ty-59.601f3a66.0.tgz
sudo mv iwlwifi-ty-59.601f3a66.0/iwlwifi-ty-a0-gf-a0-59.ucode /lib/firmware


sudo lshw -C network
  *-network                 
       description: Wireless interface
       product: Intel Corporation
       vendor: Intel Corporation
       physical id: 14.3
       bus info: pci@0000:00:14.3
       logical name: wlo1
       version: 01
       serial: 70:a8:d3:17:c6:6d
       width: 64 bits
       clock: 33MHz
       capabilities: pm msi pciexpress msix bus_master cap_list ethernet physical wireless
       configuration: broadcast=yes driver=iwlwifi driverversion=6.2.0-31-generic firmware=72.a764baac.0 so-a0-gf-a0-72.uc ip=192.168.1.20 latency=0 link=yes multicast=yes wireless=IEEE 802.11
       resources: iomemory:620-61f irq:16 memory:622d1a4000-622d1a7fff
  *-network
    
sudo apt install backport-iwlwifi-dkms
```


# Install TensorRT

NVIDIA TensorRT Support Matrix --> TensorRT 8.5.3

https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-853/support-matrix/index.html

Supported NVIDIA CUDA® versions	11.8
Supported cuDNN versions	cuDNN 8.6.0

https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html

### Tar file Installation
TensorRT 8.5 GA Update 2 for Linux x86_64 and CUDA 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7 and 11.8 TAR Package
https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.5.3/tars/TensorRT-8.5.3.1.Linux.x86_64-gnu.cuda-11.8.cudnn8.6.tar.gz


Unpack the tar file.
```
version="8.5.3.1"
arch=$(uname -m)
cuda="cuda-11.8"
cudnn="cudnn8.6"
tar -xzvf TensorRT-${version}.Linux.${arch}-gnu.${cuda}.${cudnn}.tar.gz

ls TensorRT-${version}
bin  data  doc  graphsurgeon  include  lib  onnx_graphsurgeon  python  samples  targets  uff

sudo mv TensorRT-${version} /usr/local
```

Add the absolute path to the TensorRTlib directory to the environment variable LD_LIBRARY_PATH:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/TensorRT-${version}/lib >> ~/.bashrc
```

Install the Python TensorRT wheel file.
```
cd TensorRT-${version}/python

python3 -m pip install tensorrt-*-cp3x-none-linux_x86_64.whl
```

Install the Python UFF wheel file. This is only required if you plan to use TensorRT with TensorFlow.
```
cd TensorRT-${version}/uff

python3 -m pip install uff-0.6.9-py2.py3-none-any.whl
```

Check the installation with:
```
which convert-to-uff
```

Install the Python graphsurgeon wheel file.
```
cd TensorRT-${version}/graphsurgeon

python3 -m pip install graphsurgeon-0.4.6-py2.py3-none-any.whl
```
Install the Python onnx-graphsurgeon wheel file.
```
cd TensorRT-${version}/onnx_graphsurgeon
	
python3 -m pip install onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl
```

* Python Package Index Installation

https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#installing-pip


### ~~Debian Installation~~
https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-853/install-guide/index.html#installing-debian

* download TensorRT deb
* TensorRT 8.5 GA Update 2 for Ubuntu 22.04 and CUDA 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7 and 11.8 DEB local repo Package
https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.5.3/local_repos/nv-tensorrt-local-repo-ubuntu2204-8.5.3-cuda-11.8_1.0-1_amd64.deb

Install TensorRT from the Debian local repo package. Replace ubuntuxx04, 8.x.x, and cuda-x.x with your specific OS version, TensorRT version, and CUDA version.d

```
os="ubuntu2204"
tag="8.5.3-cuda-11.8"
sudo dpkg -i nv-tensorrt-local-repo-${os}-${tag}_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-${os}-${tag}/*-keyring.gpg /usr/share/keyrings/

sudo apt-get update
sudo apt-get install tensorrt
```

If using Python 3.x:
```
python3 -m pip install numpy==1.24.3

ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tensorflow 2.13.0 requires numpy<=1.24.3,>=1.22, but you have numpy 1.25.2 which is incompatible.

```
* numpy troubleshooting

https://numpy.org/devdocs/user/troubleshooting-importerror.html#check-environment-variables


The following additional packages will be installed:
```
sudo apt-get install python3-libnvinfer-dev

python3-libnvinfer
```
If you plan to use TensorRT with TensorFlow:
```
python3 -m pip install protobuf
sudo apt-get install uff-converter-tf
```
The graphsurgeon-tf package will also be installed with the preceding command.

If you would like to run the samples that require ONNX graphsurgeon or use the Python module for your own project, run:
```
python3 -m pip install numpy onnx
sudo apt-get install onnx-graphsurgeon
```

Verify the installation.
```
dpkg -l | grep TensorRT
```
