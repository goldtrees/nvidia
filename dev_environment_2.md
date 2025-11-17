# System Requirements
* OS : Ubuntu 24.04
* GPU : NVIDIA® GeForce RTX™ 4070 Laptop GPU
* CUDA Toolkit : 12.5.1
* cuDNN : 9.3
* Tensorflow : 2.18.0

# Pre-installation

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions

### verify CUDA-capable GPU
```
lspci | grep -i nvidia
01:00.0 3D controller: NVIDIA Corporation AD106M [GeForce RTX 4070 Max-Q / Mobile] (rev a1)
```
* NVDIA Driver (https://www.nvidia.com/Download/index.aspx?lang=en-us)

* supported version of linux
```
uname -m && cat /etc/*release
```

### verify gcc installed
```
gcc --version
gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
Copyright (C) 2023 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

```

* kernel header (https://docs.nvidia.com/cuda/archive/12.5.1/cuda-installation-guide-linux/index.html)

```
uname -r
6.14.0-35-generic
```
### NVIDIA Driver check

* CUDA 11 and Later Defaults to Minor Version Compatibility

https://docs.nvidia.com/deploy/cuda-compatibility/#default-to-minor-version

Table 1. Example CUDA Toolkit 11.x Minimum Required Driver Versions (Refer to CUDA Release Notes)
CUDA Toolkit	Linux x86_64 Minimum Required Driver Version	Windows Minimum Required Driver Version
CUDA 12.x	>=525.60.13	>=527.41
CUDA 11.x	>= 450.80.02*	>=452.39*

### Check Tensorflow GPU support CUDA, cuDNN version

https://www.tensorflow.org/install/source?hl=ko#gpu

Version	Python version	Compiler	Build tools	cuDNN	CUDA
> tensorflow-2.18.0	3.9-3.12	Clang 17.0.6	Bazel 6.5.0	9.3	12.5

# CUDA Toolkit download and install (deb(network))

https://developer.nvidia.com/cuda-12-5-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_network

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-5
```

# NVIDIA Driver install

```
sudo apt-get install -y nvidia-open

...

nvidia-smi
Mon Nov 17 20:41:04 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4070 ...    Off |   00000000:01:00.0 Off |                  N/A |
| N/A   50C    P3             10W /   35W |       0MiB /   8188MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

```

* driver version 555 install error
```
sudo apt-get install -y nvidia-driver-555-open
```
or
```
sudo apt-get install -y cuda-drivers-555
```

### CUDA Tookit env 

```
sudo sh -c "echo 'export PATH=$PATH:/usr/local/cuda-12.5/bin' >> $HOME/.profile"
sudo sh -c "echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.5/lib64' >> $HOME/.profile"
sudo sh -c "echo 'export CUDADIR=/usr/local/cuda-12.5' >> $HOME/.profile"

source ~/.profile
``` 

* check toolkit version
```
nvcc -V

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Jun__6_02:18:23_PDT_2024
Cuda compilation tools, release 12.5, V12.5.82
Build cuda_12.5.r12.5/compiler.34385749_0
```

# cuDNN install

https://developer.nvidia.com/cudnn-9-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_network


```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

sudo apt-get -y install cudnn9-cuda-12
```

* verifying cuDNN 

https://docs.nvidia.com/deeplearning/cudnn/installation/latest/linux.html#verifying-the-install-on-linux

```
sudo apt-get install libfreeimage3 libfreeimage-dev
sudo apt-get -y install libcudnn9-samples

cp -r /usr/src/cudnn_samples_v9/ $HOME
cd  $HOME/cudnn_samples_v9/mnistCUDNN

make clean && make
./mnistCUDNN


Executing: mnistCUDNN
cudnnGetVersion() : 91600 , CUDNN_VERSION from cudnn.h : 91600 (9.16.0)
Host compiler version : GCC 13.3.0

There are 1 CUDA capable devices on your machine :
device 0 : sms 36  Capabilities 8.9, SmClock 1230.0 Mhz, MemSize (Mb) 7807, MemClock 8001.0 Mhz, Ecc=0, boardGroupID=0
Using device 0

...

Result of classification: 1 3 5

Test passed!

```

# Tensorflow install

Version	Python version	Compiler	Build tools	cuDNN	CUDA
> tensorflow-2.18.0	3.9-3.12	Clang 17.0.6	Bazel 6.5.0	9.3	12.5

https://www.tensorflow.org/install/pip


### Create a virtual environment with venv

```
sudo apt install python3.12-venv

python3 -m venv tf 

source tf/bin/activate    
```

### [GPU only] Virtual environment configuration

* Create symbolic links to NVIDIA shared libraries:
```
pushd $(dirname $(python -c 'print(__import__("tensorflow").__file__)'))
ln -svf ../nvidia/*/lib/*.so* .
popd
```
* Create a symbolic link to ptxas:
```
ln -sf $(find $(dirname $(dirname $(python -c "import nvidia.cuda_nvcc;         
print(nvidia.cuda_nvcc.__file__)"))/*/bin/) -name ptxas -print -quit) $VIRTUAL_ENV/bin/ptxas
```


### Install Tensorflow

```
python3 -m pip install tensorflow[and-cuda]==2.18.*
```

* verify install
```
# GPU
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

...

[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
````
