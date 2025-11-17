# 

'''
$ sudo apt update
$ sudo apt install build-essential -y
'''

# NVIDIA Driver Install
 https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html

'''
$ distro=ubuntu2404

$ wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/x86_64/cuda-keyring_1.1-1_all.deb
$ dpkg -i cuda-keyring_1.1-1_all.deb
$ apt update

$ apt install nvidia-open
'''

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html


# NVIDIA Container Toolkit Install

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

'''
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
   curl \
   gnupg2

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update

export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.18.0-1
  sudo apt-get install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
'''

# 
https://www.yunseo.kim/ko/posts/how-to-build-a-deep-learning-development-environment-with-nvidia-container-toolkit-and-docker-1/



# Podman Install

'''
sudo apt install podman

sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

$ nvidia-ctk cdi list
INFO[0000] Found 3 CDI devices                          
nvidia.com/gpu=0
nvidia.com/gpu=GPU-115a17b8-4afe-668b-2566-f889012497db
nvidia.com/gpu=all
'''

# Test Sample workload
'''
podman run --rm --device nvidia.com/gpu=all --security-opt=label=disable ubuntu nvidia-smi

Sun Nov 16 08:30:52 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4070 ...    On  |   00000000:01:00.0 Off |                  N/A |
| N/A   38C    P3             10W /   35W |      14MiB /   8188MiB |     17%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
'''

# Tensoflow Install
https://www.tensorflow.org/install/source?hl=ko#setup_for_linux_and_macos

버전	파이썬 버전	컴파일러	빌드 도구	cuDNN	쿠다
텐서플로우-2.20.0	3.9-3.13	클랭 18.1.8	바젤 7.4.1	9.3	12.5


