FROM nvcr.io/nvidia/pytorch:21.06-py3
ARG DEBIAN_FRONTEND=noninteractive

# Install basics
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        vim \
        tmux \
        wget \
        bzip2 \
        unzip \
        g++ \
        ca-certificates \
        ffmpeg \
        libx264-dev \
        imagemagick \
        libnss3-dev
RUN pip install cmake
RUN pip install pynvml
RUN pip install Pillow>=8.3.2
RUN pip install scipy==1.3.3
RUN pip install scikit-image
RUN pip install tqdm==4.35.0
RUN pip install wget
RUN pip install cython
RUN pip install lmdb
RUN pip install av
RUN pip install opencv-python
RUN pip install opencv-contrib-python
RUN pip install imutils
RUN pip install imageio-ffmpeg
RUN pip install qimage2ndarray
RUN pip install albumentations
RUN pip install requests==2.25.1
RUN pip install nvidia-ml-py3==7.352.0
RUN pip install pyglet
RUN pip install timm
RUN pip install diskcache
RUN pip install boto3
RUN pip install awscli_plugin_endpoint
RUN pip install awscli
RUN pip install rsa
RUN pip install wandb
RUN pip install tensorboard
RUN pip install lpips
RUN pip install face-alignment
RUN pip install dlib
RUN pip install clean-fid
COPY imaginaire/third_party/correlation correlation
RUN cd correlation && rm -rf build dist *-info && python setup.py install
COPY imaginaire/third_party/channelnorm channelnorm
RUN cd channelnorm && rm -rf build dist *-info && python setup.py install
COPY imaginaire/third_party/resample2d resample2d
RUN cd resample2d && rm -rf build dist *-info && python setup.py install
COPY imaginaire/third_party/bias_act bias_act
RUN cd bias_act && rm -rf build dist *-info && python setup.py install
COPY imaginaire/third_party/upfirdn2d upfirdn2d
RUN cd upfirdn2d && rm -rf build dist *-info && python setup.py install
COPY imaginaire/model_utils/gancraft/voxlib gancraft/voxlib
RUN cd gancraft/voxlib && make
