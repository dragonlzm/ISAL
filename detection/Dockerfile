ARG PYTORCH="1.5"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

#ARG VERSION="v2.3.0"
ARG VERSION="v2.4.0"
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y build-essential git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
#RUN pip install mmcv-full==1.0.5 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.5.0/index.html
RUN pip install mmcv-full==1.1.1 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.5.0/index.html

# Install MMDetection
RUN conda clean --all
RUN git clone -b ${VERSION} https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .
RUN pip install tensorboard