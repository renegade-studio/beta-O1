# 🚀 MiniMax Models vLLM Deployment Guide

[vLLM中文版部署指南](./vllm_deployment_guide_cn.md)

## 📖 Introduction

We recommend using [vLLM](https://docs.vllm.ai/en/latest/) to deploy [MiniMax-M1](https://huggingface.co/MiniMaxAI/MiniMax-M1-40k) model. Based on our testing, vLLM performs excellently when deploying this model, with the following features:

- 🔥 Outstanding service throughput performance
- ⚡ Efficient and intelligent memory management
- 📦 Powerful batch request processing capability
- ⚙️ Deeply optimized underlying performance

The MiniMax-M1 model can run efficiently on a single server equipped with 8 H800 or 8 H20 GPUs. In terms of hardware configuration, a server with 8 H800 GPUs can process context inputs up to 2 million tokens, while a server equipped with 8 H20 GPUs can support ultra-long context processing capabilities of up to 5 million tokens.

## 💾 Obtaining MiniMax Models

### MiniMax-M1 Model Obtaining

You can download the model from our official HuggingFace repository: [MiniMax-M1-40k](https://huggingface.co/MiniMaxAI/MiniMax-M1-40k), [MiniMax-M1-80k](https://huggingface.co/MiniMaxAI/MiniMax-M1-80k)

Download command:
```
pip install -U huggingface-hub
huggingface-cli download MiniMaxAI/MiniMax-M1-40k
# huggingface-cli download MiniMaxAI/MiniMax-M1-80k

# If you encounter network issues, you can set a proxy
export HF_ENDPOINT=https://hf-mirror.com
```

Or download using git:

```bash
git lfs install
git clone https://huggingface.co/MiniMaxAI/MiniMax-M1-40k
git clone https://huggingface.co/MiniMaxAI/MiniMax-M1-80k
```

⚠️ **Important Note**: Please ensure that [Git LFS](https://git-lfs.github.com/) is installed on your system, which is necessary for completely downloading the model weight files.

## 🛠️ Deployment Options

### Option 1: Deploy Using Docker (Recommended)

To ensure consistency and stability of the deployment environment, we recommend using Docker for deployment.

⚠️ **Version Requirements**: 
- MiniMax-M1 model requires vLLM version 0.9.2 or later for full support
- Special Note: Using vLLM versions below 0.9.2 may result in incompatibility or incorrect precision for the model:
  - For details, see: [Fix minimax model cache & lm_head precision #19592](https://github.com/vllm-project/vllm/pull/19592)

1. Get the container image:

Currently, the official vLLM Docker image for version v0.9.2 has not been released yet.
As an example, we will demonstrate how to manually build vLLM using version v0.8.3.
```bash
docker pull vllm/vllm-openai:v0.8.3
```

2. Run the container:
```bash
# Set environment variables
IMAGE=vllm/vllm-openai:v0.8.3
MODEL_DIR=<model storage path>
CODE_DIR=<code path>
NAME=MiniMaxImage

# Docker run configuration
DOCKER_RUN_CMD="--network=host --privileged --ipc=host --ulimit memlock=-1 --shm-size=2gb --rm --gpus all --ulimit stack=67108864"

# Start the container
sudo docker run -it \
    -v $MODEL_DIR:$MODEL_DIR \
    -v $CODE_DIR:$CODE_DIR \
    --name $NAME \
    $DOCKER_RUN_CMD \
    $IMAGE /bin/bash

# install vLLM
cd $CODE_DIR
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

💡 If you are using other environment configurations, please refer to the [vLLM Installation Guide](https://docs.vllm.ai/en/latest/getting_started/installation.html)

## 🚀 Starting the Service

### Launch MiniMax-M1 Service

```bash
export SAFETENSORS_FAST_GPU=1
export VLLM_USE_V1=0
python3 -m vllm.entrypoints.openai.api_server \
--model <model storage path> \
--tensor-parallel-size 8 \
--trust-remote-code \
--quantization experts_int8  \
--max_model_len 4096 \
--dtype bfloat16
```

### API Call Example

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniMaxAI/MiniMax-M1",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": "Who won the world series in 2020?"}]}
        ]
    }'
```

## ❗ Common Issues

### Module Loading Problems
If you encounter the following error:
```
import vllm._C  # noqa
ModuleNotFoundError: No module named 'vllm._C'
```

Or

```
MiniMax-M1 model is not currently supported
```

We provide two solutions:

#### Solution 1: Copy Dependency Files
```bash
cd <working directory>
git clone https://github.com/vllm-project/vllm.git
cd vllm
cp /usr/local/lib/python3.12/dist-packages/vllm/*.so vllm 
cp -r /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/* vllm/vllm_flash_attn
```

#### Solution 2: Install from Source
```bash
cd <working directory>
git clone https://github.com/vllm-project/vllm.git

cd vllm/
pip install -e .
```

## 📮 Getting Support

If you encounter any issues while deploying MiniMax-M1 model:
- Please check our official documentation
- Contact our technical support team through official channels
- Submit an [Issue](https://github.com/MiniMax-AI/MiniMax-M1/issues) on our GitHub repository

We will continuously optimize the deployment experience of this model and welcome your feedback!
