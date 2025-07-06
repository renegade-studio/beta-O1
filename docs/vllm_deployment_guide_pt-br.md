# 🚀 Guia de Deploy dos Modelos MiniMax com vLLM

[vLLM中文版部署指南](./vllm_deployment_guide_cn.md)

## 📖 Introdução

Recomendamos utilizar o [vLLM](https://docs.vllm.ai/en/latest/) para fazer o deploy do modelo [MiniMax-M1](https://huggingface.co/MiniMaxAI/MiniMax-M1-40k). Com base nos nossos testes, o vLLM apresenta excelente desempenho ao executar este modelo, oferecendo as seguintes vantagens:

- 🔥 Desempenho excepcional em throughput de serviço
- ⚡ Gerenciamento de memória eficiente e inteligente
- 📦 Capacidade robusta de processamento de requisições em lote
- ⚙️ Otimização profunda de desempenho em baixo nível

O modelo MiniMax-M1 pode ser executado de forma eficiente em um servidor único equipado com 8 GPUs H800 ou 8 GPUs H20. Em termos de configuração de hardware, um servidor com 8 GPUs H800 consegue processar entradas de contexto com até 2 milhões de tokens, enquanto um servidor equipado com 8 GPUs H20 suporta contextos ultra longos de até 5 milhões de tokens.

## 💾 Obtendo os Modelos MiniMax

### Download do Modelo MiniMax-M1

Você pode baixar o modelo diretamente do nosso repositório oficial no HuggingFace: [MiniMax-M1-40k](https://huggingface.co/MiniMaxAI/MiniMax-M1-40k) ou [MiniMax-M1-80k](https://huggingface.co/MiniMaxAI/MiniMax-M1-80k).

Comando para download:
```
pip install -U huggingface-hub
huggingface-cli download MiniMaxAI/MiniMax-M1-40k

# huggingface-cli download MiniMaxAI/MiniMax-M1-80k

# Se você encontrar problemas de rede, pode configurar um proxy

export HF\_ENDPOINT=[https://hf-mirror.com](https://hf-mirror.com)
```

Ou faça o download usando git:

```bash
git lfs install
git clone https://huggingface.co/MiniMaxAI/MiniMax-M1-40k
git clone https://huggingface.co/MiniMaxAI/MiniMax-M1-80k
```

⚠️ **Atenção Importante**: Certifique-se de que o [Git LFS](https://git-lfs.github.com/) está instalado no seu sistema, pois ele é necessário para baixar completamente os arquivos de pesos do modelo.

## 🛠️ Opções de Deploy

### Opção 1: Deploy Utilizando Docker (Recomendado)

Para garantir consistência e estabilidade no ambiente de deployment, recomendamos utilizar Docker.

⚠️ **Requisitos de Versão**:

* O modelo MiniMax-M1 requer vLLM na versão 0.9.2 ou superior para suporte completo.
* Nota especial: Si se utiliza una versión de vLLM inferior a 0.9.2, pueden surgir problemas de incompatibilidad o precisión incorrecta del modelo:

  * Para más detalles, consulta: [Fix minimax model cache & lm_head precision #19592](https://github.com/vllm-project/vllm/pull/19592)

1. Obtenha a imagem do container:

```bash
docker pull vllm/vllm-openai:v0.8.3
```

2. Execute o container:

```bash
# Defina variáveis de ambiente
IMAGE=vllm/vllm-openai:v0.8.3
MODEL_DIR=<caminho onde estão os modelos>
CODE_DIR=<caminho onde está o código>
NAME=MiniMaxImage

# Configuração do Docker run
DOCKER_RUN_CMD="--network=host --privileged --ipc=host --ulimit memlock=-1 --shm-size=2gb --rm --gpus all --ulimit stack=67108864"

# Inicie o container
sudo docker run -it \
    -v $MODEL_DIR:$MODEL_DIR \
    -v $CODE_DIR:$CODE_DIR \
    --name $NAME \
    $DOCKER_RUN_CMD \
    $IMAGE /bin/bash

cd $CODE_DIR
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

💡 Se você estiver utilizando outra configuração de ambiente, consulte o [Guia de Instalação do vLLM](https://docs.vllm.ai/en/latest/getting_started/installation.html).

## 🚀 Inicializando o Serviço

### Iniciando o Serviço com MiniMax-M1

```bash
export SAFETENSORS_FAST_GPU=1
export VLLM_USE_V1=0
python3 -m vllm.entrypoints.openai.api_server \
--model <caminho onde estão os modelos> \
--tensor-parallel-size 8 \
--trust-remote-code \
--quantization experts_int8  \
--max_model_len 4096 \
--dtype bfloat16
```

### Exemplo de Chamada via API

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

## ❗ Problemas Comuns

### Problemas ao Carregar Módulos

Se você encontrar o erro:

```
import vllm._C  # noqa
ModuleNotFoundError: No module named 'vllm._C'
```

Ou

```
MiniMax-M1 model is not currently supported
```

Disponibilizamos duas soluções:

#### Solução 1: Copiar Arquivos de Dependência

```bash
cd <diretório de trabalho>
git clone https://github.com/vllm-project/vllm.git
cd vllm
cp /usr/local/lib/python3.12/dist-packages/vllm/*.so vllm 
cp -r /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/* vllm/vllm_flash_attn
```

#### Solução 2: Instalar a partir do Código-Fonte

```bash
cd <diretório de trabalho>
git clone https://github.com/vllm-project/vllm.git

cd vllm/
pip install -e .
```

## 📮 Suporte

Se você tiver qualquer problema durante o deploy do modelo MiniMax-M1:

* Consulte nossa documentação oficial
* Entre em contato com nossa equipe de suporte técnico pelos canais oficiais
* Abra uma [Issue](https://github.com/MiniMax-AI/MiniMax-M1/issues) no nosso repositório do GitHub

Estamos constantemente otimizando a experiência de deploy deste modelo e valorizamos muito seu feedback!
