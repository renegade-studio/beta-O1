# 🚀 Guia de Deploy do Modelo MiniMax com Transformers

[Transformers中文版部署指南](./transformers_deployment_guide_cn.md)

## 📖 Introdução

Este guia irá te ajudar a fazer o deploy do modelo MiniMax-M1 utilizando a biblioteca [Transformers](https://huggingface.co/docs/transformers/index). O Transformers é uma biblioteca de deep learning amplamente utilizada, que oferece uma vasta coleção de modelos pré-treinados e interfaces flexíveis para operação dos modelos.

## 🛠️ Configuração do Ambiente

### Instalando o Transformers

```bash
pip install transformers torch accelerate
```

## 📋 Exemplo de Uso Básico

O modelo pré-treinado pode ser utilizado da seguinte maneira:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL_PATH = "{MODEL_PATH}"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

messages = [
    {"role": "user", "content": [{"type": "text", "text": "What is your favourite condiment?"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"}]},
    {"role": "user", "content": [{"type": "text", "text": "Do you have mayonnaise recipes?"}]}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

generation_config = GenerationConfig(
    max_new_tokens=20,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=True,
)

generated_ids = model.generate(**model_inputs, generation_config=generation_config)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

## ⚡ Otimização de Desempenho

### Acelerando com Flash Attention

O exemplo acima mostra uma inferência sem nenhum tipo de otimização. No entanto, é possível acelerar significativamente o modelo utilizando [Flash Attention](../perf_train_gpu_one#flash-attention-2), que é uma implementação mais rápida do mecanismo de atenção usado no modelo.

Primeiro, certifique-se de instalar a versão mais recente do Flash Attention 2:

```bash
pip install -U flash-attn --no-build-isolation
```

Além disso, é necessário que seu hardware seja compatível com o Flash Attention 2. Consulte mais informações na documentação oficial do [repositório do Flash Attention](https://github.com/Dao-AILab/flash-attention). Também é recomendado carregar seu modelo em meia precisão (por exemplo, `torch.float16`).

Para carregar e executar um modelo utilizando Flash Attention 2, utilize o exemplo abaixo:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "{MODEL_PATH}"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

prompt = "My favourite condiment is"

model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
response = tokenizer.batch_decode(generated_ids)[0]
print(response)
```

## 📮 Suporte

Se você encontrar qualquer problema durante o deploy do modelo MiniMax-M1:

* Verifique nossa documentação oficial
* Entre em contato com nossa equipe de suporte técnico pelos canais oficiais
* Abra uma Issue no nosso repositório no GitHub

Estamos continuamente otimizando a experiência de deploy no Transformers e valorizamos muito seu feedback!
