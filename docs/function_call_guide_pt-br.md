# Guia de Uso de Function Call no MiniMax-M1

[FunctionCall中文使用指南](./function_call_guide_cn.md)

## 📖 Introdução

O modelo MiniMax-M1 possui suporte para chamadas de funções (Function Call), permitindo que o modelo identifique quando funções externas precisam ser chamadas e gere os parâmetros dessas chamadas em um formato estruturado. Este documento fornece instruções detalhadas sobre como utilizar o recurso de chamadas de funções do MiniMax-M1.

## 🚀 Início Rápido

### Usando vLLM para Function Calls (Recomendado)

Na implantação real, para suportar capacidades nativas de Function Calling (chamada de ferramentas) semelhantes à API OpenAI, o modelo MiniMax-M1 integra um parser dedicado `tool_call_parser=minimax`, evitando análise regex adicional da saída do modelo.

#### Configuração do Ambiente e Recompilação do vLLM

Como este recurso ainda não foi oficialmente lançado na versão PyPI, é necessária compilação a partir do código fonte. O seguinte é um processo de exemplo baseado na imagem oficial do Docker vLLM `vllm/vllm-openai:v0.8.3`:

```bash
IMAGE=vllm/vllm-openai:v0.8.3
DOCKER_RUN_CMD="--network=host --privileged --ipc=host --ulimit memlock=-1 --shm-size=32gb --rm --gpus all --ulimit stack=67108864"

# Executar docker
sudo docker run -it -v $MODEL_DIR:$MODEL_DIR \
                    -v $CODE_DIR:$CODE_DIR \
                    --name vllm_function_call \
                    $DOCKER_RUN_CMD \
                    --entrypoint /bin/bash \
                    $IMAGE
```

#### Compilando o Código Fonte do vLLM

Após entrar no container, execute os seguintes comandos para obter o código fonte e reinstalar:

```bash
cd $CODE_DIR
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

#### Iniciando o Serviço API vLLM

```bash
export SAFETENSORS_FAST_GPU=1
export VLLM_USE_V1=0

python3 -m vllm.entrypoints.openai.api_server \
--model MiniMax-M1-80k \
--tensor-parallel-size 8 \
--trust-remote-code \
--quantization experts_int8  \
--enable-auto-tool-choice \
--tool-call-parser minimax \
--chat-template vllm/examples/tool_chat_template_minimax_m1.jinja \
--max_model_len 4096 \
--dtype bfloat16 \
--gpu-memory-utilization 0.85
```

**⚠️ Nota:**
- `--tool-call-parser minimax` é um parâmetro chave para habilitar o parser personalizado MiniMax-M1
- `--enable-auto-tool-choice` habilita a seleção automática de ferramentas
- `--chat-template` arquivo de template precisa ser adaptado para o formato de chamada de ferramentas

#### Exemplo de Script de Teste de Function Call

O seguinte script Python implementa um exemplo de chamada de função de consulta meteorológica baseado no SDK OpenAI:

```python
from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

def get_weather(location: str, unit: str):
    return f"Getting the weather for {location} in {unit}..."

tool_functions = {"get_weather": get_weather}

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and state, e.g., 'San Francisco, CA'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location", "unit"]
        }
    }
}]

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[{"role": "user", "content": "What's the weather like in San Francisco? use celsius."}],
    tools=tools,
    tool_choice="auto"
)

print(response)

tool_call = response.choices[0].message.tool_calls[0].function
print(f"Function called: {tool_call.name}")
print(f"Arguments: {tool_call.arguments}")
print(f"Result: {get_weather(**json.loads(tool_call.arguments))}")
```

**Exemplo de Saída:**
```
Function called: get_weather
Arguments: {"location": "San Francisco, CA", "unit": "celsius"}
Result: Getting the weather for San Francisco, CA in celsius...
```

### Análise Manual da Saída do Modelo

Se você não puder usar o parser integrado do vLLM, ou precisar usar outros frameworks de inferência (como transformers, TGI, etc.), você pode usar o seguinte método para analisar manualmente a saída bruta do modelo. Este método requer que você analise o formato de tags XML da saída do modelo.

#### Exemplo Usando Transformers

O seguinte é um exemplo completo usando a biblioteca transformers:

```python
from transformers import AutoTokenizer

def get_default_tools():
    return [
        {
          "name": "get_current_weather",
          "description": "Get the latest weather for a location",
          "parameters": {
              "type": "object", 
              "properties": {
                  "location": {
                      "type": "string", 
                      "description": "A certain city, such as Beijing, Shanghai"
                  }
              }, 
          }
          "required": ["location"],
          "type": "object"
        }
    ]

# Carregar modelo e tokenizador
tokenizer = AutoTokenizer.from_pretrained(model_id)
prompt = "What's the weather like in Shanghai today?"
messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant created by Minimax based on MiniMax-M1 model."}]},
    {"role": "user", "content": [{"type": "text", "text": prompt}]},
]

# Habilitar ferramentas de chamada de função
tools = get_default_tools()

# Aplicar template de chat e adicionar definições de ferramentas
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    tools=tools
)

# Enviar requisição (usando qualquer serviço de inferência aqui)
import requests
payload = {
    "model": "MiniMaxAI/MiniMax-M1-40k",
    "prompt": text,
    "max_tokens": 4000
}
response = requests.post(
    "http://localhost:8000/v1/completions",
    headers={"Content-Type": "application/json"},
    json=payload,
    stream=False,
)

# Saída do modelo precisa de análise manual
raw_output = response.json()["choices"][0]["text"]
print("Saída bruta:", raw_output)

# Use a função de análise abaixo para processar a saída
function_calls = parse_function_calls(raw_output)
```

## 🛠️ Definição de Function Call

### Estrutura da Função

As funções precisam ser definidas no campo `tools` do corpo da requisição. Cada função é composta pelos seguintes elementos:

```json
{
  "tools": [
    {
      "name": "search_web",
      "description": "Função de busca.",
      "parameters": {
        "properties": {
          "query_list": {
            "description": "Palavras-chave para busca, com contagem de elementos da lista de 1.",
            "items": { "type": "string" },
            "type": "array"
          },
          "query_tag": {
            "description": "Classificação da consulta",
            "items": { "type": "string" },
            "type": "array"
          }
        },
        "required": [ "query_list", "query_tag" ],
        "type": "object"
      }
    }
  ]
}
```

**Descrição dos Campos:**

* `name`: Nome da função
* `description`: Descrição da função
* `parameters`: Definição dos parâmetros da função

  * `properties`: Definições dos parâmetros, onde a chave é o nome do parâmetro e o valor contém a descrição detalhada do parâmetro
  * `required`: Lista de parâmetros obrigatórios
  * `type`: Tipo de parâmetro (geralmente "object")

### Formato de Processamento Interno do Modelo

Quando processadas internamente pelo modelo, as definições de função são convertidas para um formato especial e concatenadas ao texto de entrada:

```
<begin_of_document><beginning_of_sentence>system ai_setting=MiniMax AI
MiniMax AI是由上海稀宇科技有限公司（MiniMax）自主研发的AI助理。<end_of_sentence>
<beginning_of_sentence>system tool_setting=tools
You are provided with these tools:
<tools>
{"name": "search_web", "description": "搜索函数。", "parameters": {"properties": {"query_list": {"description": "进行搜索的关键词，列表元素个数为1。", "items": {"type": "string"}, "type": "array"}, "query_tag": {"description": "query的分类", "items": {"type": "string"}, "type": "array"}}, "required": ["query_list", "query_tag"], "type": "object"}}
</tools>
If you need to call tools, please respond with <tool_calls></tool_calls> XML tags, and provide tool-name and json-object of arguments, following the format below:
<tool_calls>
{"name": <tool-name>, "arguments": <args-json-object>}
...
</tool_calls><end_of_sentence>
<beginning_of_sentence>user name=用户
OpenAI 和 Gemini 的最近一次发布会都是什么时候?<end_of_sentence>
<beginning_of_sentence>ai name=MiniMax AI
```

### Formato de Saída do Modelo

O modelo gera chamadas de função no seguinte formato:

```xml
<think>
Ok, vou procurar a versão mais recente do OpenAI e do Gemini.
</think>
<tool_calls>
{"name": "search_web", "arguments": {"query_tag": ["technology", "events"], "query_list": ["\"OpenAI\" \"latest\" \"release\""]}}
{"name": "search_web", "arguments": {"query_tag": ["technology", "events"], "query_list": ["\"Gemini\" \"latest\" \"release\""]}}
</tool_calls>
```

## 📥 Análise Manual dos Resultados de Function Call

### Fazendo o Parse das Chamadas de Função

Quando a análise manual é necessária, você precisa analisar o formato de tags XML da saída do modelo:

```python
import re
import json
def parse_function_calls(content: str):
    """
    Analisar chamadas de função da saída do modelo
    """
    function_calls = []
    
    # Corresponder conteúdo dentro das tags <tool_calls>
    tool_calls_pattern = r"<tool_calls>(.*?)</tool_calls>"
    tool_calls_match = re.search(tool_calls_pattern, content, re.DOTALL)
    
    if not tool_calls_match:
        return function_calls
    
    tool_calls_content = tool_calls_match.group(1).strip()
    
    # Analisar cada chamada de função (um objeto JSON por linha)
    for line in tool_calls_content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        try:
            # Analisar chamada de função em formato JSON
            call_data = json.loads(line)
            function_name = call_data.get("name")
            arguments = call_data.get("arguments", {})
            
            function_calls.append({
                "name": function_name,
                "arguments": arguments
            })
            
            print(f"Chamada de função: {function_name}, Argumentos: {arguments}")
            
        except json.JSONDecodeError as e:
            print(f"Falha na análise de parâmetros: {line}, Erro: {e}")
    
    return function_calls

# Exemplo: Manipular função de consulta de clima
def execute_function_call(function_name: str, arguments: dict):
    """
    Executar chamada de função e retornar resultado
    """
    if function_name == "get_current_weather":
        location = arguments.get("location", "Localização desconhecida")
        # Construir resultado da execução da função
        return {
            "role": "tool", 
            "content": [
              {
                "name": function_name,
                "type": "text",
                "text": json.dumps({
                    "location": location, 
                    "temperature": "25", 
                    "unit": "celsius", 
                    "weather": "Ensolarado"
                }, ensure_ascii=False)
              }
            ] 
          }
    elif function_name == "search_web":
        query_list = arguments.get("query_list", [])
        query_tag = arguments.get("query_tag", [])
        # Simular resultados de pesquisa
        return {
            "role": "tool",
            "content": [
              {
                "name": function_name,
                "type": "text",
                "text": f"Palavras-chave de busca: {query_list}, Categorias: {query_tag}\nResultados da busca: Informações relevantes encontradas"
              }
            ]
          }
    
    return None
```

### Retornando os Resultados da Execução de Função para o Modelo

Após analisar com sucesso as chamadas de função, você deve adicionar os resultados da execução da função ao histórico da conversa para que o modelo possa acessar e utilizar essas informações em interações subsequentes.

#### Resultado Único

Se o modelo chamar a função `search_web`, você pode se referir ao seguinte formato para adicionar resultados de execução, com o campo `name` sendo o nome específico da função.

```json
{
  "role": "tool", 
  "content": [
    {
      "name": "search_web",
      "type": "text",
      "text": "test_result"
    }
  ]
}
```

Formato de entrada correspondente do modelo:
```
<beginning_of_sentence>tool name=tools
tool name: search_web
tool result: test_result
<end_of_sentence>
```

#### Vários Resultados

Se o modelo chamar simultaneamente as funções `search_web` e `get_current_weather`, você pode se referir ao seguinte formato para adicionar resultados de execução, com `content` contendo vários resultados.

```json
{
  "role": "tool", 
  "content": [
    {
      "name": "search_web",
      "type": "text",
      "text": "test_result1"
    },
    {
      "name": "get_current_weather",
      "type": "text",
      "text": "test_result2"
    }
  ]
}
```

Formato de entrada correspondente do modelo:
```
<beginning_of_sentence>tool name=tools
tool name: search_web
tool result: test_result1
tool name: get_current_weather
tool result: test_result2<end_of_sentence>
```

Embora recomendemos seguir os formatos acima, desde que a entrada retornada ao modelo seja fácil de entender, o conteúdo específico de `name` e `text` é inteiramente de sua escolha.

## 📚 Referências

- [Repositório do Modelo MiniMax-M1](https://github.com/MiniMaxAI/MiniMax-M1)
- [Página Principal do Projeto vLLM](https://github.com/vllm-project/vllm)
- [PR de Function Calling do vLLM](https://github.com/vllm-project/vllm/pull/20297)
- [SDK Python OpenAI](https://github.com/openai/openai-python)