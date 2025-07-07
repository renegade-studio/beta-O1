# MiniMax-M1 函数调用（Function Call）功能指南

## 📖 简介

MiniMax-M1 模型支持函数调用功能，使模型能够识别何时需要调用外部函数，并以结构化格式输出函数调用参数。本文档详细介绍了如何使用 MiniMax-M1 的函数调用功能。

## 🚀 快速开始

### 使用 vLLM 进行 Function Calls（推荐）

在实际部署过程中，为了支持类似 OpenAI API 的原生 Function Calling（工具调用）能力，MiniMax-M1 模型集成了专属 `tool_call_parser=minimax` 解析器，从而避免对模型输出结果进行额外的正则解析处理。

#### 环境准备与重新编译 vLLM

由于该功能尚未正式发布在 PyPI 版本中，需基于源码进行编译。以下为基于 vLLM 官方 Docker 镜像 `vllm/vllm-openai:v0.8.3` 的示例流程：

```bash
IMAGE=vllm/vllm-openai:v0.8.3
DOCKER_RUN_CMD="--network=host --privileged --ipc=host --ulimit memlock=-1 --shm-size=32gb --rm --gpus all --ulimit stack=67108864"

# 运行 docker
sudo docker run -it -v $MODEL_DIR:$MODEL_DIR \
                    -v $CODE_DIR:$CODE_DIR \
                    --name vllm_function_call \
                    $DOCKER_RUN_CMD \
                    --entrypoint /bin/bash \
                    $IMAGE
```

#### 编译 vLLM 源码

进入容器后，执行以下命令以获取源码并重新安装：

```bash
cd $CODE_DIR
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

#### 启动 vLLM API 服务

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

**⚠️ 注意：**
- `--tool-call-parser minimax` 为关键参数，用于启用 MiniMax-M1 自定义解析器
- `--enable-auto-tool-choice` 启用自动工具选择
- `--chat-template` 模板文件需要适配 tool calling 格式

#### Function Call 测试脚本示例

以下 Python 脚本基于 OpenAI SDK 实现了一个天气查询函数的调用示例：

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

**输出示例：**
```
Function called: get_weather
Arguments: {"location": "San Francisco, CA", "unit": "celsius"}
Result: Getting the weather for San Francisco, CA in celsius...
```

### 手动解析模型输出

如果您无法使用 vLLM 的内置解析器，或者需要使用其他推理框架（如 transformers、TGI 等），可以使用以下方法手动解析模型的原始输出。这种方法需要您自己解析模型输出的 XML 标签格式。

#### 使用 Transformers 的示例

以下是使用 transformers 库的完整示例：

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

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)
prompt = "What's the weather like in Shanghai today?"
messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant created by Minimax based on MiniMax-M1 model."}]},
    {"role": "user", "content": [{"type": "text", "text": prompt}]},
]

# 启用函数调用工具
tools = get_default_tools()

# 应用聊天模板，并加入工具定义
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    tools=tools
)

# 发送请求（这里使用任何推理服务）
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

# 模型输出需要手动解析
raw_output = response.json()["choices"][0]["text"]
print("原始输出:", raw_output)

# 使用下面的解析函数处理输出
function_calls = parse_function_calls(raw_output)
```

## 🛠️ 函数调用的定义

### 函数结构体

函数调用需要在请求体中定义 `tools` 字段，每个函数由以下部分组成：

```json
{
  "tools": [
    {
      "name": "search_web",
      "description": "搜索函数。",
      "parameters": {
        "properties": {
          "query_list": {
            "description": "进行搜索的关键词，列表元素个数为1。",
            "items": { "type": "string" },
            "type": "array"
          },
          "query_tag": {
            "description": "query的分类",
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

**字段说明：**
- `name`: 函数名称
- `description`: 函数功能描述
- `parameters`: 函数参数定义
  - `properties`: 参数属性定义，key 是参数名，value 包含参数的详细描述
  - `required`: 必填参数列表
  - `type`: 参数类型（通常为 "object"）

### 模型内部处理格式

在模型内部处理时，函数定义会被转换为特殊格式并拼接到输入文本中：

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

### 模型输出格式

模型会以以下格式输出函数调用：

```xml
<think>
Okay, I will search for the OpenAI and Gemini latest release.
</think>
<tool_calls>
{"name": "search_web", "arguments": {"query_tag": ["technology", "events"], "query_list": ["\"OpenAI\" \"latest\" \"release\""]}}
{"name": "search_web", "arguments": {"query_tag": ["technology", "events"], "query_list": ["\"Gemini\" \"latest\" \"release\""]}}
</tool_calls>
```

## 📥 手动解析函数调用结果

### 解析函数调用

当需要手动解析时，您需要解析模型输出的 XML 标签格式：

```python
import re
import json
def parse_function_calls(content: str):
    """
    解析模型输出中的函数调用
    """
    function_calls = []
    
    # 匹配 <tool_calls> 标签内的内容
    tool_calls_pattern = r"<tool_calls>(.*?)</tool_calls>"
    tool_calls_match = re.search(tool_calls_pattern, content, re.DOTALL)
    
    if not tool_calls_match:
        return function_calls
    
    tool_calls_content = tool_calls_match.group(1).strip()
    
    # 解析每个函数调用（每行一个JSON对象）
    for line in tool_calls_content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        try:
            # 解析JSON格式的函数调用
            call_data = json.loads(line)
            function_name = call_data.get("name")
            arguments = call_data.get("arguments", {})
            
            function_calls.append({
                "name": function_name,
                "arguments": arguments
            })
            
            print(f"调用函数: {function_name}, 参数: {arguments}")
            
        except json.JSONDecodeError as e:
            print(f"参数解析失败: {line}, 错误: {e}")
    
    return function_calls

# 示例：处理天气查询函数
def execute_function_call(function_name: str, arguments: dict):
    """
    执行函数调用并返回结果
    """
    if function_name == "get_current_weather":
        location = arguments.get("location", "未知位置")
        # 构建函数执行结果
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
                    "weather": "晴朗"
                }, ensure_ascii=False)
              }
            ] 
          }
    elif function_name == "search_web":
        query_list = arguments.get("query_list", [])
        query_tag = arguments.get("query_tag", [])
        # 模拟搜索结果
        return {
            "role": "tool",
            "content": [
              {
                "name": function_name,
                "type": "text",
                "text": f"搜索关键词: {query_list}, 分类: {query_tag}\n搜索结果: 相关信息已找到"
              }
            ]
          }
    
    return None
```

### 将函数执行结果返回给模型

成功解析函数调用后，您应将函数执行结果添加到对话历史中，以便模型在后续交互中能够访问和利用这些信息。

#### 单个结果

假如模型调用了 `search_web` 函数，您可以参考如下格式添加执行结果，`name` 字段为具体的函数名称。

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

对应如下的模型输入格式：
```
<beginning_of_sentence>tool name=tools
tool name: search_web
tool result: test_result
<end_of_sentence>
```

#### 多个结果

假如模型同时调用了 `search_web` 和 `get_current_weather` 函数，您可以参考如下格式添加执行结果，`content`包含多个结果。

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

对应如下的模型输入格式：
```
<beginning_of_sentence>tool name=tools
tool name: search_web
tool result: test_result1
tool name: get_current_weather
tool result: test_result2<end_of_sentence>
```

虽然我们建议您参考以上格式，但只要返回给模型的输入易于理解，`name` 和 `text` 的具体内容完全由您自主决定。

## 📚 参考资料

- [MiniMax-M1 模型仓库](https://github.com/MiniMaxAI/MiniMax-M1)
- [vLLM 项目主页](https://github.com/vllm-project/vllm)
- [vLLM Function Calling PR](https://github.com/vllm-project/vllm/pull/20297)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
