# MiniMax-M1 Function Call Guide

[FunctionCall中文使用指南](./function_call_guide_cn.md)

## 📖 Introduction

The MiniMax-M1 model supports function calling capabilities, enabling the model to identify when external functions need to be called and output function call parameters in a structured format. This document provides detailed instructions on how to use the function calling feature of MiniMax-M1.

## 🚀 Quick Start

### Using vLLM for Function Calls (Recommended)

In actual deployment, to support native Function Calling (tool calling) capabilities similar to OpenAI API, the MiniMax-M1 model integrates a dedicated `tool_call_parser=minimax` parser, avoiding additional regex parsing of model output.

#### Environment Setup and vLLM Recompilation

Since this feature has not been officially released in the PyPI version, compilation from source code is required. The following is an example process based on the official vLLM Docker image `vllm/vllm-openai:v0.8.3`:

```bash
IMAGE=vllm/vllm-openai:v0.8.3
DOCKER_RUN_CMD="--network=host --privileged --ipc=host --ulimit memlock=-1 --shm-size=32gb --rm --gpus all --ulimit stack=67108864"

# Run docker
sudo docker run -it -v $MODEL_DIR:$MODEL_DIR \
                    -v $CODE_DIR:$CODE_DIR \
                    --name vllm_function_call \
                    $DOCKER_RUN_CMD \
                    --entrypoint /bin/bash \
                    $IMAGE
```

#### Compiling vLLM Source Code

After entering the container, execute the following commands to get the source code and reinstall:

```bash
cd $CODE_DIR
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

#### Starting vLLM API Service

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

**⚠️ Note:**
- `--tool-call-parser minimax` is a key parameter for enabling the MiniMax-M1 custom parser
- `--enable-auto-tool-choice` enables automatic tool selection
- `--chat-template` template file needs to be adapted for tool calling format

#### Function Call Test Script Example

The following Python script implements a weather query function call example based on OpenAI SDK:

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

**Output Example:**
```
Function called: get_weather
Arguments: {"location": "San Francisco, CA", "unit": "celsius"}
Result: Getting the weather for San Francisco, CA in celsius...
```

### Manual Parsing of Model Output

If you cannot use vLLM's built-in parser, or need to use other inference frameworks (such as transformers, TGI, etc.), you can use the following method to manually parse the model's raw output. This method requires you to parse the XML tag format of the model output yourself.

#### Using Transformers Example

The following is a complete example using the transformers library:

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

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
prompt = "What's the weather like in Shanghai today?"
messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant created by Minimax based on MiniMax-M1 model."}]},
    {"role": "user", "content": [{"type": "text", "text": prompt}]},
]

# Enable function call tools
tools = get_default_tools()

# Apply chat template and add tool definitions
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    tools=tools
)

# Send request (using any inference service here)
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

# Model output needs manual parsing
raw_output = response.json()["choices"][0]["text"]
print("Raw output:", raw_output)

# Use the parsing function below to process the output
function_calls = parse_function_calls(raw_output)
```

## 🛠️ Function Call Definition

### Function Structure

Function calls need to be defined in the `tools` field of the request body. Each function consists of the following components:

```json
{
  "tools": [
    {
      "name": "search_web",
      "description": "Search function.",
      "parameters": {
        "properties": {
          "query_list": {
            "description": "Keywords for search, with list element count of 1.",
            "items": { "type": "string" },
            "type": "array"
          },
          "query_tag": {
            "description": "Classification of the query",
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

**Field Descriptions:**
- `name`: Function name
- `description`: Function description
- `parameters`: Function parameter definition
  - `properties`: Parameter property definitions, where key is the parameter name and value contains detailed parameter description
  - `required`: List of required parameters
  - `type`: Parameter type (usually "object")

### Internal Model Processing Format

When processed internally by the model, function definitions are converted to a special format and concatenated to the input text:

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

### Model Output Format

The model outputs function calls in the following format:

```xml
<think>
Okay, I will search for the OpenAI and Gemini latest release.
</think>
<tool_calls>
{"name": "search_web", "arguments": {"query_tag": ["technology", "events"], "query_list": ["\"OpenAI\" \"latest\" \"release\""]}}
{"name": "search_web", "arguments": {"query_tag": ["technology", "events"], "query_list": ["\"Gemini\" \"latest\" \"release\""]}}
</tool_calls>
```

## 📥 Manual Parsing of Function Call Results

### Parsing Function Calls

When manual parsing is required, you need to parse the XML tag format of the model output:

```python
import re
import json
def parse_function_calls(content: str):
    """
    Parse function calls from model output
    """
    function_calls = []
    
    # Match content within <tool_calls> tags
    tool_calls_pattern = r"<tool_calls>(.*?)</tool_calls>"
    tool_calls_match = re.search(tool_calls_pattern, content, re.DOTALL)
    
    if not tool_calls_match:
        return function_calls
    
    tool_calls_content = tool_calls_match.group(1).strip()
    
    # Parse each function call (one JSON object per line)
    for line in tool_calls_content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        try:
            # Parse JSON format function call
            call_data = json.loads(line)
            function_name = call_data.get("name")
            arguments = call_data.get("arguments", {})
            
            function_calls.append({
                "name": function_name,
                "arguments": arguments
            })
            
            print(f"Function call: {function_name}, Arguments: {arguments}")
            
        except json.JSONDecodeError as e:
            print(f"Parameter parsing failed: {line}, Error: {e}")
    
    return function_calls

# Example: Handle weather query function
def execute_function_call(function_name: str, arguments: dict):
    """
    Execute function call and return result
    """
    if function_name == "get_current_weather":
        location = arguments.get("location", "Unknown location")
        # Build function execution result
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
                    "weather": "Sunny"
                }, ensure_ascii=False)
              }
            ] 
          }
    elif function_name == "search_web":
        query_list = arguments.get("query_list", [])
        query_tag = arguments.get("query_tag", [])
        # Simulate search results
        return {
            "role": "tool",
            "content": [
              {
                "name": function_name,
                "type": "text",
                "text": f"Search keywords: {query_list}, Categories: {query_tag}\nSearch results: Relevant information found"
              }
            ]
          }
    
    return None
```

### Returning Function Execution Results to the Model

After successfully parsing function calls, you should add the function execution results to the conversation history so that the model can access and utilize this information in subsequent interactions.

#### Single Result

If the model calls the `search_web` function, you can refer to the following format to add execution results, with the `name` field being the specific function name.

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

Corresponding model input format:
```
<beginning_of_sentence>tool name=tools
tool name: search_web
tool result: test_result
<end_of_sentence>
```

#### Multiple Results

If the model calls both `search_web` and `get_current_weather` functions simultaneously, you can refer to the following format to add execution results, with `content` containing multiple results.

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

Corresponding model input format:
```
<beginning_of_sentence>tool name=tools
tool name: search_web
tool result: test_result1
tool name: get_current_weather
tool result: test_result2<end_of_sentence>
```

While we recommend following the above formats, as long as the input returned to the model is easy to understand, the specific content of `name` and `text` is entirely up to you.

## 📚 References

- [MiniMax-M1 Model Repository](https://github.com/MiniMaxAI/MiniMax-M1)
- [vLLM Project Homepage](https://github.com/vllm-project/vllm)
- [vLLM Function Calling PR](https://github.com/vllm-project/vllm/pull/20297)
- [OpenAI Python SDK](https://github.com/openai/openai-python)