# DeepSeek Chat Completion API 结构化笔记

来源：[DeepSeek API Docs - 对话补全](https://api-docs.deepseek.com/zh-cn/api/create-chat-completion)

整理日期：2026-06-01

## 基本信息

| 项目 | 值 |
|---|---|
| Endpoint | `POST /chat/completions` |
| 常规 base_url | `https://api.deepseek.com` |
| Beta base_url | `https://api.deepseek.com/beta` |
| Content-Type | `application/json` |
| 功能 | 根据输入上下文补全对话内容 |

Beta 功能包括对话前缀续写等，需要使用 `https://api.deepseek.com/beta`。

## 最小请求

```python
from openai import OpenAI

client = OpenAI(
    api_key="...",
    base_url="https://api.deepseek.com",
)

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {"role": "user", "content": "Hello"}
    ],
)

print(response.choices[0].message.content)
```

## 请求体总览

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---:|---:|---:|---|
| `messages` | `object[]` | 是 | - | 对话消息列表，至少 1 条 |
| `model` | `string` | 是 | - | 模型 ID，支持 `deepseek-v4-flash`、`deepseek-v4-pro` |
| `thinking` | `object \| null` | 否 | `{"type": "enabled"}` | 控制思考模式 |
| `reasoning_effort` | `string` | 否 | `high` | 推理强度：`high`、`max` |
| `max_tokens` | `integer \| null` | 否 | 见模型文档 | 限制输出 token 数 |
| `response_format` | `object \| null` | 否 | `{"type": "text"}` | 输出格式；JSON 模式用 `{"type": "json_object"}` |
| `stop` | `string \| string[] \| null` | 否 | - | 停止词；数组最多 16 个字符串 |
| `stream` | `boolean \| null` | 否 | `false` | 是否使用 SSE 流式输出 |
| `stream_options` | `object \| null` | 否 | - | 流式输出选项，仅 `stream=true` 时可设 |
| `temperature` | `number \| null` | 否 | `1` | 采样温度，最大 `2` |
| `top_p` | `number \| null` | 否 | `1` | nucleus sampling，最大 `1` |
| `tools` | `object[] \| null` | 否 | - | 可调用工具列表，目前仅支持 function，最多 128 个 |
| `tool_choice` | `string \| object \| null` | 否 | 见下文 | 控制工具调用行为 |
| `logprobs` | `boolean \| null` | 否 | - | 是否返回输出 token 的 logprob |
| `top_logprobs` | `integer \| null` | 否 | - | 每个位置返回 top N token 概率，最大 `20`；要求 `logprobs=true` |
| `user_id` | `string \| null` | 否 | - | 业务侧用户标识，最大 512 字符，不要包含隐私信息 |
| `frequency_penalty` | deprecated | 否 | - | 已不支持，传入无效果 |
| `presence_penalty` | deprecated | 否 | - | 已不支持，传入无效果 |

## `messages`

`messages` 是对话历史，数组元素可以是以下四类。

### System Message

```json
{
  "role": "system",
  "content": "You are a helpful assistant.",
  "name": "optional_name"
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---:|---:|---|
| `role` | `string` | 是 | 固定为 `system` |
| `content` | `string` | 是 | system 消息内容 |
| `name` | `string` | 否 | 参与者名称，用于区分相同角色 |

### User Message

```json
{
  "role": "user",
  "content": "Hello",
  "name": "optional_name"
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---:|---:|---|
| `role` | `string` | 是 | 固定为 `user` |
| `content` | `string` | 是 | user 消息内容 |
| `name` | `string` | 否 | 参与者名称 |

### Assistant Message

```json
{
  "role": "assistant",
  "content": "Hello!",
  "name": "optional_name",
  "prefix": true,
  "reasoning_content": "..."
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---:|---:|---|
| `role` | `string` | 是 | 固定为 `assistant` |
| `content` | `string \| null` | 是 | assistant 消息内容 |
| `name` | `string` | 否 | 参与者名称 |
| `prefix` | `boolean` | 否 | Beta；设为 `true` 时，强制模型以这条 assistant 消息内容作为回复前缀继续生成 |
| `reasoning_content` | `string \| null` | 否 | Beta；思考模式 + 前缀续写时，作为最后一条 assistant 思维链内容输入；使用时 `prefix` 必须为 `true` |

前缀续写注意事项：

- `messages` 最后一条必须是 `assistant`。
- 最后一条必须设置 `"prefix": true`。
- 必须使用 `base_url="https://api.deepseek.com/beta"`。

示例：

```python
messages = [
    {"role": "user", "content": "Please write quick sort code"},
    {"role": "assistant", "content": "```python\n", "prefix": True},
]

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
    stop=["```"],
)
```

### Tool Message

```json
{
  "role": "tool",
  "content": "Cloudy 7~13°C",
  "tool_call_id": "call_xxx"
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---:|---:|---|
| `role` | `string` | 是 | 固定为 `tool` |
| `content` | `string` | 是 | 工具执行结果 |
| `tool_call_id` | `string` | 是 | 对应 assistant 返回的 tool call ID |

## 思考模式

### `thinking`

```json
{
  "type": "enabled"
}
```

| 字段 | 可选值 | 默认值 | 说明 |
|---|---|---|---|
| `type` | `enabled` / `disabled` | `enabled` | `enabled` 使用思考模式；`disabled` 使用非思考模式 |

Python SDK 常见写法：

```python
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
    extra_body={"thinking": {"type": "enabled"}},
)
```

### `reasoning_effort`

| 值 | 说明 |
|---|---|
| `high` | 普通请求默认值 |
| `max` | 更高推理强度，复杂 Agent 类请求可能自动设置为此值 |

兼容映射：

| 传入值 | 实际映射 |
|---|---|
| `low` | `high` |
| `medium` | `high` |
| `xhigh` | `max` |

## JSON 输出

请求中设置：

```json
{
  "response_format": {
    "type": "json_object"
  }
}
```

注意：

- JSON 模式保证模型生成的消息是有效 JSON。
- 仍然必须在 system 或 user 消息中明确要求模型输出 JSON。
- 如果没有提示模型输出 JSON，模型可能持续输出空白直到 token 上限。
- 如果 `finish_reason="length"`，说明达到 `max_tokens` 或上下文长度限制，JSON 可能被截断。

用于数据标注时建议：

```python
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {"role": "system", "content": "Return only valid JSON."},
        {"role": "user", "content": prompt},
    ],
    response_format={"type": "json_object"},
)
```

## Tool Calls

### `tools`

目前仅支持 function。

```json
[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get weather of a location.",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {"type": "string"},
          "date": {"type": "string"}
        },
        "required": ["location", "date"]
      },
      "strict": false
    }
  }
]
```

| 字段 | 类型 | 必填 | 说明 |
|---|---:|---:|---|
| `type` | `string` | 是 | 固定为 `function` |
| `function` | `object` | 是 | function 定义 |

`function` 内部字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---:|---:|---|
| `name` | `string` | 是 | 函数名；允许字母、数字、下划线、连字符；最大 64 字符 |
| `description` | `string` | 否 | 函数用途说明，供模型判断何时调用 |
| `parameters` | `object` | 否 | JSON Schema 参数定义；省略表示无参数 |
| `strict` | `boolean` | 否 | 默认 `false`；Beta；设为 `true` 时使用 strict 模式，约束函数调用参数符合 schema |

官方说明提醒：模型生成的 function arguments 不保证永远是有效 JSON，也可能生成 schema 外参数。执行工具前必须在代码中验证参数。

### `tool_choice`

控制模型是否以及如何调用工具。

| 值 | 说明 |
|---|---|
| `none` | 不调用工具，只生成消息 |
| `auto` | 模型自行选择生成消息或调用工具 |
| `required` | 模型必须调用一个或多个工具 |
| `{"type": "function", "function": {"name": "my_function"}}` | 强制调用指定函数 |

默认值：

- 没有 `tools` 时：`none`
- 有 `tools` 时：`auto`

### Tool 调用循环

典型流程：

1. 请求时传入 `tools`。
2. 模型返回 `message.tool_calls`，此时 `finish_reason` 通常是 `tool_calls`。
3. 你的本地程序解析 `tool_calls[*].function.arguments`。
4. 你的本地程序执行对应函数。
5. 把结果作为 `role="tool"` 消息追加回 `messages`。
6. 再次调用 `/chat/completions`，让模型基于工具结果继续回答。

## 非流式响应

响应对象结构：

```json
{
  "id": "chatcmpl_xxx",
  "object": "chat.completion",
  "created": 1705651092,
  "model": "deepseek-v4-pro",
  "system_fingerprint": "fp_xxx",
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": {
        "role": "assistant",
        "content": "Hello!",
        "reasoning_content": "...",
        "tool_calls": []
      },
      "logprobs": null
    }
  ],
  "usage": {
    "prompt_tokens": 16,
    "completion_tokens": 10,
    "total_tokens": 26,
    "prompt_cache_hit_tokens": 0,
    "prompt_cache_miss_tokens": 16,
    "completion_tokens_details": {
      "reasoning_tokens": 0
    }
  }
}
```

顶层字段：

| 字段 | 类型 | 说明 |
|---|---:|---|
| `id` | `string` | 对话唯一标识符 |
| `object` | `string` | 固定为 `chat.completion` |
| `created` | `integer` | Unix 时间戳，秒 |
| `model` | `string` | 生成 completion 的模型 |
| `system_fingerprint` | `string` | 后端配置指纹 |
| `choices` | `object[]` | 生成结果列表 |
| `usage` | `object` | token 用量信息 |

### `choices[]`

| 字段 | 类型 | 说明 |
|---|---:|---|
| `index` | `integer` | 当前 choice 索引 |
| `finish_reason` | `string` | 停止原因 |
| `message` | `object` | assistant 消息 |
| `logprobs` | `object \| null` | token 对数概率信息 |

`finish_reason`：

| 值 | 说明 |
|---|---|
| `stop` | 自然停止，或遇到 `stop` 序列 |
| `length` | 达到上下文长度或 `max_tokens` 限制 |
| `content_filter` | 内容触发过滤 |
| `tool_calls` | 模型请求调用工具 |
| `insufficient_system_resource` | 系统推理资源不足，生成被打断 |

### `message`

| 字段 | 类型 | 说明 |
|---|---:|---|
| `role` | `string` | 固定为 `assistant` |
| `content` | `string \| null` | 最终答案内容 |
| `reasoning_content` | `string \| null` | 思考模式下，最终答案前的推理内容 |
| `tool_calls` | `object[]` | 模型生成的工具调用 |

### `tool_calls[]`

| 字段 | 类型 | 说明 |
|---|---:|---|
| `id` | `string` | tool call ID |
| `type` | `string` | 固定为 `function` |
| `function.name` | `string` | 函数名 |
| `function.arguments` | `string` | JSON 字符串格式的参数 |

### `usage`

| 字段 | 类型 | 说明 |
|---|---:|---|
| `prompt_tokens` | `integer` | prompt token 数，等于 cache hit + miss |
| `completion_tokens` | `integer` | completion token 数 |
| `total_tokens` | `integer` | prompt + completion 总 token |
| `prompt_cache_hit_tokens` | `integer` | 命中上下文缓存的 prompt token |
| `prompt_cache_miss_tokens` | `integer` | 未命中上下文缓存的 prompt token |
| `completion_tokens_details.reasoning_tokens` | `integer` | 思考模式产生的 reasoning token 数 |

## 流式响应

设置：

```json
{
  "stream": true
}
```

响应类型：`text/event-stream`

每个 SSE chunk 是一个 `chat.completion.chunk`：

```json
{
  "id": "chunk_xxx",
  "object": "chat.completion.chunk",
  "created": 1718345013,
  "model": "deepseek-v4-pro",
  "system_fingerprint": "fp_xxx",
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "Hello",
        "reasoning_content": null
      },
      "finish_reason": null,
      "logprobs": null
    }
  ],
  "usage": null
}
```

流结束标记：

```text
data: [DONE]
```

### `stream_options`

仅 `stream=true` 时可设置。

```json
{
  "stream_options": {
    "include_usage": true
  }
}
```

| 字段 | 类型 | 说明 |
|---|---:|---|
| `include_usage` | `boolean` | 若为 `true`，在 `[DONE]` 前额外返回一个 usage 块；该块 `choices` 为空数组，`usage` 是整个请求的 token 统计。其他普通 chunk 也会有 `usage` 字段，但值为 `null` |

### 流式 `choices[].delta`

| 字段 | 类型 | 说明 |
|---|---:|---|
| `role` | `string` | 通常首次 chunk 中为 `assistant` |
| `content` | `string \| null` | 本次增量文本 |
| `reasoning_content` | `string \| null` | 思考模式下的推理增量 |

## Logprobs

请求参数：

```json
{
  "logprobs": true,
  "top_logprobs": 5
}
```

| 字段 | 类型 | 说明 |
|---|---:|---|
| `logprobs` | `boolean` | 是否返回输出 token 的对数概率 |
| `top_logprobs` | `integer` | 返回每个输出位置 top N token 的概率；范围 0-20；要求 `logprobs=true` |

响应中 `logprobs.content[]` 或 `logprobs.reasoning_content[]` 的元素结构：

| 字段 | 类型 | 说明 |
|---|---:|---|
| `token` | `string` | 输出 token |
| `logprob` | `number` | token 对数概率；`-9999.0` 表示概率极小，不在 top 20 内 |
| `bytes` | `integer[] \| null` | token 的 UTF-8 字节表示 |
| `top_logprobs` | `object[]` | 该位置 top N token 及其 logprob |

## 面向数据标注的推荐配置

### 普通 JSON 标注

```python
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a strict data labeling assistant. "
                "Return only valid JSON."
            ),
        },
        {"role": "user", "content": prompt},
    ],
    response_format={"type": "json_object"},
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
)

label = response.choices[0].message.content
```

### 需要工具调用时

```python
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
    tools=tools,
    tool_choice="auto",
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
)
```

你的程序必须负责：

- 校验 `tool_calls[*].function.arguments` 是否为合法 JSON。
- 校验参数是否符合本地函数预期。
- 执行本地函数。
- 把结果作为 `role="tool"` 消息追加回 `messages`。
- 再次调用 API 获取最终答案。

## 常见坑

1. JSON 模式不能只设 `response_format`，prompt 里也要明确要求输出 JSON。
2. `finish_reason="length"` 时，输出可能被截断，JSON 也可能不完整。
3. Tool call 的 arguments 是模型生成的字符串，不保证合法或可信，执行前必须验证。
4. `temperature` 和 `top_p` 通常二选一调，不建议同时改。
5. 前缀续写是 Beta 功能，必须使用 `/beta` base URL。
6. 流式输出的 `usage` 默认不一定在每个 chunk 有统计；需要总用量时设置 `stream_options.include_usage=true`。
7. `frequency_penalty` 和 `presence_penalty` 已废弃，传入不会有效果。
