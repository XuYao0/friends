# AC Agent 项目框架

本文档定义长期对话情感识别 Agent 的首版雏形。目标不是一次性实现完整系统，而是先固定 agent/prompt 模块、程序模块、数据流和目录边界，后续按模块逐步实现。

## 1. 任务边界

AC Agent 面向“基于剧本叙事的长期情绪理解”任务。Agent 按剧情顺序在线接收剧本片段，在评测点预测当前台词说话人的真实情绪，并给出可证据化的情感分析。

核心约束：

1. 预测第 t 句时，只能使用第 t 句及之前已经到达的信息。
2. 人物状态、事件索引、事件详情、摘要和 embedding 都必须在线增量构建。
3. Agent 输入可以包含说话人、台词、场景说明、动作描写和当前局部上下文，但不能包含完整剧本预生成的人物表、全局关系图或未来剧情。
4. 评估目标是情感标签和情感分析质量；记忆质量作为辅助日志、抽查和案例分析对象。
5. 主实验不向记忆更新注入 gold 情绪标签；带标签反馈更新只能作为单独对照组报告。

## 2. 首版运行策略

为控制成本，首版采用“确定性外层 workflow、分支内 LLM tool-calling”：

1. `FriendsTranscriptChunkSource` 按文件顺序读取 `description` 和 `utterance`。
2. `description` 进入 transcript buffer，但不计入 batch size。
3. 普通 `utterance` 进入 transcript buffer，并计入 batch size。
4. 未遇到 eval point 且累计到 `batch_size` 条 utterance 时，产出 `status = "update_memory"` chunk。
5. 读到 `long_context_selected = true` 的 utterance 时，产出 `status = "label"` chunk，`target_utterance_id` 指向该 utterance。
6. `WorkflowRunner` 根据 chunk status 分派到 `MemoryUpdateLoop` 或 `LabelLoop`。
7. 分支内 LLM 可以调用 `search_events`、`read_event`、`update_memory`。
8. `ToolExecutor` 执行 LLM 发起的 tool call，并把 tool result 送回下一轮 LLM 输入。
9. `LabelLoop` 在无 tool call 的最终 assistant message 中返回情绪预测 JSON，程序解析为 `PredictionRecord`。
10. 默认处理完整个 `screenplays/friends_records_renamed_with_selected.jsonl`，读到 EOF 后结束。
11. 若评测点落在 batch 中间，只能使用截至该评测 utterance 的信息，不能使用评测点之后的 utterance。

当前代码状态：

1. `WorkflowRunner -> MemoryUpdateLoop/LabelLoop -> ToolExecutor -> ToolRegistry/AgentTool` 是当前主路径。
2. `OnlineAgentRunner -> AgentLoop -> EmptyAgentModel` 仍保留为早期骨架，后续应迁移或删除。
3. `search_events`、`read_event`、`update_memory` 已经包装成 tool；空参数允许返回空结果或空 delta。
4. 已接入 DeepSeek OpenAI-compatible chat completions；真实运行需要 `DEEPSEEK_API_KEY`。
5. 记忆结构初版见 `acagent/MEMORY_SCHEMA.md`。
6. 已提供 `JsonMemoryStore`：tools 可在每次调用时从本地 JSON 读取 memory，`update_memory` 后写回 JSON。
7. 已提供 `JsonlPredictionStore`：标注结果可追加写入本地 JSONL。
8. 已提供 `JsonlTraceWriter`：每个 chunk 的完整 LLM 输入输出、tool result 和最终预测可追加写入 JSONL。
9. 默认 workflow 读取 `screenplays/friends_records_renamed_with_selected.jsonl`，输出到 `acagent_outputs/default/`。
10. `build_workflow_runner_from_config()` 可从 `acagent/configs/*.yaml` 构建完整 runner。

推荐两档模型：

1. `cheap_update_model`：用于批量生成记忆 delta，要求低成本、稳定 JSON、无需深度推理。
2. `reasoning_model`：用于评测点情绪识别，要求更强推理和证据整合能力。

## 3. Agent / Tool / Prompt 模块

首版 Agent 不再把检索、读取和记忆更新写死在 runner 中，而是统一做成 LLM 可调用工具。Prompt 文件仍用于定义模型行为，但执行能力由 tool registry 暴露。

### 3.0 Tool-Calling Branch Loops

职责：在确定性外层 workflow 的某个分支内，驱动 LLM 决定调用哪些工具，或输出最终结果。

当前接口分两层：

1. 当前主路径：`MemoryUpdateLoop` 和 `LabelLoop` 直接调用 chat completion，解析 provider tool calls，执行工具并回填 tool messages。
2. 早期骨架：`AgentAction`、`AgentModel`、`AgentLoop`、`EmptyAgentModel` 仍保留，但不再是主 workflow 的执行入口。

每轮 LLM 输入应包含：

1. 当前 utterance 的最小字段。
2. transcript。
3. 当前 batch 对应的 utterance 列表，用于 tool 自动补 `source_utterance_ids`。
4. 当前 memory version。
5. 已有 tool results。
6. 可用 tools 的 name、description、input schema。

当前实现中的输入细节：

1. `current_utterance` 由 chunk 中目标 utterance 或最后一条 utterance 构造，最小字段为：

```json
{
  "utterance_id": "string",
  "speaker": "string",
  "text": "string"
}
```

2. `transcript` 是由 transcript 组织函数生成的文本。
3. `WorkflowInput.batch` 保留当前 chunk 内 utterance 列表；prompt 主要使用 transcript，tool 可用 batch 自动补 source ids。
4. `is_eval_point` 标记当前 utterance 是否为评测点。
5. `memory_version` 输入当前 memory 版本号。
6. `tool_results` 作为 provider tool message 回填给下一轮 LLM，包含 tool name、output/error、调用前后 memory version。
7. `tools` 独立输入可用工具的 `name`、`description`、`input_schema`；DeepSeek client 会转换为 function tool schema。

暂不默认输入的内容：

1. `episode_id`、`scene_id`、`turn_index`。
2. `stage_direction`、`scene_context`、`visible_characters`。
3. 独立的 local context 输入。

后续若需要增加这些字段，应作为显式实验变体或明确版本升级处理，避免无意扩大 prompt 信息量。

标注分支存在一个待评估实验变体：当 chunk status 为 `label` 时，只向 LLM 输入目标单句，不携带同一 chunk 中此前积累的 transcript；所有上下文必须来自此前在线记忆更新形成的 `current_event`、人物状态、事件检索和详细事件读取。当前实现仍输入 label chunk 的 transcript，先不改默认行为。

每轮 LLM 输出应记录：

1. 原始 structured response。
2. tool call 及其 arguments。
3. final prediction。
4. finish 标记。

### 3.0.1 Tool Contract

所有业务工具统一使用 `AgentTool` 契约：

```json
{
  "name": "string",
  "description": "string",
  "input_schema": {},
  "output_schema": {},
  "is_read_only": true,
  "is_concurrency_safe": false
}
```

当前核心数据结构：

1. `ToolCall`：LLM 请求调用的工具名和 arguments。
2. `ToolResult`：工具输出、错误、调用前后 memory version。
3. `ToolError`：结构化错误，避免直接中断 agent loop。
4. `ToolRegistry`：集中注册工具。
5. `ToolExecutor`：统一查找工具、执行工具、返回结果。

当前已注册工具：

1. `search_events`：只读；空参数返回 `{"events": []}`；非空参数调用关键词事件检索。
2. `read_event`：只读；空参数返回 `{}`；非空参数读取事件详情并加入 `WorkflowInput.retrieved_event_details`。
3. `update_memory`：写入；空参数应用空 `MemoryDelta`，非空参数会解析人物、关系、事件和当前事件 delta，写入 memory，并推进 memory version。

当前工具细节：

`search_events`

输入：

```json
{
  "query": "string",
  "characters": ["string"],
  "keywords": ["string"],
  "top_k": 5
}
```

实现方式：

1. 只检索 `MemoryState.event_index`，不直接检索 `event_details`。
2. `query` 会按简单词项拆分。
3. 检索词来自 `query`、`characters`、`keywords`。
4. 每个 `EventIndex` 的 `characters`、`keywords`、`short_summary`、`location`、`scene_id`、`time_label` 作为匹配字段。
5. 排序分数由词项命中、人物命中和 `importance` 共同决定。
6. 返回 `event_id`、`short_summary`、`score`、`matched_terms`。

`read_event`

输入：

```json
{
  "event_id": "string"
}
```

实现方式：

1. 根据 `event_id` 查 `MemoryState.event_index[event_id]`。
2. 从索引对象读取 `detail_id`。
3. 根据 `detail_id` 查 `MemoryState.event_details[detail_id]`。
4. 返回 `detail_id`、`event_id`、`description`。
5. 读取到的 `EventDetail` 会追加到 `WorkflowInput.retrieved_event_details`，供后续 LLM turn 使用。

`update_memory`

输入是 `MemoryDelta` 风格 JSON：

```json
{
  "character_updates": [],
  "relationship_updates": [],
  "event_updates": [],
  "current_event_update": {},
  "uncertainties": [],
  "source_utterance_ids": [],
  "mode": "unsupervised"
}
```

实现方式：

1. 如果存在 `JsonMemoryStore`，调用前先读取本地 JSON 中的最新 memory。
2. 将 tool arguments 解析成 `MemoryDelta`。
3. 若 `source_utterance_ids` 为空，自动使用当前 batch utterance ids。
4. 调用 `MemoryState.apply_delta(delta)`。
5. 如果存在 `JsonMemoryStore`，更新后全量写回 JSON。

当前合并策略：

1. `character_updates`：按 `operation` 更新对应人物字段，并递增人物版本。
2. `relationship_updates`：按 `operation` 更新对应关系条目，并递增关系版本。
3. `event_updates`：按 `event_id` 覆盖或新增事件索引；若提供 detail，则按 `detail_id` 覆盖或新增详细事件。
4. `current_event_update`：整体替换当前事件。
5. 每次 delta 应用后，memory 全局版本递增，并记录 `applied_sources`。
6. `operation` 支持：
   - `append`：新增 `MemoryItem`，也是默认值。
   - `update`：用 `target_index` 替换已有 `MemoryItem`。
7. 程序内部保留 `supersede` 操作，但不暴露在模型可见的 `update_memory` tool schema 中。
8. 当前没有自动冲突消解、去重或压缩规则。

### 3.1 Memory Update Agent

职责：从一批在线到达的 utterance 中抽取人物状态、事件和关系变化，输出可合并的 memory delta。

当前实现：

1. `memory_update.md` 已改为 tool-calling prompt。
2. `MemoryUpdateLoop` 会渲染 prompt、调用 LLM、执行 tool calls，并把 tool results 作为 tool message 放回下一轮 LLM 输入。
3. 循环直到 LLM 返回不含 tool call 的普通 assistant message，或达到 `max_llm_turns`。
4. 记忆更新分支可用工具包括 `search_events`、`read_event`、`update_memory`。
5. `update_memory` 已能把 LLM tool arguments 解析成 `MemoryDelta` 并应用到 `MemoryState` 或 `JsonMemoryStore`。
6. `MemoryUpdateLoop` 已接入 `WorkflowRunner`：`status = "update_memory"` chunk 会进入该分支。

输入：

1. batch utterances：按时间顺序排列的台词、说话人、场景说明、动作描写。
2. current scene state：当前 scene 内已有短摘要。
3. compact character states：相关人物的压缩状态。
4. active event summaries：仍在推进或未解决的事件摘要。
5. update mode：`unsupervised` 或 `label_feedback`。

输出：

```json
{
  "character_updates": [],
  "relationship_updates": [],
  "event_updates": [],
  "current_event_update": {},
  "uncertainties": []
}
```

要求：

1. 只记录当前 batch 中可观察到或可谨慎推断的信息。
2. 区分事实、推断和不确定假设。
3. 不输出完整人物状态，只输出 patch/delta。
4. 不把未来信息、episode 概述或标注分析写入主实验记忆。

### 3.2 Event Search Planning Agent

职责：在 LLM tool-calling loop 中根据当前台词、batch、局部上下文和 memory state，决定是否调用 `search_events`。

输入：

1. 当前 utterance 和局部上下文。
2. 当前说话人和被提及人物。
3. compact character states。
4. 当前事件摘要。
5. 事件索引表的可见字段说明。

输出：不再是单独 agent 的最终 JSON，而是 LLM 的 tool call：

```json
{
  "tool": "search_events",
  "arguments": {
    "query": "string",
    "characters": [],
    "keywords": [],
    "top_k": 5
  }
}
```

要求：

1. 查询数量保持少量，默认 1 到 3 个。
2. 优先检索与说话人目标、关系冲突、未解决事件、反常行为相关的历史事件。
3. 检索计划本身需要写入 trace log。

### 3.3 Event Read Decision Agent

职责：根据 `search_events` 返回的候选事件索引，决定是否调用 `read_event`。

输入：

1. 当前 utterance 和局部上下文。
2. event search results。
3. 当前人物状态和当前事件摘要。

输出：LLM 的 tool call：

```json
{
  "tool": "read_event",
  "arguments": {
    "event_id": "string"
  }
}
```

要求：

1. 每个评测点读取详情数量设上限，默认 0 到 3 个。
2. 不相关事件只保留索引级证据，不读取详情。
3. 读取决策需要写入 trace log。

### 3.4 Emotion Labeling Agent

职责：在 `status = "label"` 的 chunk 上预测目标台词说话人的真实情绪；必要时查询事件记忆、读取详细事件，并在返回最终标注前更新记忆。

当前实现：

1. `emotion_labeling.md` 已改为 tool-calling prompt。
2. `LabelLoop` 会渲染 prompt、调用 LLM、执行 tool calls，并把 tool results 作为 tool message 放回下一轮 LLM 输入。
3. 循环直到 LLM 返回不含 tool call 的普通 assistant message，或达到 `max_llm_turns`。
4. 标注分支可用工具包括 `search_events`、`read_event`、`update_memory`。
5. 最终 assistant message 必须是固定 JSON，程序解析为 `EmotionPrediction`，并调用 `EmotionPrediction.validate()`。
6. 可选传入 `JsonlPredictionStore`，将预测记录追加保存到本地 JSONL。
7. `LabelLoop` 已接入 `WorkflowRunner`：`status = "label"` chunk 会进入该分支。

编排顺序：

1. 检查目标台词、transcript、相关人物状态和当前事件。
2. 判断是否需要历史事件；需要时调用 `search_events`。
3. 如果事件索引摘要不足，再调用 `read_event`。
4. 形成情绪预测和 reason。
5. 如果当前 transcript 产生了新的长期或当前事件信息，在最终回答前最多调用一次 `update_memory`。
6. LLM 不再调用工具，返回最终 JSON；程序解析、校验并保存。

输入：

1. `target_utterance_id` 和 `target_utterance`。
2. transcript chunk。
3. chunk speakers。
4. 当前事件摘要。
5. 相关人物状态。
6. retrieved event details。
7. emotion label schema。

输出：eval point 的 final prediction：

```json
{
  "utterance_id": "S01E01_U0010",
  "emotions": ["neutral"],
  "intensities": ["none"],
  "analysis": {
    "observable_facts": [],
    "memory_evidence": [],
    "inferences": [],
    "uncertainties": [],
    "final_reason": "string"
  }
}
```

可用情绪标签使用 `workzone/情绪模型.md` 中的当前 schema：

`anger`, `disgust`, `fear`, `happiness`, `surprise`, `sadness`, `contentment`, `relief`, `interest`, `contempt`, `shame`, `guilt`, `embarrassment`, `neutral`。

要求：

1. 目标是说话人真实情绪，不是台词表面情绪、听者感知情绪或剧情功能情绪。
2. 必须区分可观察事实、历史证据支持的推断和不确定假设。
3. 非 `neutral` 情绪需要强度：`low`, `medium`, `high`。
4. `neutral` 的强度固定为 `none`，且不能与其他情绪并列。
5. 不确定时少列标签，优先保留对当前台词解释有实质贡献的情绪。

保存协议：

`LabelLoop` 返回 `LabelLoopResult`：

```json
{
  "prediction_record": {
    "utterance_id": "S01E01_U0010",
    "memory_version": "mem_00013",
    "prediction": {
      "emotions": ["sadness"],
      "intensities": ["medium"],
      "analysis": {}
    },
    "trace_id": "label-..."
  },
  "raw_prediction": {},
  "tool_results": [],
  "llm_turns": []
}
```

若传入 `JsonlPredictionStore`，每条 JSONL 记录包含：

1. `trace_id`
2. `utterance_id`
3. `memory_version`
4. `prediction`
5. `raw_prediction`
6. `gold_meld`，仅用于离线评测保存，不进入 prompt。

### 3.5 Memory Merge Agent / Rule

职责：把 memory delta 合并到结构化 memory state。

首版使用程序规则合并，不再额外调用 LLM。冲突判断交给分支内 LLM 在调用 `update_memory` 前完成：新信息用 `append`，修正已有条目用 `update`。程序层只执行这些结构化操作，不主动判断语义冲突。

输入：

1. existing memory state。
2. memory delta。
3. merge policy。

输出：

1. 新版本 memory state。
2. version id。
3. merge audit。

要求：

1. 保留版本号和来源 utterance 范围。
2. 程序层执行 `append` / `update`，不做自动冲突消解。
3. 不确定假设不得静默升级为事实。

### 3.6 Memory Compression Agent

职责：当人物状态或事件详情过长时，压缩为可长期保留的紧凑状态。

输入：

1. long character state 或 event detail。
2. token budget。
3. 必须保留字段。

输出：

1. compressed state。
2. dropped_items。
3. retained_evidence_refs。

要求：

1. 保留对后续情绪判断有用的信息：目标、需求、关系张力、未解决问题、稳定行为模式。
2. 删除纯复述、低重要性细节和已被更高层摘要覆盖的内容。

### 3.7 Judge Agent

职责：评估情感分析文本是否支持情绪标签，用于离线评测和错误分析。

输入：

1. gold 或 silver label。
2. model prediction。
3. 当前评测点可用上下文。
4. 可选历史证据。

输出：

```json
{
  "label_correctness": "correct|partial|wrong|ambiguous",
  "analysis_quality": "good|acceptable|poor",
  "evidence_use": "valid|weak|invalid",
  "conflict_with_context": true,
  "comments": "string"
}
```

要求：

1. 判断分析是否支持标签。
2. 判断是否引用有效上下文证据。
3. 判断是否与已知剧情冲突。
4. 判断是否过度臆测。

## 4. 程序模块

### 4.1 数据输入模块 `data_io`

职责：

1. 读取清洗后的剧本 utterance。
2. 读取评测点和 silver/gold labels。
3. 统一 episode、scene、utterance id。
4. 提供在线迭代器，保证不会提前暴露未来 utterance。
5. 隔离具体数据文件格式，后续换数据源时不影响 runner。

建议组件：

1. `UtteranceSource`
2. `FriendsJsonlUtteranceSource`
3. `UtteranceStream`
4. `EvalPointLoader`
5. `EpisodeLoader`
6. `LabelSchema`

当前实现：

1. `FriendsJsonlUtteranceSource` 默认读取 `screenplays/friends_records_renamed_with_selected.jsonl`。
2. `description` 记录不直接产出 utterance，而是更新后续 utterance 的 `scene_context`。
3. `utterance` 记录映射为 `Utterance`。
4. `inline_description` 映射为 `stage_direction`。
5. `episode_id` 格式为 `SxxEyy`。
6. `scene_id` 格式为 `SxxEyy_SCnnn`。
7. `utterance_id` 格式为 `SxxEyy_U000001`，基于 `global_utterance_id`。
8. transcript 专用读取接口 `iter_transcript_items()` 会按原始顺序产出 `description` 和 `utterance`。
9. transcript 中的 description id 格式为 `SxxEyy_D000001`。

### 4.2 在线调度模块 `runner`

职责：

1. 按顺序消费 transcript chunks。
2. 根据 chunk status 分派到记忆更新分支或情绪标注分支。
3. 管理 memory load/save。
4. 在 label chunk 生成 `PredictionRecord`。
5. 记录每个 chunk 对应的 trace。
6. 默认处理完整个 `screenplays/friends_records_renamed_with_selected.jsonl` 后结束。

首版不再单独维护 `local_context`。近距离上下文由 batch transcript 或 label chunk transcript 承担：

1. 非评测点：未满 batch 时只继续累积；满 `batch_size` 时用 batch transcript 更新记忆。
2. 评测点：当前 utterance 加入 transcript chunk，产出 `status = "label"`。
3. 是否让 label chunk 只包含目标单句，作为 `target_only_memory` 实验变体另行实现。
4. 不读取未来 utterance。

当前实现：

1. `WorkflowRunner` 是当前主循环。
2. 默认数据源为 `FriendsTranscriptChunkSource("screenplays/friends_records_renamed_with_selected.jsonl", batch_size=20)`。
3. `status = "update_memory"` 时调用 `MemoryUpdateLoop`。
4. `status = "label"` 时调用 `LabelLoop`。
5. 每个 chunk 创建一条 `AgentTrace`；trace 中保存 LLM turn 输入输出、tool result 和最终 prediction。
6. 若传入 `JsonMemoryStore`，运行前从本地 JSON 加载 memory；每个分支结束后重新加载，保证 tool 内写入后的状态回到 runner。
7. 若传入 `JsonlPredictionStore`，label 结果追加写入 JSONL。
8. 若传入 `JsonlTraceWriter`，每个 chunk 完成后追加一条完整 trace JSONL。
9. `build_default_workflow_runner()` 使用 DeepSeek chat client、本地 JSON memory、predictions JSONL 和 traces JSONL，默认跑完整 Friends JSONL。
10. `build_workflow_runner_from_config()` 会从 `ExperimentConfig` / YAML 接入模型、batch size、tool budget、prompt path 和输出目录。
11. 旧 `OnlineAgentRunner -> AgentLoop` 仍保留为早期骨架，尚未删除，后续应迁移或清理。

### 4.2.1 Transcript 组织模块 `transcript`

职责：

1. 把一个或多个 `Utterance` 或 `TranscriptItem` 组织成稳定、可读、可测试的 transcript 字符串。
2. 从 `screenplays/friends_records_renamed_with_selected.jsonl` 按原始顺序读取 description 和 utterance，切分成可交给下一步处理的 transcript chunk。
3. 只负责 transcript 组织和 chunk 状态，不负责记忆更新、检索、情绪标注等任务指令。
4. 供记忆更新和情绪标注复用。

utterance 行输入字段：

1. `utterance_id`
2. `speaker`
3. `text`

description 行输入字段：

1. `item_id`
2. `kind = "description"`
3. `text`

首版输出示例：

```text
[S01E01_D000001] [description] Scene: Central Perk, Chandler, Joey, Phoebe, and Monica are there.
[S01E01_U000001] Monica: There's nothing to tell!
[S01E01_U000002] Joey: C'mon, you're going out with the guy!
```

使用方式：

1. 记忆更新：`TranscriptBuilder.render(batch)`，再追加 memory update 指令。
2. 情绪标注：当前默认使用 label chunk 的 transcript，再追加 emotion labeling 指令。
3. 若从原始 Friends JSONL 构造 batch transcript，应使用 `iter_transcript_items()`，保证 description 不丢失。
4. 后续若增加 stage direction 或 scene context，应先在 transcript 模块中形成明确版本，而不是散落在 prompt 组织逻辑里。

chunk 输出：

```json
{
  "status": "update_memory|label",
  "transcript": "string",
  "utterance_count": 20,
  "speakers": ["string"],
  "target_utterance_id": "string|null",
  "meld": "object|null"
}
```

chunk 切分规则：

1. `description` 进入当前 transcript buffer，但不计入 `utterance_count`。
2. `utterance` 进入当前 transcript buffer，并计入 `utterance_count`。
3. 读到 `long_context_selected = true` 的 utterance 时，产出 `status = "label"` 的 chunk，`target_utterance_id` 指向该 utterance；`meld` 若存在，仅作为离线评测 gold 元数据保存，不作为触发条件。
4. 未遇到 selected utterance 且累计到 `batch_size` 条 utterance 时，产出 `status = "update_memory"` 的 chunk。
5. 产出 chunk 后清空当前 buffer。
6. 文件结束时若 buffer 中仍有内容，产出一次 `status = "update_memory"` 的 chunk。
7. `speakers` 是当前 chunk 内 utterance speaker 的去重排序列表，用于后续注入人物状态；description 不参与 speaker 集合。

建议组件：

1. `WorkflowRunner`
2. `WorkflowRunnerConfig`
3. `WorkflowRunResult`
4. `build_default_workflow_runner`
5. `build_workflow_runner_from_config`
6. `WorkflowInput`
7. `OnlineAgentRunner`
8. `AgentLoop`
9. `AgentAction`
10. `AgentModel`
11. `ExperimentConfig`

### 4.3 记忆存储模块 `memory`

职责：

1. 存储人物状态、关系状态、事件索引、事件详情、当前事件。
2. 管理 memory version。
3. 支持快照、回放和消融。

建议组件：

1. `MemoryState`
2. `CharacterStore`
3. `RelationshipStore`
4. `EventIndexStore`
5. `EventDetailStore`
6. `CurrentEventState`
7. `MemorySnapshot`

### 4.4 事件检索模块 `retrieval`

职责：

1. 实现 `search_events(query, characters, keywords, time_range, top_k)`。
2. 实现 `read_event(event_id)`。
3. 保证检索库只包含已处理历史。
4. 支持关键词检索、向量检索和混合检索。

建议组件：

1. `EventSearcher`
2. `EventReader`
3. `EmbeddingIndexer`
4. `KeywordIndexer`
5. `HybridRanker`

注意：embedding 只能在事件被写入记忆后生成，不能基于完整剧本预先生成。

### 4.4.1 Tool 模块 `tools`

职责：

1. 定义统一 tool contract。
2. 集中注册业务 tools。
3. 统一执行 tool call。
4. 返回结构化 tool result。

当前组件：

1. `AgentTool`
2. `ToolCall`
3. `ToolResult`
4. `ToolError`
5. `ToolRegistry`
6. `ToolExecutor`
7. `event_tools`
8. `memory_tools`

当前工具：

1. `search_events`
2. `read_event`
3. `update_memory`

### 4.5 LLM 调用模块 `llm`

职责：

1. 封装 OpenAI 或其他模型 API。
2. 支持 cheap/expensive model 分流。
3. 统一 structured tool call / final output 解析。
4. 统一 JSON schema 校验、重试、错误记录。
5. 记录 token、latency、cost。

建议组件：

1. `LLMClient`
2. `PromptRenderer`
3. `DeepSeekChatCompletionClient`
4. `StructuredOutputParser`
5. `RetryPolicy`
6. `CostTracker`

当前实现：

1. `PromptRenderer` 支持 `$变量` 渲染，dict/list/dataclass 会转为稳定 JSON 字符串。
2. `DeepSeekChatCompletionClient` 封装 DeepSeek OpenAI-compatible chat completions。
3. API key 默认读取环境变量 `DEEPSEEK_API_KEY`。
4. 默认 `base_url` 为 `https://api.deepseek.com`。
5. 支持传入 `messages`、完整 tool list、`tool_choice`、`response_format={"type":"json_object"}`、`temperature`、`max_tokens`。
6. 支持 DeepSeek `extra_body` 中的 `thinking` 和 `reasoning_effort`。
7. 支持把项目内部 tool schema 转成 DeepSeek function tool 格式。
8. 返回 `ChatCompletionResult`，包含 raw response、assistant message、content、tool_calls、finish_reason、usage。
9. `StructuredToolCallingAdapter` 已把 provider response 统一转换为内部 action。
10. adapter 输出包含 final action、tool-call action、parse-error action，并保留 raw response、finish reason、usage、provider tool call id。
11. `MemoryUpdateLoop` 和 `LabelLoop` 已通过 adapter 解析 tool calls 和生成 tool result message。
12. tool arguments JSON 解析失败时，loop 会把结构化错误反馈给模型重新生成，而不是直接中断。
13. `MemoryUpdateLoop` 会校验 final memory-update JSON，失败时反馈给模型修正。
14. `LabelLoop` 会校验 final prediction JSON、emotion/intensity schema 和 neutral 约束，失败时反馈给模型修正。
15. `LabelLoop` 达到 `max_llm_turns` 仍无合法 prediction 时，会产生 fallback neutral，并把原因写入 trace。
16. 非法 tool name 由 `ToolExecutor` 返回结构化 tool error，作为 tool result 进入下一轮模型输入并保存在 trace 中。
17. `ToolBudgetGuard` 在程序层强制执行 `max_read_events` 和 `max_tool_calls`。
18. 超过工具预算时不执行真实 tool，loop 会把结构化说明消息追加给模型，并把预算错误保存在 trace/tool result 中。

### 4.6 Prompt 管理模块 `prompts`

职责：

1. 保存各 agent 的 system prompt、developer prompt 和 output schema。
2. 支持版本化。
3. 支持 prompt 渲染测试。

建议文件：

1. `memory_update.md`
2. `event_search_plan.md`
3. `event_read_decision.md`
4. `emotion_labeling.md`
5. `emotion_reasoning.md`
6. `memory_compression.md`
7. `judge.md`

### 4.7 Trace 日志模块 `trace`

职责：

1. 记录每个进入 `WorkflowRunner` 分支的 chunk trace。
2. 重点记录 LLM 每轮输入和输出。
3. 将 tool result 作为下一轮 LLM 输入的一部分记录。
4. 记录 agent 前后的 memory version。
5. 支持错误回溯、复现实验和消融分析。

建议组件：

1. `TraceLogger`
2. `AgentTrace`
3. `LlmTurnTrace`
4. `LlmTurnInput`
5. `LlmTurnOutput`
6. `JsonlTraceWriter`

当前实现：

1. `TraceLogger` 保留内存 trace，供运行中查询。
2. `JsonlTraceWriter` 将每个完成的 `AgentTrace` 追加写入 JSONL。
3. 默认 workflow 写入 `acagent_outputs/default/traces.jsonl`；配置化运行写入 `{output_dir}/traces.jsonl`。
4. 每条 trace 包含完整 `LlmTurnTrace`、LLM input messages、tools、provider raw output、tool call、tool result、memory version 和 final prediction。
5. `JsonlTraceWriter.read_records()` 可重新读取 trace JSONL records。

当前 trace 主结构：

```json
{
  "trace_id": "trace_00001",
  "utterance_id": "S01E01_U0010",
  "episode_id": "S01E01",
  "scene_id": "S01E01_SC001",
  "turn_index": 10,
  "is_eval_point": true,
  "memory_version_before_agent": "mem_00012",
  "memory_version_after_agent": "mem_00013",
  "llm_turns": [
    {
      "input": {
        "messages": [],
        "tools": []
      },
      "output": {
        "raw": {},
        "tool_call": null,
        "final_prediction": null,
        "finish": false,
        "error": null
      },
      "tool_result": null
    }
  ],
  "final_prediction": null
}
```

### 4.8 评测模块 `eval`

职责：

1. 计算情绪标签准确率、micro/macro F1、多标签指标。
2. 分层评估局部可判断、中程依赖、长程依赖、人物关系/性格依赖样本。
3. 调用 Judge Agent 评估分析文本。
4. 输出实验报告。

建议组件：

1. `EmotionMetrics`
2. `StratifiedEvaluator`
3. `JudgeRunner`
4. `ReportBuilder`

### 4.9 消融实验模块 `baselines`

职责：构建可比较的对照实验。

建议 baseline：

1. `current_only`：只看当前 utterance 和可观察上下文。
2. `short_window`：当前 utterance + 短窗口上下文。
3. `long_window`：当前 utterance + 长窗口上下文。
4. `summary_memory`：只使用滚动摘要。
5. `retrieval_memory`：只使用事件检索记忆。
6. `structured_memory`：使用人物/关系/事件结构化记忆。
7. `full_agent`：完整 AC Agent。
8. `full_agent_label_feedback`：带标注反馈记忆更新的对照组。
9. `target_only_memory`：标注点只输入目标单句；近距离上下文必须通过此前记忆更新和工具读取获得。

### 4.10 配置模块 `configs`

职责：

1. 管理模型、batch size、检索 top_k、事件读取上限、prompt 版本。
2. 区分主实验、debug、小样本、消融实验。
3. 固定随机种子和输出路径。

当前配置项：

1. `transcript_path`
2. `batch_size`
3. `cheap_update_model`
4. `reasoning_model`
5. `judge_model`
6. `temperature`
7. `max_tokens`
8. `event_search_top_k`
9. `max_read_events`
10. `max_tool_calls`
11. `local_context_window`
12. `memory_update_mode`
13. `prompt_version`
14. `memory_update_prompt_path`
15. `emotion_labeling_prompt_path`
16. `output_dir`

当前实现：

1. `ExperimentConfig.from_yaml()` 可读取简单 YAML。
2. `build_workflow_runner_from_config()` 根据配置创建 `WorkflowRunner`。
3. 记忆更新分支使用 `cheap_update_model`。
4. 标注分支使用 `reasoning_model`。
5. `temperature`、`max_tokens` 传入 DeepSeek client。
6. `batch_size`、`max_read_events`、`max_tool_calls`、prompt path、`event_search_top_k` 和 `output_dir` 已接入。

## 5. 核心数据结构

### 5.1 Utterance

```json
{
  "episode_id": "S01E01",
  "scene_id": "S01E01_SC001",
  "utterance_id": "S01E01_U0001",
  "turn_index": 1,
  "speaker": "Monica",
  "text": "There's nothing to tell!",
  "stage_direction": "",
  "scene_context": "",
  "visible_characters": []
}
```

### 5.2 Character State

```json
{
  "character": "Monica",
  "version": 12,
  "recent_events": [],
  "short_term_traits": [],
  "long_term_traits": [],
  "relationships": []
}
```

### 5.3 Event Index

```json
{
  "event_id": "EVT_S01E01_0003",
  "scene_id": "S01E01_SC001",
  "time_label": "S01E01 scene 1",
  "location": "Central Perk",
  "characters": [],
  "keywords": [],
  "short_summary": "string",
  "importance": 3,
  "knowledge_scope": "who knows this",
  "detail_id": "EVTD_S01E01_0003"
}
```

### 5.4 Event Detail

```json
{
  "detail_id": "EVTD_S01E01_0003",
  "event_id": "EVT_S01E01_0003",
  "description": "string",
  "scene_id": "S01E01_SC001",
  "time_label": "S01E01 scene 1",
  "location": "Central Perk",
  "knowledge_scope": "string"
}
```

### 5.5 Current Event

```json
{
  "scene_id": "S01E01_SC001",
  "time_label": "S01E01 scene 1",
  "location": "Monica's apartment",
  "summary": "string",
  "characters": [],
  "knowledge_scope": "string"
}
```

### 5.6 Prediction Record

```json
{
  "utterance_id": "S01E01_U0100",
  "memory_version": "mem_00042",
  "prediction": {
    "emotions": [],
    "intensities": [],
    "analysis": {}
  },
  "trace_id": "trace_00042"
}
```

## 6. 首版目录建议

```text
acagent/
  FRAMEWORK.md
  README.md
  configs/
    default.yaml
    debug.yaml
    baselines.yaml
  prompts/
    memory_update.md
    event_search_plan.md
    event_read_decision.md
    emotion_labeling.md
    emotion_reasoning.md
    memory_compression.md
    judge.md
  src/acagent/
    __init__.py
    data_io/
      sources.py
      stream.py
    runner/
      agent_loop.py
      context.py
      label.py
      memory_update.py
      online.py
      workflow.py
    memory/
    retrieval/
    storage/
      json_store.py
      prediction_store.py
    tools/
      base.py
      event_tools.py
      memory_tools.py
      registry.py
      executor.py
    llm/
    trace/
    eval/
    baselines/
  tests/
    test_online_constraints.py
    test_memory_merge.py
    test_event_retrieval.py
    test_friends_jsonl_source.py
    test_json_memory_store.py
    test_label_loop.py
    test_memory_update_loop.py
    test_tool_calling_interfaces.py
    test_prompt_outputs.py
    test_structured_tool_calling_adapter.py
    test_transcript.py
    test_workflow_runner.py
```

## 7. 最小可行实现顺序

当前已完成：

1. 实现 `UtteranceStream` 和 `EvalPointLoader`，保证在线输入顺序。
2. 实现内存版 `MemoryState`、事件索引、事件详情和空 delta merge。
3. 实现关键词版事件检索。
4. 定义 `AgentTool`、`ToolCall`、`ToolResult`、`ToolRegistry`、`ToolExecutor`。
5. 将 `search_events`、`read_event`、`update_memory` 包装为 tools。
6. 实现早期 `AgentLoop` 和占位 `EmptyAgentModel`，作为历史骨架保留。
7. 实现 `FriendsTranscriptChunkSource`，按 `update_memory` / `label` chunk 切分完整 JSONL。
8. 实现 `DeepSeekChatCompletionClient`。
9. 实现 `MemoryUpdateLoop` 和 `LabelLoop` 两个真实 tool-calling 分支。
10. 实现 `WorkflowRunner`，接入两个分支并默认处理完整 Friends JSONL。
11. 实现以 LLM 输入/输出为核心的 `AgentTrace`。
12. 实现 `StructuredToolCallingAdapter`，统一解析 provider tool calls、final action 和 parse-error action。
13. 实现 `JsonlTraceWriter`，将完整 trace 持久化为 JSONL。
14. 实现基础错误处理：parse 失败反馈给模型、非法 prediction 反馈修正、无合法 prediction 时 fallback neutral。
15. 实现 memory item 更新协议：模型可见 `append`、`update`；内部保留 `supersede`。
16. 实现程序层执行约束：`max_read_events` 和 `max_tool_calls` 硬上限，超限后通知模型继续。
17. 实现配置接入：模型、temperature、max tokens、batch size、tool budget、event search top_k、prompt path 和 output_dir。
18. 增加测试，当前 `54 passed`。

下一步：

1. 为 trace 增加 model、latency、token usage、cost 等可选元信息。
2. 跑通真实 DeepSeek 小样本 workflow。
3. 增加 baseline、指标评测和 Judge Agent。

## 8. 需要优先防止的问题

1. 未来信息泄漏：任何摘要、索引、embedding、人物状态都必须来自已处理 utterance。
2. 评测点 batch 泄漏：若 batch 内含评测点，不能用评测点之后内容更新该点之前的记忆。
3. 记忆膨胀：人物状态和事件详情必须有压缩策略。
4. 标签反馈污染主实验：带标签反馈只能作为独立配置。
5. prompt 输出不可解析：所有 agent 输出都必须有 JSON schema 校验和重试。
6. 检索退化成长上下文：每次读取事件详情数量必须有上限。
7. 分析过度臆测：Emotion Reasoning Agent 必须显式列出事实、推断和不确定性。

## 9. 当前阶段交付物

当前阶段已交付可运行的 workflow 和 tool-calling 骨架：

1. `acagent/FRAMEWORK.md`：本文档。
2. `acagent/prompts/*.md`：记忆更新、情绪标注和辅助 agent prompt。
3. `acagent/configs/default.yaml`：默认实验配置。
4. `acagent/src/acagent/`：workflow runner、分支 loop、DeepSeek client、tool contract、memory、retrieval、storage、trace 等模块。
5. `acagent/tests/`：transcript chunk、workflow runner、DeepSeek client、memory merge、事件检索、tool calling 接口和 schema 校验测试。
