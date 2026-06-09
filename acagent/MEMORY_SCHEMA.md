# AC Agent Memory Schema

本文档定义 AC Agent 首版记忆结构。目标是先固定可存储、可检索、可输入 prompt 的对象边界，后续再逐步替换占位实现和检索策略。

## 1. 总体原则

1. 所有记忆必须在线增量构建，只能来自已经处理过的 utterance。
2. 人物状态是高度压缩的长期记忆，预测时可全量输入。
3. 事件索引用于检索和排序，默认可全量扫描或进入轻量检索。
4. 详细事件记忆只在 Agent 调用 `read_event` 时读取。
5. 当前事件是当前 scene 的近距离摘要，预测时始终输入。
6. 事实、推断和不确定假设用 `MemoryItem.fact_type` 标记，不在 schema 层面拆成三套字段。

## 2. 人物状态

代码结构：`CharacterState`

字段：

1. `character`：人物名或匿名 id。
2. `version`：人物状态版本。
3. `recent_events`：近期对人物有影响的事件摘要。
4. `short_term_traits`：短期状态和临时倾向，例如当前目标、压力、需求、情绪基调、刚形成的解释倾向。
5. `long_term_traits`：长期稳定特点，例如人格倾向、行为模式、价值观、自我形象、稳定脆弱点。
6. `relationships`：人物视角下的人际关系状态。

存储方式：

1. `MemoryState.characters: dict[str, CharacterState]`
2. key 为人物名或匿名 id。
3. 每个字段是 `MemoryItem` 列表。

输入策略：

1. 预测时可以全量输入 compact character states。
2. 若人物状态过长，由 `memory_compression` 压缩。
3. prompt 中必须提醒模型区分事实、推断和不确定假设。

## 3. 事件索引

代码结构：`EventIndex`

字段：

1. `event_id`：事件 id。
2. `scene_id`：事件所属 scene。
3. `time_label`：可读时间标签，例如 episode 内顺序、scene 时间或剧情时间。
4. `location`：地点。
5. `characters`：涉及人物。
6. `keywords`：检索关键词。
7. `short_summary`：短摘要。
8. `importance`：重要性，建议 1 到 5。
9. `knowledge_scope`：谁知道这件事。
10. `detail_id`：对应详细事件记忆 id。

存储方式：

1. `MemoryState.event_index: dict[str, EventIndex]`
2. 索引对象应保持短小，可用于关键词、人物、地点、scene、时间标签和重要性排序。

检索策略：

1. 首版使用关键词/人物/地点/摘要的轻量检索。
2. 后续可在事件写入后生成 embedding，加入混合检索。
3. embedding 只能对已写入记忆的事件生成，不能预先对完整剧本生成。
4. `search_events` 返回事件索引，不默认返回详细事件。

## 4. 详细事件记忆

代码结构：`EventDetail`

字段：

1. `detail_id`：详细事件 id。
2. `event_id`：对应事件索引 id。
3. `description`：完整事件描述。
4. `scene_id`：所属 scene。
5. `time_label`：可读时间标签。
6. `location`：地点。
7. `knowledge_scope`：谁知道这件事。

存储方式：

1. `MemoryState.event_details: dict[str, EventDetail]`
2. key 为 `detail_id`。
3. 由 `EventIndex.detail_id` 关联。

读取策略：

1. 详细事件不默认进入 prompt。
2. Agent 必须先调用 `search_events`，再决定是否调用 `read_event`。
3. `ToolBudgetGuard` 对每个 chunk / eval point 的 `read_event` 次数执行硬上限，默认 `max_read_events = 3`。

## 5. 当前事件

代码结构：`CurrentEventState`

字段：

1. `scene_id`：当前 scene。
2. `time_label`：当前 scene 内时间标签。
3. `location`：地点。
4. `summary`：当前 scene 内已经发生的事件摘要。
5. `characters`：当前 scene 中已出现或被显著提及的人物。
6. `knowledge_scope`：当前信息对谁可见。

存储方式：

1. `MemoryState.current_event: CurrentEventState`
2. 当前通过 `current_event_update` 整体替换。
3. scene 切换时重置或压缩进事件记忆仍是待实现策略。

输入策略：

1. 当前事件始终作为近距离上下文输入 Agent。
2. 当前事件比长期事件详情优先级更高。
3. 当前事件只能包含截至当前 utterance 的内容。

## 6. 基础条目

代码结构：`MemoryItem`

字段：

1. `text`：记忆文本。
2. `evidence_refs`：证据引用。
3. `fact_type`：`fact`、`inference`、`uncertain`。
4. `status`：`active`、`superseded` 等状态。

`MemoryItem` 是人物状态和关系状态的通用最小单位。虽然详细事件记忆不再单独拆影响字段，人物状态和关系状态仍可通过 `MemoryItem.evidence_refs` 保留证据回链。

## 7. Memory Item 更新协议

`character_updates` 和 `relationship_updates` 面向模型开放两个操作：

1. `append`：新增一条 `MemoryItem`。这是默认操作，兼容不传 `operation` 的旧调用。
2. `update`：用新的 `MemoryItem` 替换已有条目。必须提供 `target_index`，它是对应列表里的零基下标。

程序内部仍保留 `supersede` 操作，但不列入模型可见的 tool schema。`supersede` 不是删除，它表示旧记忆曾经是可用判断，但现在被更新、更准确或更具体的记忆替代。后续如果由程序策略或更高权限模块使用，它会把旧条目标记为 `superseded` 并追加新条目。

示例：

```json
{
  "character_updates": [
    {
      "character": "Ross",
      "field": "short_term_traits",
      "operation": "update",
      "target_index": 0,
      "item": {
        "text": "Ross is distressed about Carol moving out.",
        "fact_type": "fact",
        "evidence_refs": [{"utterance_id": "S01E01_U000002"}]
      }
    }
  ]
}
```

## 8. 对检索和存储的影响

首版存储：

1. 默认运行态仍使用 `MemoryState` 内存对象。
2. 已提供 `JsonMemoryStore`，可把完整 `MemoryState` 保存为本地 JSON 文件。
3. tools 若接收到 `memory_store`，每次调用时从 JSON 读取最新 memory。
4. `update_memory` tool 更新后会把最新 memory 写回 JSON。
5. 当前不考虑性能，读写都是全量 JSON。

当前更新策略：

1. `update_memory` 接收 `MemoryDelta` 风格 JSON，包括人物状态、关系状态、事件索引、详细事件、当前事件和不确定项。
2. 人物状态模型可见更新支持 `append`、`update`，作用于 `CharacterState.recent_events`、`short_term_traits`、`long_term_traits` 或 `relationships`。
3. 关系状态模型可见更新支持 `append`、`update`，作用于 `RelationshipState.items`。
4. 事件索引按 `event_id` 新增或覆盖。
5. 详细事件按 `detail_id` 新增或覆盖。
6. 当前事件是整体替换。
7. 每次应用 delta 后全局 memory version 递增，并记录 `applied_sources`。
8. 当前没有自动去重、冲突消解或压缩规则；模型只在可见记忆范围内决定 `append` 或 `update`。

首版检索：

1. `search_events` 只查 `EventIndex`。
2. 排序先用关键词、人物、地点、摘要命中，再叠加 `importance`。
3. `read_event` 根据 `event_id -> detail_id -> EventDetail` 读取。
4. 若使用 `JsonMemoryStore`，检索前先从本地 JSON 读取最新 memory。

后续扩展：

1. 为 `EventIndex` 增加 embedding 字段或外部向量索引。
2. 为人物状态增加压缩版本和证据回链。
3. 为 current event 增加 scene 边界检测和自动归档。
