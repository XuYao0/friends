# AC Agent Todo

本文档只记录当前仍需要补齐的工作。已完成的脚手架、prompt、主 workflow、基础 tool-calling loop 和基础测试不再保留在 todo 中。

## 1. 人物选择规则

目标：明确每次 prompt 默认注入哪些人物状态，避免人物状态越注越多，也避免漏掉对当前判断关键的人。

- [ ] 定义相关人物选择规则。
  - [ ] 至少包含当前 speaker。
  - [ ] 包含当前 chunk transcript 中出现过的 speakers。
  - [ ] 可选：从已有 `memory.characters` 中按文本提及做简单匹配。
  - [ ] 明确人物状态注入上限和排序规则。

## 2. Memory Update 后续策略

目标：当前 `update_memory` 已支持新增和修改；冲突判断交给模型基于可见人物状态和当前事件处理，程序层只执行 `append` / `update`。下一步主要是去重和压缩。

- [ ] 设计去重策略。
  - [ ] 同一人物状态重复出现时是否覆盖或跳过。
  - [ ] 事实、推断、不确定假设之间是否需要压缩成更短条目。
  - [ ] 关系状态重复项如何压缩。

## 3. Trace 元信息增强

目标：基础 trace JSONL 持久化已经完成；下一步补充实验复盘和成本分析需要的元信息。

- [ ] 可选元信息。
  - [ ] model name。
  - [ ] latency。
  - [ ] token usage。
  - [ ] cost。

## 4. 后续扩展

- [ ] 增加 `target_only_memory` 标注实验变体。
  - [ ] label chunk prompt 只输入目标单句。
  - [ ] 不输入同 chunk 内此前 transcript。
  - [ ] 依赖 `current_event`、人物状态、`search_events/read_event` 获得上下文。

- [ ] memory compression。
- [ ] embedding/hybrid retrieval。
- [ ] label schema loader。
- [ ] eval metrics 扩展。
- [ ] snapshot/replay。
- [ ] 消融实验配置。
