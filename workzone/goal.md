# 本文件说明
该文件是ai-user交流idea的工作文件，有些时候user会在 ## ai-user交互 里写入对ai助手的回应，ai助手在回应“回应”时需要把对应的user“回应”删除，维护一个clean文档。

# 当前任务

现阶段需要完成以下两项工作：

1. 构建长期对话情感识别基准数据集。
2. 设计基于智能体记忆的情感识别 Agent 架构。

## 理想场景
用完整剧本模拟长期对话。

给定一部完整剧本，一个广义的 AI 智能体按剧情顺序阅读剧本。当它读到每一句台词时，都应该能够结合此前剧情、人物关系、事件记忆和当前语境，对该台词所体现的人物情感形成正确理解。

## attention
1. 情感识别目标应明确为“说话人真实情绪”，而不是台词表面情绪、听者感知情绪或剧情功能情绪。后续标注规则需要处理伪装、反讽、压抑、多重情绪等歧义场景。
2. Agent 输入可以包含场景说明、动作描写和说话人身份，以更接近真实对话中的可观察上下文；但不应包含人物表、全局人物关系图、章节边界或其他需要智能体在阅读过程中自行构建的记忆内容。
3. 预测第 t 句台词情绪时，只能使用第 t 句及其之前的信息。人物关系、事件摘要、角色性格、长期记忆等内容都必须在线增量构建，不能使用未来剧情或基于完整剧本预先生成的全局信息。该限制主要约束数据预处理、记忆初始化、摘要生成和评测协议；在线 Agent 按剧情顺序接收输入，本身不会获得第 t 句之后的内容。
4. 需要区分短上下文、长上下文和显式记忆机制的贡献。对照实验至少应包括：当前/短窗口上下文、长窗口上下文、摘要记忆、检索记忆、结构化人物/事件记忆，以及最终 Agent 架构。
5. 需要证明部分样本确实依赖长期信息，而不是只靠当前台词、动作描写或短程上下文即可判断。数据集可以按局部可判断、中程依赖、长程依赖、隐含人物关系/性格依赖等类型分层。
6. 剧本数据适合构造长期叙事情绪理解任务，但不应过度声称其完全代表真实长期对话。论文表述上应强调“基于剧本叙事的长期情绪理解基准”，并说明真实长期对话数据存在隐私、可获得性和标注成本问题，因此剧本是当前更可行的数据来源。
7. 标注阶段可以保留电视剧人物原名，并给强 LLM 提供当季/当集剧情概述和一定长度的对话片段，以利用预训练语料中的角色知识和剧情知识提高标注质量。但需要在论文中明确这是 LLM-assisted / silver-label 标注流程，而不是完全人工 gold label。
8. 标注时应避免使用最终 Agent 的专属记忆机制，否则会造成标注协议和被评测方法之间的循环偏置。更合理的做法是使用相对独立、透明、固定的标注输入协议，例如人物原名、剧情概述、当前对话窗口和必要的历史上下文。
9. 可以通过短上下文标注器与长上下文标注器的分歧来筛选长期依赖样本：若短上下文下不确定或误判，而长上下文下多个标注器稳定一致，并能指出历史证据，则该样本可作为长程依赖候选；若短上下文已经稳定判断，则归为局部可判断样本；若长上下文标注仍不一致，则应标为歧义样本或剔除。
10. 使用当季/当集剧情概述进行标注时，不必简单排斥“上帝视角”信息。若任务目标是识别说话人的真实情绪，那么后续剧情揭示的信息可以帮助标注 latent emotion；某些在第 t 句时缺乏直接证据的情绪，本身也可能通过怪异、反常或不合常理的行为被推断为“存在未知背景信息”。但仍需要区分真实情绪标签、t 时刻可观测证据、后续剧情证据和外部知识证据，避免把完全不可推断的标签混入普通在线识别评测。
11. 保留人物原名会引入参数化知识泄漏风险：模型可能依赖预训练中记住的角色关系和剧情，而不是从给定对话中构建记忆。若研究目标是验证 Agent 记忆机制，需要考虑报告原名版本与匿名/别名版本的对照，或至少在实验分析中单独讨论预训练知识带来的偏置。
12. Agent 记忆机制是提升情感识别的手段，核心评估对象是情感标签和情感分析；记忆内容本身的准确性可以先不作为主评估指标，但需要在论文中明确当前评估是端到端效果评估，并可在后续补充记忆质量抽查或案例分析。
13. 记忆更新可以设置不同对照组：一类只基于第 t 句及可观察上下文更新记忆；另一类额外使用标注情感和标注分析更新记忆，用于检验带监督情感反馈的记忆更新是否能提升后续识别效果。两类设置必须分开报告，避免把标注信息注入主实验设置。
14. 人物状态可以采用较宽的 schema，包括目标、需求、关切、近期事件、行为模式、解释模式、其他值得关注的特点和人际关系等。事实、推断和不确定假设不必在 schema 层面强拆，但应在构建情感推理 prompt 时要求模型显式区分。
15. 情感分析文本的评估标准暂定为 LLM-as-a-judge。评估 prompt 应重点判断分析是否支持情感标签、是否引用有效上下文证据、是否存在与已知剧情冲突的解释，以及是否过度臆测。

## agent架构
Agent 按剧情顺序不断接收台词作为输入。预测第 t 句时，系统只向 Agent 提供第 t 句及其之前已经到达的信息。

每一步的基本流程如下：

1. 接收当前台词及可观察上下文，包括说话人、台词文本、必要的场景说明和动作描写。
2. 将压缩后的人物状态全量放入当前推理上下文，作为长期稳定背景。
3. 通过类工具调用机制访问事件记忆：Agent 先查询事件索引表，再自主决定是否读取少量详细事件描述。
4. 构建情感推理 prompt，将当前台词、可观察上下文、人物状态和读取到的事件记忆输入 LLM，要求输出情感标签和情感分析。
5. 在得到当前情感预测后，更新人物状态、当前事件、事件索引和必要的详细事件记忆。

记忆内容包括：

1. 人物状态：目标、需求、关切、近期事件、行为模式、解释模式、其他值得关注的特点、人际关系。人物状态是高度压缩的长期记忆，每次预测时可以全量输入。
2. 事件索引：时间、地点、人物、事件关键词、短摘要、重要性、事件状态、可知范围、对应详细事件 id。索引用于快速判断哪些历史事件可能与当前台词相关。
3. 详细事件记忆：更完整的事件描述、相关台词范围、人物反应、对人物状态或关系的影响、未解决的问题、可知范围。详细事件不默认全量输入，只在 Agent 判断需要时读取。
4. 当前事件：当前 scene 内已经发生的事件摘要，包括时间、地点、人物、做了什么、可知范围。当前事件始终作为近距离上下文的一部分。

事件记忆访问采用类工具调用接口：

1. `search_events(query, characters, keywords, time_range, top_k)`：根据当前说话人、被提及人物、当前地点、台词关键词、未解决事件和 Agent 主动生成的查询词，返回候选事件索引。
2. `read_event(event_id)`：读取某个候选事件的详细描述。每次预测只能读取少量事件详情，避免退化为把长历史全文塞入 prompt。

事件检索必须满足在线约束。预测第 t 句时，事件索引、详细事件、人物状态和所有摘要只能来自第 t 句之前已经处理过的内容，不能使用完整剧本、未来剧情、全局人物关系图或基于未来内容生成的 embedding/关键词/摘要。

每次情绪预测需要记录记忆访问轨迹，包括输入的人物状态版本、搜索查询、返回的事件索引、实际读取的详细事件 id 和最终进入 prompt 的记忆内容。该日志用于复现实验、消融分析和错误回溯。

构建情感推理 prompt 时，需要提醒模型区分可观察事实、基于历史信息的推断和不确定假设。Agent 的主要输出是情感标签和情感分析，记忆机制用于辅助该输出。

实验中可以设置记忆更新对照：

1. 无标签反馈更新：只使用当前输入和可观察上下文更新记忆。
2. 带标签反馈更新：额外使用标注情感和标注分析更新记忆，用于分析监督式情感反馈对长期记忆和后续情感识别的影响。

## 清洗数据
清洗结果采用 JSON/JSONL 格式，核心单位是一句台词。每条记录包含：

1. `season`：第几季。
2. `episode`：第几集。
3. `scene_id`：本集内第几个场景。
4. `scene`：当前场景/舞台说明文本。`*[Scene: ...]*`、`*[Time Lapse]*`、`*[Cut to ...]*`、`*[Flashback ...]*` 等统一视为场景上下文，不再拆分 `location` 和 `description`。
5. `utterance_id`：本集内第几句台词。
6. `global_utterance_id`：全剧连续台词编号，用于长期记忆和在线顺序评测。
7. `speaker`：说话人。
8. `utterance`：台词文本。
9. `inline_actions`：台词内部动作或语气描写，例如 `*(mortified)*`。
10. `raw`：原始文本片段，用于回溯和调试。
11. `emotion`：后续标注的说话人真实情绪，清洗阶段可置为 `null`。
12. `reason`：后续标注的情绪判断理由，清洗阶段可置为 `null`。

示例：

```json
{
  "season": 1,
  "episode": 1,
  "scene_id": 1,
  "scene": "Scene: Central Perk, Chandler, Joey, Phoebe, and Monica are there.",
  "utterance_id": 1,
  "global_utterance_id": 1,
  "speaker": "Monica",
  "utterance": "There's nothing to tell! He's just some guy I work with!",
  "inline_actions": [],
  "raw": "Monica: There's nothing to tell! He's just some guy I work with!",
  "emotion": null,
  "reason": null
}
```

建议同时输出两种形态：

1. JSONL：一行一条台词，适合标注、训练和 Agent 按顺序读取。
2. JSON：按 episode/scene 嵌套，适合人工检查和数据浏览。

### 剧本解析规则
MD 文件底层可以逐行读取，但解析时不能假设“一行等于一个事件”。一行内部可能同时包含多个事件，例如多个 `speaker: utterance`，或台词后直接粘连 `*[Scene: ...]*`。因此每一行需要先经过事件切分，再进入结构化解析。

解析流程建议采用状态机：

1. metadata 阶段：读取标题、`Written by:`、`Transcribed by:`、`Additional transcribing by:` 等信息，不进入主台词数据。
2. 等待 first scene：正式正文通常从第一个 `*[...]*` 场景/舞台说明开始。
3. scene 内解析：按顺序解析 stage event、dialogue、action 等事件。
4. 遇到单行 `End` 停止正文解析，后续网页 footer 或链接信息忽略。

主要事件类型：

1. scene/stage event：`*[...]*`，包括 `Scene:`、`Time Lapse`、`Cut to`、`Flashback`、`Fade to Black` 等，统一更新当前 `scene_id` 和 `scene`。
2. dialogue：`speaker: utterance`。
3. action：独立动作或神态描写，例如 `*(They all stare, bemused.)*`。
4. inline action：台词内部动作或神态描写，例如 `Ross: *(mortified)* Hi.`。
5. note/meta：转录者备注、网页残留或其他非剧情信息，必要时单独记录或忽略。

场景解析注意事项：

1. 场景切换不只有 `Scene:`，还包括 `Time Lapse`、`Cut to`、`Flashback`、`Fade to Black` 等；这些统一视为场景上下文，写入清洗结果的 `scene` 字段。
2. `scene` 字段保留完整舞台说明文本，不拆 `location` 和 `description`，避免因逗号、句号、冒号等分隔不稳定导致错误。
3. 每次遇到新的 `*[...]*` 场景/舞台说明，都递增 `scene_id`，并将后续台词关联到新的 `scene`。

台词解析注意事项：

1. `speaker` 不限于六个主角，可能包括 `All`、`Waitress`、`Priest on TV`、`Ross's Mom`、`The Presenter`、`Phoebe, Ross, Chandler, and Joey` 等。
2. 不要见到冒号就当作 speaker。台词内部也会出现冒号，例如 `Here's a question: ...`、`Chapter One: ...`。speaker 判断应要求冒号出现在事件边界附近，冒号前文本较短，且符合说话人命名模式或已知 speaker 候选。
3. 同一行可能出现多个 speaker，需要切成多条 dialogue 事件。
4. 台词中出现的 `*(...)*` 应抽取为 `inline_actions`。

备注和结束规则：

1. 独立 `*(...)*` 作为 action 保留在内部事件流中，因为动作和神态可能为情绪识别提供重要证据。
2. `{Transcriber’s Note: ...}`、网页 footer、导航链接等非剧情内容可归为 note/meta，默认不进入主 JSONL。
3. 单行 `End` 表示本集正文结束，后续内容不参与台词清洗。

清洗流程采用“clean agent 格式校准 + 确定性脚本抽取”的组合：

1. 先让 clean agent 根据 `skills/clean-friends-screenplay` 对 MD 文件做格式更正，重点处理粘连台词、坏掉的场景括号、无 speaker 的歌词/海报文字、metadata 变体、冒号误判等问题。
2. 再运行确定性脚本生成 JSON/JSONL 和 parse warnings。
3. 对 warnings 继续交给 clean agent 或人工复查；不要试图让脚本覆盖所有语义歧义。

### 情绪模型
情绪标签采用 Plutchik 情绪轮，并额外加入 `neutral`。Plutchik 情绪轮包含 8 个基础情绪维度，每个维度按强度分为低、中、高三级。程序字段建议保留为英文枚举值，中文名称用于标注说明和人工检查。

可用标签如下：

| emotion_family | intensity | emotion | 中文 |
| --- | --- | --- | --- |
| joy | low | serenity | 安宁/愉悦 |
| joy | medium | joy | 高兴 |
| joy | high | ecstasy | 狂喜 |
| trust | low | acceptance | 接纳 |
| trust | medium | trust | 信任 |
| trust | high | admiration | 敬佩 |
| fear | low | apprehension | 忧虑 |
| fear | medium | fear | 恐惧 |
| fear | high | terror | 惊恐 |
| surprise | low | distraction | 分心/诧异 |
| surprise | medium | surprise | 惊讶 |
| surprise | high | amazement | 震惊 |
| sadness | low | pensiveness | 沉思/惆怅 |
| sadness | medium | sadness | 悲伤 |
| sadness | high | grief | 悲痛 |
| disgust | low | boredom | 厌烦 |
| disgust | medium | disgust | 厌恶 |
| disgust | high | loathing | 憎恶 |
| anger | low | annoyance | 烦躁 |
| anger | medium | anger | 愤怒 |
| anger | high | rage | 暴怒 |
| anticipation | low | interest | 兴趣 |
| anticipation | medium | anticipation | 期待 |
| anticipation | high | vigilance | 警觉/高度期待 |
| neutral | none | neutral | 中性/无明显情绪 |

Plutchik 情绪轮还定义了由两个基础情绪交织形成的复合情绪，即 dyads。复合情绪可以作为后续标注的可选扩展字段：

```json
{
  "compound_emotion": "love",
  "emotion_components": ["joy", "trust"]
}
```

复合情绪不替代主标签 `emotion`。主标签仍从上表 `emotion` 列中选择；当一句台词明显体现混合情绪时，再补充 `compound_emotion` 和 `emotion_components`。

可用复合情绪如下：

| dyad_type | emotion_components | compound_emotion | 中文 |
| --- | --- | --- | --- |
| primary | anticipation + joy | optimism | 乐观 |
| primary | joy + trust | love | 爱/喜爱 |
| primary | trust + fear | submission | 顺从 |
| primary | fear + surprise | awe | 敬畏/惊叹 |
| primary | surprise + sadness | disapproval | 不赞同/失望 |
| primary | sadness + disgust | remorse | 懊悔 |
| primary | disgust + anger | contempt | 轻蔑 |
| primary | anger + anticipation | aggressiveness | 攻击性 |
| secondary | joy + fear | guilt | 内疚 |
| secondary | trust + surprise | curiosity | 好奇 |
| secondary | fear + sadness | despair | 绝望 |
| secondary | surprise + disgust | unbelief | 难以置信 |
| secondary | sadness + anger | envy | 嫉妒 |
| secondary | disgust + anticipation | cynicism | 犬儒/讥诮 |
| secondary | anger + joy | pride | 自豪 |
| secondary | anticipation + trust | hope | 希望 |
| tertiary | joy + surprise | delight | 欣喜 |
| tertiary | trust + sadness | sentimentality | 伤感/多愁善感 |
| tertiary | fear + disgust | shame | 羞耻 |
| tertiary | surprise + anger | outrage | 愤慨 |
| tertiary | sadness + anticipation | pessimism | 悲观 |
| tertiary | disgust + joy | morbidness | 病态快感/怪异愉悦 |
| tertiary | anger + trust | dominance | 支配感 |
| tertiary | anticipation + fear | anxiety | 焦虑 |

关于情绪轮距离：primary dyads 是相邻情绪组合，secondary dyads 是轮上距离更远一格的组合，tertiary dyads 是再远一格的组合。若继续相隔到正对面，则是对立情绪组合，例如 `joy + sadness`、`trust + disgust`、`fear + anger`、`surprise + anticipation`。对立组合通常不纳入 Plutchik 标准 24 个 dyads，可在本任务中视为复杂矛盾情绪；如确有需要，后续可单独增加 `mixed_conflict` 或 `ambivalent` 标记。

清洗阶段 `emotion` 可置为 `null`；标注阶段 `emotion` 应从上表 `emotion` 列中选择。必要时可以在后续标注 schema 中额外加入 `emotion_family`、`intensity`、`compound_emotion` 和 `emotion_components`，但主标签以 `emotion` 为准。

## ai-user 交互


S05E24合并到了S05E23。
S06E16合并到了S06E15。
S06E25合并到了S06E24。
S07E24合并到了S07E23。
S08E24合并到了S08E23。
S09E24合并到了S09E23。
S10E18合并到了S10E17。