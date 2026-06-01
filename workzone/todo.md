## 已做的工作
  已将 Friends 剧本 Markdown 清洗/转换流程改成新的 utterance/description JSONL schema，并通过
  分批 sub-agent 核对和人工确认修复了剩余格式问题；最终全量转换 warnings 为 0。产出文件在
  tools/clean_screenplay_md.py、screenplays/converted_chunks/cleaned/friends_records.jsonl、
  screenplays/converted_chunks/cleaned/friends_episodes.json、screenplays/converted_chunks/
  cleaned/friends_parse_warnings.jsonl，规范更新在 workzone/archived/剧本清洗.md。

  已将 workzone/archived/各集简介.md 转换为包含 season、episode、description 的 JSONL，并添加了可复用转
  换脚本。产出文件在 tools/convert_episode_descriptions.py 和 screenplays/converted_chunks/
  episode_descriptions.jsonl。

  已添加 DeepSeek/OpenAI SDK 调用测试脚本，用于后续数据标注前验证 API key、模型、thinking 参数
  和 prompt 调用。产出文件在 tools/test_deepseek_labeling.py。

  已整理 DeepSeek Chat Completion API 官方文档为结构化 Markdown，覆盖入参、出参、messages、
  thinking、JSON mode、tool calls、streaming、logprobs 和标注推荐配置。产出文件在 workzone/
  deepseek_chat_completion_api.md。

  已形成情绪模型，产出文件在 workzone/情绪模型.md 

## 要做的工作
写一个脚本，用deepseek标注s01e01的每句台词的情感。输入为 指令+该集简介+剧本台词。
指令应该包含待标注的标签及其描述。
简介从各级简介文件里读取。
剧本台词采用全量方法，第n句台词包括前n-1句历史。为了保证命中缓存，应该把每一次请求的增量内容也就是新的台词放在末尾。
注意统计每次请求输出里的命中缓存token,未命中缓存token,输出token,也通过json格式保存下来。
结果文件保存到 workzone/tmp.json。