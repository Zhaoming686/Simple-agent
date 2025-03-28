
# SimpleAgent: A ReAct-Style Lightweight Language Agent with Tool Calling（轻量级语言智能体）

This project implements a lightweight intelligent agent powered by a small language model (TinyLlama) that can reason and interact with external tools (plugins) to answer user questions.

本项目实现了一个基于小语言模型（TinyLlama）的轻量级智能体，能够进行推理并调用外部工具（插件）来回答用户问题。

It follows the **ReAct (Reasoning + Acting)** paradigm:
该系统遵循 **ReAct（推理 + 行动）** 模式：

> Think → Choose Tool → Use Tool → Observe → Conclude  
> 思考 → 选择工具 → 调用工具 → 观察结果 → 得出结论

---

## Features | 特性

- ✅ Lightweight LLM (TinyLlama) for reasoning and generation  
  轻量级语言模型 TinyLlama 用于推理与文本生成

- ✅ ReAct-style prompt design with structured logic  
  ReAct 风格的提示词设计，结构清晰

- ✅ Modular plugin system for easy tool extension  
  模块化插件系统，便于扩展更多工具

- ✅ Currently supports **Google Search** API  
  当前已接入 **谷歌搜索 API**

---

## Project Structure | 项目结构

```
.
├── agent_demo.ipynb     # Demo notebook 示例入口
├── Agent.py             # 核心逻辑：提示构造 + 工具调用
├── LLM.py               # TinyLlama 模型加载与调用
├── tool.py              # 工具定义与实现
└── README.md            # 项目说明
```

---

## Workflow | 工作流程

1. 用户输入问题  
2. 构造系统提示，描述任务与工具  
3. 模型生成 Thought → Action → Action Input  
4. Agent 解析工具调用并执行  
5. 插入 Observation 工具结果  
6. 模型生成 Final Answer 回答结果

---

## Available Tools | 可用工具

目前支持：

- `google_search`: 调用 Serper.dev 的 Google 搜索接口

模型调用格式如下：

```text
Action: google_search
Action Input: { "search_query": "ChatGPT-5 发布了吗" }
```

---

## Prompt Template | 提示词格式

```
Question: 输入问题  
Thought: 分析思路  
Action: 工具名  
Action Input: JSON 格式的工具输入  
Observation: 工具返回结果  
...  
Final Answer: 最终答案
```

---

## How to Run | 如何运行

### 1. 安装依赖

```bash
pip install transformers torch json5 requests
```

### 2. 下载模型

例如使用 `TinyLlama-1.1B-Chat-v1.0`，或更换你自己的模型路径。

### 3. 运行示例

在 `agent_demo.ipynb` 中运行：

```python
from Agent import Agent

agent = Agent("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
response, history = agent.text_completion("ChatGPT-5 已经发布了吗？")
print(response)
```

---

## Add New Tools | 添加新工具

1. 在 `tool.py` 中添加工具信息和函数  
2. 在 `call_plugin()` 中扩展工具调用逻辑  
3. 更新提示词模板（可选）

---

## Notes | 注意事项

- 当前模型仅运行在 CPU，使用 float32 模式
- 可更换更强语言模型，如 Qwen、Mistral、ChatGPT 等

