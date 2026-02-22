# PyAgent UI 界面增强计划 (方案A - 纯净用户体验)

## 设计理念

**用户无感知后台能力**：Hooks、RAG、Memory、State 作为后台能力自动运行，
用户只需通过简单的界面上传文档、聊天、保存对话即可。

---

## 一、整体布局设计

```
+---------------------------------------------------------------+
|                        PyAgent Pro                            |
|        AI Agent with Knowledge, Memory & Persistence          |
+---------------------------------------------------------------+
|  [Settings Bar - 可折叠]                                       |
|  Provider: [Zhipu]  Model: [glm-4-flash]  API Key: [***]      |
|  Temperature: [0.7]  Max Tokens: [2048]                       |
+---------------------------------------------------------------+
|                                                               |
|                     Chat History                              |
|                                                               |
|                                                               |
+---------------------------------------------------------------+
|  [Upload] [Input message...              ] [Send] [Save]      |
+---------------------------------------------------------------+
|  Uploaded: doc1.pdf (indexed)  [Clear Memory] [Reset]         |
+---------------------------------------------------------------+
```

**核心原则：简洁、直观、无配置**

---

## 二、用户可见功能

| 功能 | UI 入口 | 后台行为 |
|------|--------|---------|
| 上传文档 | 📎 Upload 按钮 | 自动建立 RAG 索引 |
| 聊天问答 | 输入框 + Send | Agent 自动调用 RAG/记忆 |
| 保存对话 | 💾 Save 按钮 | 创建 Checkpoint |
| 清除记忆 | Clear Memory | 清空 Memory + Context |
| 重置会话 | Reset | 清空一切，重新开始 |

---

## 三、后台自动运行的能力

### 3.1 Hooks (自动启用)
- LoggingHook: 记录执行日志 (控制台输出)
- TimingHook: 监控性能，慢操作告警
- ErrorHandlingHook: 自动重试

### 3.2 RAG (按需启用)
- 用户上传文档后自动建立索引
- Agent 自动判断何时使用 RAG 检索
- 支持 PDF、TXT、MD 文件

### 3.3 Memory (自动运行)
- 对话自动存入 ConversationMemory
- 重要信息自动提取到 SemanticMemory
- 事件自动记录到 EpisodicMemory

### 3.4 State (手动触发)
- 用户点击 Save 保存当前状态
- 下次打开可恢复对话

---

## 四、UI 组件设计

### 4.1 顶部设置栏 (可折叠)
```
[Settings v]
  Provider: [Zhipu GLM    v]
  Model:    [glm-4-flash  v]
  API Key:  [************* ]
  Temperature: [----O----] 0.7
  Max Tokens:  [----O----] 2048
```

### 4.2 主聊天区域
```
+-------------------------------------------+
| User: 北京今天天气怎么样？                   |
| Agent: 北京今天是晴天，气温25°C...          |
|                                           |
| User: 帮我总结一下上传的文档                 |
| Agent: 根据文档内容，主要包含...            |
+-------------------------------------------+
```

### 4.3 底部操作栏
```
[📎 Upload] [Type message here...    ] [Send] [💾]
```

### 4.4 状态栏
```
📎 Documents: 2  |  💾 Last saved: 14:30  |  [🗑️ Clear Memory] [🔄 Reset]
```

---

## 五、实现文件结构

```
src/pyagent/ui/
├── __init__.py
├── gradio_app.py          # 主应用 (重构)
└── backend.py             # 后台能力封装
```

---

## 六、核心代码逻辑

### 6.1 Agent 初始化 (自动集成所有能力)

```python
def create_enhanced_agent(settings):
    # 1. 创建 Provider
    provider = create_provider(settings)

    # 2. 创建 Memory Manager
    memory_manager = MemoryManager(
        conversation_memory=ConversationMemory(),
        semantic_memory=SemanticMemory(vectorstore, embedding),
        episodic_memory=EpisodicMemory(vectorstore, embedding),
    )

    # 3. 注册内置 Hooks
    hooks_registry = HookRegistry()
    hooks_registry.register(LoggingHook())
    hooks_registry.register(TimingHook(warn_threshold_ms=3000))
    hooks_registry.register(ErrorHandlingHook(max_retries=3))

    # 4. 创建 Agent
    agent = Agent(
        provider=provider,
        config=config,
        tools=tools + rag_tools,
        hooks_registry=hooks_registry,
    )

    return agent, memory_manager
```

### 6.2 文档上传处理

```python
async def handle_document_upload(files):
    for file in files:
        # 读取文件内容
        content = read_file(file)

        # 创建 Document 并索引
        docs = [Document(content=content, metadata={"source": file.name})]
        await rag_pipeline.index(docs)

    return f"已索引 {len(files)} 个文档"
```

### 6.3 对话处理 (自动使用 RAG + Memory)

```python
async def chat(message, history):
    # 1. 从 Memory 获取上下文
    context = await memory_manager.build_context(message)

    # 2. RAG 检索相关知识
    rag_results = await rag_pipeline.retrieve(message, k=3)

    # 3. 增强 prompt
    enhanced_message = f"""
    相关知识：{rag_results}
    历史记忆：{context}

    用户问题：{message}
    """

    # 4. 调用 Agent
    response = await agent.run(enhanced_message)

    # 5. 自动存储记忆
    await memory_manager.store_experience(message, {"type": "episode"})

    return response
```

---

## 七、演示流程

1. **启动应用**
   ```
   python -m pyagent.ui.gradio_app
   ```

2. **配置 API Key** (或使用环境变量)

3. **上传知识文档** (可选)
   - 点击 Upload，选择 PDF/TXT 文件
   - 自动建立索引

4. **开始对话**
   - Agent 自动使用 RAG 检索知识
   - Agent 自动记忆对话内容

5. **保存对话**
   - 点击 Save 创建 Checkpoint

6. **恢复对话**
   - 刷新页面后可加载之前的 Checkpoint

---

## 八、预期效果

### 用户视角
- 简洁的聊天界面
- 上传文档后 Agent "变聪明了"
- Agent 能记住之前的对话
- 可以保存/恢复对话

### 技术视角 (后台)
- Hooks 自动监控性能
- RAG 自动检索知识
- Memory 自动管理记忆
- State 自动持久化状态

---

## 九、实现步骤

| 步骤 | 内容 |
|------|------|
| 1 | 重构 gradio_app.py 主布局 |
| 2 | 实现 backend.py 后台能力封装 |
| 3 | 实现文档上传和 RAG 索引 |
| 4 | 实现 Memory 自动存储 |
| 5 | 实现 State 保存/加载 |
| 6 | 集成测试 |

---

**此计划已确认，开始实现代码。**
