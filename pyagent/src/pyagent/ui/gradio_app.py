"""
Gradio Web UI for PyAgent Pro.

A clean user interface with backend capabilities:
- Hooks (auto): Logging, Timing, Error Handling
- RAG (auto): Document indexing and retrieval
- Memory (auto): Conversation, Semantic, Episodic memory
- State (manual): Save/Load conversation checkpoints
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import gradio as gr
from gradio.components.chatbot import ChatMessage

from .backend import get_backend, PyAgentBackend


# Provider registry
PROVIDERS: dict[str, dict[str, Any]] = {}
PROVIDER_MODELS: dict[str, list[str]] = {}


def _register_providers():
    """Register available providers."""
    global PROVIDERS, PROVIDER_MODELS

    # Zhipu GLM
    try:
        from pyagent.providers.zhipu import ZhipuProvider
        PROVIDERS["zhipu"] = {
            "class": ZhipuProvider,
            "name": "智谱 GLM",
            "env_key": "ZHIPUAI_API_KEY",
        }
        PROVIDER_MODELS["zhipu"] = ZhipuProvider().get_available_models()
    except ImportError:
        pass

    # OpenAI
    try:
        from pyagent.providers.openai import OpenAIProvider
        PROVIDERS["openai"] = {
            "class": OpenAIProvider,
            "name": "OpenAI",
            "env_key": "OPENAI_API_KEY",
        }
        PROVIDER_MODELS["openai"] = OpenAIProvider().get_available_models()
    except ImportError:
        pass

    # Anthropic
    try:
        from pyagent.providers.anthropic import AnthropicProvider
        PROVIDERS["anthropic"] = {
            "class": AnthropicProvider,
            "name": "Anthropic Claude",
            "env_key": "ANTHROPIC_API_KEY",
        }
        PROVIDER_MODELS["anthropic"] = AnthropicProvider().get_available_models()
    except ImportError:
        pass


_register_providers()


def run_async(coro):
    """Run async function in sync context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def initialize_backend(
    provider: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
) -> str:
    """Initialize the backend with settings."""
    if provider not in PROVIDERS:
        return f"未知的 Provider: {provider}"

    backend = get_backend()
    provider_info = PROVIDERS[provider]
    provider_class = provider_info["class"]

    # Get API key
    final_api_key = api_key or os.environ.get(provider_info["env_key"])
    if not final_api_key:
        return f"请输入 API Key 或设置环境变量 {provider_info['env_key']}"

    # Get default model
    if not model:
        models = PROVIDER_MODELS.get(provider, [])
        model = models[0] if models else "unknown"

    try:
        backend.initialize(
            provider_name=provider,
            provider_class=provider_class,
            api_key=final_api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )
        return f"✅ 已连接 {provider_info['name']} ({model})"
    except Exception as e:
        return f"❌ 初始化失败: {str(e)}"


def handle_upload(files: list, provider: str, model: str, api_key: str,
                  temperature: float, max_tokens: int, system_prompt: str) -> str:
    """Handle document upload."""
    # Ensure backend is initialized
    backend = get_backend()
    if not backend.agent:
        msg = initialize_backend(provider, model, api_key, temperature, max_tokens, system_prompt)
        if "失败" in msg or "请输入" in msg:
            return msg

    if not files:
        return "请选择要上传的文件"

    results = []
    for file in files:
        result = run_async(backend.index_document(file.name, os.path.basename(file.name)))
        results.append(result)

    return "\n".join(results)


def handle_chat(
    message: str,
    history: list[ChatMessage],
    provider: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
):
    """Handle chat message."""
    # Ensure backend is initialized
    backend = get_backend()
    if not backend.agent:
        msg = initialize_backend(provider, model, api_key, temperature, max_tokens, system_prompt)
        if "失败" in msg or "请输入" in msg:
            history = history + [ChatMessage(role="user", content=message)]
            history = history + [ChatMessage(role="assistant", content=msg)]
            yield history
            return

    # Add user message
    history = history + [ChatMessage(role="user", content=message)]

    # Get response
    response = run_async(backend.chat(message))

    # Add assistant response
    history = history + [ChatMessage(role="assistant", content=response)]
    yield history


def handle_save(provider: str, model: str, api_key: str,
                temperature: float, max_tokens: int, system_prompt: str) -> str:
    """Save conversation checkpoint."""
    backend = get_backend()
    if not backend.agent:
        return "请先开始对话"

    return run_async(backend.save_checkpoint())


def handle_load(provider: str, model: str, api_key: str,
                temperature: float, max_tokens: int, system_prompt: str) -> str:
    """Load conversation checkpoint."""
    backend = get_backend()
    return run_async(backend.load_checkpoint())


def handle_clear_memory() -> str:
    """Clear memory."""
    backend = get_backend()
    return backend.clear_memory()


def handle_reset() -> tuple[list[ChatMessage], str, str]:
    """Reset everything."""
    backend = get_backend()
    msg = backend.reset()
    return [], msg, ""


def update_status() -> str:
    """Update status bar."""
    backend = get_backend()
    status = backend.get_status()

    if not status["initialized"]:
        return "⏳ 未连接"

    parts = []
    if status["documents"] > 0:
        parts.append(f"📎 文档: {status['documents']}")
    parts.append(f"🆔 {status['thread_id'][:16]}...")

    return "  |  ".join(parts)


def update_models(provider: str) -> gr.Dropdown:
    """Update model dropdown based on provider."""
    models = PROVIDER_MODELS.get(provider, [])
    return gr.Dropdown(choices=models, value=models[0] if models else None)


def create_app(
    title: str = "PyAgent Pro",
    default_provider: str | None = None,
    default_model: str | None = None,
) -> gr.Blocks:
    """Create the Gradio application."""

    # Determine defaults
    if default_provider is None:
        available = list(PROVIDERS.keys())
        default_provider = available[0] if available else "zhipu"

    default_models = PROVIDER_MODELS.get(default_provider, [])
    if default_model is None:
        default_model = default_models[0] if default_models else None

    with gr.Blocks(
        title=title,
        theme=gr.themes.Soft(),
        css="""
        .status-bar { background: #f8f9fa; padding: 8px 16px; border-radius: 8px; margin-bottom: 10px; }
        .upload-area { min-height: 60px; }
        """
    ) as app:
        # Header
        gr.Markdown(f"""
        # {title}
        **AI Agent** with Knowledge (RAG), Memory & Persistence
        """)

        # Settings (collapsible)
        with gr.Accordion("⚙️ 设置", open=False):
            with gr.Row():
                provider_dropdown = gr.Dropdown(
                    choices=list(PROVIDERS.keys()),
                    value=default_provider,
                    label="Provider",
                    scale=1,
                )
                model_dropdown = gr.Dropdown(
                    choices=default_models,
                    value=default_model,
                    label="Model",
                    scale=1,
                )
                api_key_input = gr.Textbox(
                    label="API Key",
                    type="password",
                    placeholder="或设置环境变量",
                    scale=2,
                )

            with gr.Row():
                temperature_slider = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.7, step=0.1,
                    label="Temperature", scale=1,
                )
                max_tokens_slider = gr.Slider(
                    minimum=256, maximum=8192, value=2048, step=256,
                    label="Max Tokens", scale=1,
                )

            system_prompt_input = gr.Textbox(
                label="System Prompt",
                value="你是一个有帮助的AI助手。你可以使用工具来回答问题，也可以搜索用户上传的知识文档。",
                lines=2,
            )

        # Chat area
        chatbot = gr.Chatbot(
            label="对话",
            height=450,
            show_copy_button=True,
        )

        # Input area
        with gr.Row():
            with gr.Column(scale=1, min_width=100):
                upload_btn = gr.UploadButton(
                    "📎 上传文档",
                    file_types=[".txt", ".md", ".pdf", ".docx", ".py", ".json"],
                    file_count="multiple",
                )
            with gr.Column(scale=5):
                msg_input = gr.Textbox(
                    placeholder="输入消息... (支持上传文档后提问)",
                    show_label=False,
                    container=False,
                )
            with gr.Column(scale=1, min_width=80):
                send_btn = gr.Button("发送", variant="primary")

        # Action buttons
        with gr.Row():
            save_btn = gr.Button("💾 保存会话", variant="secondary")
            load_btn = gr.Button("📂 加载会话", variant="secondary")
            clear_btn = gr.Button("🗑️ 清除记忆", variant="secondary")
            reset_btn = gr.Button("🔄 重置", variant="stop")

        # Status bar
        status_output = gr.Textbox(
            label="",
            value="⏳ 未连接",
            interactive=False,
            show_label=False,
            container=False,
        )

        # Upload status
        upload_status = gr.Textbox(label="", visible=False)

        # Event handlers

        # Provider change
        provider_dropdown.change(
            fn=update_models,
            inputs=[provider_dropdown],
            outputs=[model_dropdown],
        )

        # Upload
        upload_btn.upload(
            fn=handle_upload,
            inputs=[upload_btn, provider_dropdown, model_dropdown, api_key_input,
                    temperature_slider, max_tokens_slider, system_prompt_input],
            outputs=[status_output],
        )

        # Chat submit (Enter)
        msg_input.submit(
            fn=handle_chat,
            inputs=[msg_input, chatbot, provider_dropdown, model_dropdown, api_key_input,
                    temperature_slider, max_tokens_slider, system_prompt_input],
            outputs=[chatbot],
        ).then(
            fn=lambda: "",
            outputs=[msg_input],
        ).then(
            fn=update_status,
            outputs=[status_output],
        )

        # Send button
        send_btn.click(
            fn=handle_chat,
            inputs=[msg_input, chatbot, provider_dropdown, model_dropdown, api_key_input,
                    temperature_slider, max_tokens_slider, system_prompt_input],
            outputs=[chatbot],
        ).then(
            fn=lambda: "",
            outputs=[msg_input],
        ).then(
            fn=update_status,
            outputs=[status_output],
        )

        # Save
        save_btn.click(
            fn=handle_save,
            inputs=[provider_dropdown, model_dropdown, api_key_input,
                    temperature_slider, max_tokens_slider, system_prompt_input],
            outputs=[status_output],
        )

        # Load
        load_btn.click(
            fn=handle_load,
            inputs=[provider_dropdown, model_dropdown, api_key_input,
                    temperature_slider, max_tokens_slider, system_prompt_input],
            outputs=[status_output],
        )

        # Clear memory
        clear_btn.click(
            fn=handle_clear_memory,
            outputs=[status_output],
        )

        # Reset
        reset_btn.click(
            fn=handle_reset,
            outputs=[chatbot, status_output, msg_input],
        )

    return app


def launch_app(
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    title: str = "PyAgent Pro",
    **kwargs,
) -> None:
    """Launch the Gradio application."""
    app = create_app(title=title)
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        **kwargs,
    )


if __name__ == "__main__":
    launch_app()
