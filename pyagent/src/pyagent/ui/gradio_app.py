"""
Gradio Web UI for PyAgent.
"""

import os
from typing import Any

import gradio as gr
from gradio.components.chatbot import ChatMessage

from pyagent.core.agent import Agent, AgentConfig
from pyagent.core.tools import Tool


# Provider registry
PROVIDERS: dict[str, dict[str, Any]] = {}

# Model configurations for each provider
PROVIDER_MODELS: dict[str, list[str]] = {}


def _register_providers():
    """Register available providers."""
    global PROVIDERS, PROVIDER_MODELS

    # Try to import and register Zhipu provider
    try:
        from pyagent.providers.zhipu import ZhipuProvider
        PROVIDERS["zhipu"] = {
            "class": ZhipuProvider,
            "name": "Zhipu GLM",
            "env_key": "ZHIPUAI_API_KEY",
        }
        PROVIDER_MODELS["zhipu"] = ZhipuProvider().get_available_models()
    except ImportError:
        pass

    # Try to import and register OpenAI provider
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

    # Try to import and register Anthropic provider
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


# Demo tools for the agent
def get_weather(location: str) -> str:
    """Get weather for a location (demo)."""
    return f"The weather in {location} is sunny with a temperature of 25°C."


def calculate(expression: str) -> str:
    """Calculate a mathematical expression (demo)."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


DEMO_TOOLS = [
    Tool(
        name="get_weather",
        description="Get the current weather for a location",
        input_schema={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g., Beijing"
                }
            },
            "required": ["location"]
        },
        handler=get_weather
    ),
    Tool(
        name="calculate",
        description="Calculate a mathematical expression",
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to calculate"
                }
            },
            "required": ["expression"]
        },
        handler=calculate
    ),
    Tool(
        name="get_current_time",
        description="Get the current date and time",
        input_schema={
            "type": "object",
            "properties": {},
            "required": []
        },
        handler=get_current_time
    ),
]


def create_agent(
    provider_name: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
) -> Agent:
    """Create an agent instance."""
    if provider_name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider_name}")

    provider_info = PROVIDERS[provider_name]
    provider_class = provider_info["class"]

    # Get API key from parameter or environment
    final_api_key = api_key or os.environ.get(provider_info["env_key"])

    provider = provider_class(api_key=final_api_key)

    # Get default model if not specified
    if not model:
        available_models = PROVIDER_MODELS.get(provider_name, [])
        model = available_models[0] if available_models else "unknown"

    config = AgentConfig(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt if system_prompt.strip() else None,
    )

    return Agent(
        provider=provider,
        config=config,
        tools=DEMO_TOOLS,
    )


# Global agent instance
_agent: Agent | None = None
_current_settings: dict = {}


def respond(
    message: str,
    history: list[ChatMessage],
    provider: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
):
    """Respond to a user message."""
    global _agent, _current_settings

    # Check if settings changed, recreate agent if needed
    new_settings = {
        "provider_name": provider,
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "system_prompt": system_prompt,
    }

    if _agent is None or _current_settings != new_settings:
        _agent = create_agent(**new_settings)
        _current_settings = new_settings.copy()

    # Add user message to history
    history = history + [ChatMessage(role="user", content=message)]

    # Run the agent synchronously (non-streaming for simplicity)
    import asyncio

    async def run_agent():
        return await _agent.run(message)

    try:
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(run_agent())
        finally:
            loop.close()

        # Add assistant response to history
        history = history + [ChatMessage(role="assistant", content=response)]
        yield history

    except Exception as e:
        history = history + [ChatMessage(role="assistant", content=f"Error: {e}")]
        yield history


def clear_history(
    provider: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
) -> tuple[list[ChatMessage], str]:
    """Clear conversation history."""
    global _agent
    if _agent:
        _agent.clear_history()
    return [], "Conversation history cleared."


def update_models(provider: str) -> gr.Dropdown:
    """Update model dropdown based on provider."""
    models = PROVIDER_MODELS.get(provider, [])
    return gr.Dropdown(
        choices=models,
        value=models[0] if models else None,
    )


def create_app(
    title: str = "PyAgent",
    default_provider: str | None = None,
    default_model: str | None = None,
) -> gr.Blocks:
    """Create the Gradio application."""

    # Determine default provider
    if default_provider is None:
        available = list(PROVIDERS.keys())
        default_provider = available[0] if available else "zhipu"

    # Get default models
    default_models = PROVIDER_MODELS.get(default_provider, [])
    if default_model is None:
        default_model = default_models[0] if default_models else None

    with gr.Blocks(title=title) as app:
        gr.Markdown(f"""
        # {title}
        A lightweight AI Agent framework with tool calling support.
        """)

        with gr.Row():
            # Left sidebar - Configuration
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Settings")

                provider_dropdown = gr.Dropdown(
                    choices=list(PROVIDERS.keys()),
                    value=default_provider,
                    label="Provider",
                    info="Select LLM provider",
                )

                model_dropdown = gr.Dropdown(
                    choices=default_models,
                    value=default_model,
                    label="Model",
                    info="Select model to use",
                )

                api_key_input = gr.Textbox(
                    label="API Key",
                    type="password",
                    placeholder="Leave empty to use environment variable",
                    info="Override environment API key",
                )

                temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    info="Higher = more creative",
                )

                max_tokens_slider = gr.Slider(
                    minimum=256,
                    maximum=8192,
                    value=2048,
                    step=256,
                    label="Max Tokens",
                    info="Maximum response length",
                )

                system_prompt_input = gr.Textbox(
                    label="System Prompt",
                    placeholder="You are a helpful assistant...",
                    lines=3,
                    value="You are a helpful AI assistant. You can use tools to help answer questions.",
                )

                clear_btn = gr.Button("Clear History", variant="secondary")

                gr.Markdown("### Available Tools")
                gr.Markdown(
                    "\n".join([
                        f"- **{t.name}**: {t.description}"
                        for t in DEMO_TOOLS
                    ])
                )

            # Main chat area
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Message",
                        placeholder="Type your message here...",
                        scale=4,
                        show_label=False,
                        container=False,
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)

                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    show_label=False,
                    container=False,
                )

        # Event handlers
        provider_dropdown.change(
            fn=update_models,
            inputs=[provider_dropdown],
            outputs=[model_dropdown],
        )

        clear_btn.click(
            fn=clear_history,
            inputs=[
                provider_dropdown,
                model_dropdown,
                api_key_input,
                temperature_slider,
                max_tokens_slider,
                system_prompt_input,
            ],
            outputs=[chatbot, status_output],
        )

        # Chat submission
        msg_input.submit(
            fn=respond,
            inputs=[
                msg_input,
                chatbot,
                provider_dropdown,
                model_dropdown,
                api_key_input,
                temperature_slider,
                max_tokens_slider,
                system_prompt_input,
            ],
            outputs=[chatbot],
        ).then(
            fn=lambda: "",
            inputs=None,
            outputs=[msg_input],
        )

        submit_btn.click(
            fn=respond,
            inputs=[
                msg_input,
                chatbot,
                provider_dropdown,
                model_dropdown,
                api_key_input,
                temperature_slider,
                max_tokens_slider,
                system_prompt_input,
            ],
            outputs=[chatbot],
        ).then(
            fn=lambda: "",
            inputs=None,
            outputs=[msg_input],
        )

    return app


def launch_app(
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    title: str = "PyAgent",
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
