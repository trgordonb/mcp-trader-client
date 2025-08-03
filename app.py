import gradio as gr
import os
from dotenv import load_dotenv

load_dotenv()

from smolagents import InferenceClientModel, CodeAgent, MCPClient

mcp_client = MCPClient({"url": os.getenv("MCP_SERVER_URL"), "transport": "streamable-http"})

try:
    tools = mcp_client.get_tools()

    model = InferenceClientModel(token=os.getenv("HUGGINGFACE_API_TOKEN"))
    agent = CodeAgent(tools=[*tools], model=model, additional_authorized_imports=["json", "ast", "urllib", "base64"])

    demo = gr.ChatInterface(
        fn=lambda message, history: str(agent.run(message)),
        type="messages",
        title="Agent with MCP Tools",
        description="This is a simple agent that uses MCP tools to answer questions.",
    )

    demo.launch()
finally:
    mcp_client.disconnect()