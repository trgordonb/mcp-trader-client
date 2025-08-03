import gradio as gr
import os
from dotenv import load_dotenv

load_dotenv()

from smolagents import CodeAgent, MCPClient, AzureOpenAIModel

mcp_client = MCPClient({"url": os.getenv("MCP_SERVER_URL"), "transport": "streamable-http"})
try:
    tools = mcp_client.get_tools()

    #model = InferenceClientModel(token=os.getenv("HUGGINGFACE_API_TOKEN"))
    model = AzureOpenAIModel(
        model_id="gpt-4.1",
        api_version="2025-01-01-preview",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
    agent = CodeAgent(
        tools=[*tools], 
        model=model, 
        additional_authorized_imports=[
            "pandas", "numpy", "datetime",
            "matplotlib", "matplotlib.pyplot", "plotly", "seaborn", "sklearn",
            "json", "ast", "urllib", "base64", "requests", "yfinance", 
            "pandas_datareader", "plotly.express", "plotly.graph_objects", "statsmodels"
        ]
    )

    demo = gr.ChatInterface(
        fn=lambda message, history: str(agent.run(message)),
        type="messages",
        title="Agent with Stock Analysis MCP Tools",
        examples=[
            "Analyze the stock AAPL",
            "What is the current RSI of NVDA compared with S&P500?",
            "Plot the closing price of TSLA for last month",
        ],
    )

    demo.launch()
finally:
    mcp_client.disconnect()