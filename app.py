import gradio as gr
import os
from langfuse import get_client
from dotenv import load_dotenv
from smolagents import CodeAgent, MCPClient, AzureOpenAIModel, DuckDuckGoSearchTool
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from fastmcp.server.auth.providers.bearer import RSAKeyPair
from pydantic import SecretStr


load_dotenv()

# Read key pair
with open("private.pem", "r") as private_key_file:
    private_key = private_key_file.read()

with open("public.pem", "r") as public_key_file:
    public_key = public_key_file.read()

# Create key pair for token generation
key_pair = RSAKeyPair(
    private_key=SecretStr(private_key),
    public_key=public_key
)

# Generate JWT token
token = key_pair.create_token(
    subject="user@example.com",
    issuer="https://mcp-server-293232845809.us-central1.run.app",
    audience="my-mcp-server",
    scopes=["read", "write"]
)


mcp_client = MCPClient({"url": os.getenv("MCP_SERVER_URL"), "headers": {"Authorization": f"Bearer {token}"},"transport": "streamable-http"})

css = '''
footer{display:none !important}
#replyimage{
    background-color: coral;
}
'''

langfuse = get_client()

# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")

SmolagentsInstrumentor().instrument()

try:
    tools = mcp_client.get_tools()
    search_tool = DuckDuckGoSearchTool()
    tools.append(search_tool)

    model = AzureOpenAIModel(
        model_id="gpt-4.1-mini",
        api_version="2025-01-01-preview",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
    agent = CodeAgent(
        tools=[*tools], 
        model=model, 
        additional_authorized_imports=[
            "pandas", "numpy", "datetime", "sklearn", "plotly", "plotly.express", "plotly.graph_objects",
            "json", "ast", "urllib", "base64", "requests", "yfinance", "kaleido", "plotly.graph_objs",
            "pandas_datareader.data", "statsmodels", "bs4", "seaborn", "io"
        ],
        instructions="""
        Use the duckduckgo search tool for getting news of the company mentioned in the query.
        Return only news in English in numbered list form, each item in a new line together with the source and its url.
        When requested for chart plotting, use the plotly library and unless otherwise stated in request, plot using line chart type. 
        Remove any null data on the x-axis, for example when the date is a holiday and no price data is available.
        If a plot is produced, first generate the base64 encoded image string and then 
        construct a Markdown string for the chatbot message, using the <img> tag 
        with the src attribute set to a data URI, which includes the Base64 string.
        Return the Markdown string as the response."""
    )

    demo = gr.ChatInterface(
        fn=lambda message, history: str(agent.run(message)),
        type="messages",
        title="Agent with Stock Analysis MCP Tools",
        examples=[
            "Analyze the stock AAPL",
            "What is the current RSI of NVDA compared with S&P500?",
            "Plot the OHLC chart of AAPL for the last week",
            "What are the latest news from Nvidia?",
        ],
        css=css
    )

    demo.launch()
finally:
    mcp_client.disconnect()