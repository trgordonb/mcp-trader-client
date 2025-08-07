import gradio as gr
import os
from langfuse import get_client
from dotenv import load_dotenv
from smolagents import CodeAgent, MCPClient, AzureOpenAIModel, DuckDuckGoSearchTool, ApiWebSearchTool
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from fastmcp.server.auth.providers.bearer import RSAKeyPair
from pydantic import SecretStr
import logging
import time

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

css = '''
footer{display:none !important}
#replyimage{
    background-color: coral;
}
'''

logger.info("Initialize Langfuse client")
langfuse = get_client()

# Verify connection
if langfuse.auth_check():
    logger.info("Langfuse client is authenticated and ready!")
else:
    logger.error("Authentication failed. Please check your credentials and host.")

logger.info("Initialize Smolagents Instrumentor")
SmolagentsInstrumentor().instrument()

model = AzureOpenAIModel(
    model_id="gpt-4.1-mini",
    api_version="2025-01-01-preview",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
mcp_client = None
last_token_time = None
agent = None
tools = None

def create_token():
    logger.info("Read key pair and create token")
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
        scopes=["read", "write"],
        expires_in_seconds=3600
    )
    logger.info("Token generated")
    token_generated_time = time.time()
    return token, token_generated_time

def get_mcp_client():
    global mcp_client, last_token_time
    logger.info("Get mcp client")
    if not mcp_client or time.time() - last_token_time > 3600:
        token, token_generated_time = create_token()
        logger.info("Token refreshed")
        last_token_time = token_generated_time
        mcp_client = MCPClient({
                    "url": os.getenv("MCP_SERVER_URL"), 
                    "headers": {"Authorization": f"Bearer {token}"},
                    "transport": "streamable-http",
                },
                {"connect_timeout": 60}
            )
    return mcp_client
    
def get_agent():
    mcp_client = get_mcp_client()
    tools = mcp_client.get_tools()
    search_tool = ApiWebSearchTool()
    tool_names = [tool.name for tool in tools]
    if 'web_search' not in tool_names:
        tools.append(search_tool)
    agent = CodeAgent(
        tools=[*tools], 
        model=model, 
        additional_authorized_imports=[
            "pandas", "numpy", "datetime", "sklearn", "matplotlib.pyplot", 
            "json", "ast", "urllib", "base64", "requests", "yfinance", "kaleido", 
            "pandas_datareader.data", "statsmodels", "bs4", "seaborn", "io"
        ],
        instructions="""
        Unless explicitly stated otherwise, disregard your pre-trained knowledge and rely solely on the context provided in this conversation.
        Use the web_search tool only for getting news of the company mentioned in the query.
        If the query is asking you to retrieve info other than listed companies, do not call any tools and say "I cannot answer that question".
        When using the analyze_stock tool, you can pass on the text produced by the tool verbatim as final answer.
        Return only news in English in numbered list form, each item in a new line together with the source and its url.
        Remove any null data on the x-axis, for example when the date is a holiday and no price data is available.
        If a plot is produced, first generate the base64 encoded image string and then 
        construct a Markdown string for the chatbot message, using the <img> tag 
        with the src attribute set to a data URI, which includes the Base64 string.
        Return the Markdown string as the response."""
    )
    logger.info("Agent created")

    return agent
    
def chat_function(message, history):
    logger.info("Chat function called")
    agent = get_agent()
    logger.info("Agent retrieved")
    return str(agent.run(message))

try:
    demo = gr.ChatInterface(
        fn=chat_function,
        type="messages",
        title="Agent with Stock Analysis MCP Tools",
        run_examples_on_click=False,
        examples=[
            "Analyze the stock AAPL",
            "What is the chart pattern of TSLA",
            "Plot the closing price chart of AAPL for the last month",
            "What are the latest headlines from Nvidia?",
        ],
        cache_examples=False,
        css=css
    )

    demo.launch()
finally:
    logger.info("Disconnect mcp client")
    mcp_client.disconnect()