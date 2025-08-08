import os
from dotenv import load_dotenv
from datetime import datetime
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun

# Load environment variables
load_dotenv()

# LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-70b-8192"
)

# Tools
wiki_tool = Tool(
    name="Wikipedia",
    func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)).run,
    description="Useful for answering factual questions using Wikipedia."
)

search_tool = Tool(
    name="DuckDuckGo Search",
    func=DuckDuckGoSearchRun().run,
    description="Search the web for current/general information."
)

def save_to_txt(data: str, filename: str = "research_output.txt") -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"\n\n--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted)
    return f"✅ Saved to {filename}"

save_tool = Tool(
    name="Save to File",
    func=save_to_txt,
    description="Save results to a local text file."
)

tools = [wiki_tool, search_tool, save_tool]

# The agent itself
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)