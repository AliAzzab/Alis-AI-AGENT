import os
from dotenv import load_dotenv
from datetime import datetime
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from ddgs import DDGS

# Load environment variables
load_dotenv()

# Initialize the Groq LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-70b-8192"
)

# Wikipedia Tool
wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
wiki_tool = Tool(
    name="Wikipedia",
    func=WikipediaQueryRun(api_wrapper=wiki_api).run,
    description="Useful for answering factual questions using Wikipedia. Always try this for definitions, historical facts, or well-known info."
)

# DuckDuckGo Tool
search_tool = Tool(
    name="DuckDuckGo Search",
    func=DuckDuckGoSearchRun().run,
    description="Use this to search the web for current or general information when Wikipedia isn't enough."
)

# Save-to-File Tool
def save_to_txt(data: str, filename: str = "research_output.txt") -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"\n\n--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted)
    return f"✅ Saved to {filename}"

save_tool = Tool(
    name="Save to File",
    func=save_to_txt,
    description="Save research results or outputs to a local text file. Use this when asked to save anything or record information."
)

# Combine tools
tools = [wiki_tool, search_tool, save_tool]

# 🧠 Custom system message to guide the agent's behavior
system_message = """
You are a smart research assistant powered by Groq's LLaMA 3.
You have access to the following tools:

1. Wikipedia - for factual or historical information.
2. DuckDuckGo Search - for live/current or general info.
3. Save to File - to save summaries or answers.

🚫 Never say "Action: None" or choose an undefined tool.
✅ Always pick one of the tools above.
If you're unsure, default to DuckDuckGo Search.
"""

# Initialize the Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"system_message": system_message}
)

# Run the agent in interactive loop
if __name__ == "__main__":
    print("🤖 Groq-powered Research Assistant Ready!")
    while True:
        try:
            query = input("\nAsk a question (or type 'exit'): ")
            if query.strip().lower() in ["exit", "quit"]:
                break
            response = agent.run(query)
            print("\n📎 Response:\n", response)
        except Exception as e:
            print("❌ Error:", e)