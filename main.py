from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
load_dotenv()

checkpointer = InMemorySaver()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"The weather in {city} is sunny."

model = init_chat_model(
    "gpt-3.5-turbo",
    temperature=0,
)

agent = create_react_agent(
    model=model,
    tools=[get_weather],
    prompt="Never answer questions about the weather.",
    checkpointer=checkpointer
)

# Run the agent
config = {"configurable":{"thread_id":"1"}}
response = agent.invoke(
    {"messages":[{"role":"user", "content":"What is the weather in sf"}]},
    config=config
)
print(response)
my_response = agent.invoke(
    {"messages":[{"role":"user", "content":"What about new york?"}]},
    config=config
)
print(my_response)