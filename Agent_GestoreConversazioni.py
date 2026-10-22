from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain.agents import create_agent

from typing import List
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Agent_GestoreConversazioni_ENV.env"))
api_key_path = os.getenv("APIKEY_PATH")
folder_path = os.getenv("FOLDER_PATH")

IS_HUMAN_CONVERSATION = False

today = datetime.now().strftime("%Y-%m-%d %H:%M")

class Person_Agent:
    def __init__(self, name: str, llm: ChatOpenAI, system_prompt: str):
        self.name = name
        self.agent = create_agent(model=llm, tools=[], system_prompt=system_prompt)
        self.history: List[BaseMessage] = []
    

def write_file_history(file_name: str, sender_name: str, text: str) -> None:
    with open(f"{folder_path}{file_name}.txt", "a", encoding="utf-8") as f:
        f.write(f"{sender_name}: {text}\n")


def read_file_history(file_name: str, owner: str) -> List[BaseMessage]:
    path = f"{folder_path}{file_name}.txt"
    if not os.path.exists(path):
        return []

    history: List[BaseMessage] = []
    with open(path, "r", encoding="utf-8") as f:
        for row in f:
            row = row.rstrip("\n")
            if not row:
                continue
            sender, sep, text = row.partition(": ")
            if not sep:  # riga senza "sender: " -> la salto
                continue
            if sender == owner:
                history.append(AIMessage(content=text))
            else:
                history.append(HumanMessage(content=text))
    return history


def date_message(message:str):   
    try:
        date = message[:16]
        datetime.strptime(date, "%Y-%m-%d %H:%M")
        return message  # the date is present
    except ValueError:
        # the date is present: add the date
        today = datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"{today} {message}" # Date is not present


with open(api_key_path, "r") as f:
    my_api_key = f.read().strip()

model = "deepseek-chat"
api_base = "https://api.deepseek.com"

llm = ChatOpenAI(model=model, openai_api_key=my_api_key, openai_api_base=api_base)

system_prompt = """
Pretend to be a person with your own personality, hobbies,
tastes, opinions, and interests.

Your goal is to maintain a natural and engaging conversation
with the other person.

Use short sentences. Show interest in the other person by asking
questions, following up on what they say, and sharing your own
experiences and opinions.

If a topic comes to an end, naturally introduce a new one.
You can talk about movies, TV shows, books, travel, video games,
hobbies, music, or other relevant topics.

Avoid repetition and mechanical responses. Do not simply repeat
or rephrase what the other person says.

IMPORTANT RULES:
- Never repeat the same message or greeting.
- If the other person says "See you tomorrow", "Goodbye",
  "Good night", or another farewell, do not reply with
  an identical greeting.
- Keep the conversation going by introducing a new topic
  without ignoring what has already been said.
- If the other person repeats a message they have already sent,
  do not go along with the repetition. Change the subject.
- Consider previous messages to avoid repeating questions,
  answers, and topics that have already been discussed.
- Every message must contribute something new.
- You do not have to ask a question in every message.

Always respond in a way that is consistent with your personality.
"""


tools = []

def run_agent(agent, user_input: str, history: List[BaseMessage]) -> tuple[AIMessage, bool]:
    try:
        result = agent.invoke(
            {"messages": history + [HumanMessage(content=user_input)]},
            config={"recursion_limit": 50}
        )
        print("")
        return result["messages"][-1], True  # success=True
    except Exception as e:
        print(f"Error: {str(e)}")
        return AIMessage(content=f"Error: {str(e)}"), False


if __name__ == "__main__":
    print("INIZIO CHAT")

    conversation_agent = Person_Agent("Adam", llm, system_prompt) # Agent
    human_name = "Human"
    agent_person_name = "Eve"

    if IS_HUMAN_CONVERSATION:
        history1: List[BaseMessage] = read_file_history(human_name, conversation_agent.name)
    else:
        history1: List[BaseMessage] = read_file_history(agent_person_name, conversation_agent.name)
        agent_person = Person_Agent(agent_person_name, llm , system_prompt) # Human simulation
        history2: List[BaseMessage] = read_file_history(agent_person.name, agent_person.name)

    print("========================================================") 
    if history1 :
        last_message = history1[-1].content
        history1 = history1[:-1] # to not repeat the last message
        human_message = last_message
    else :
        if not IS_HUMAN_CONVERSATION:
            # conversazione con agent_person
            human_message = "Hi, what are your plans today?"
            print(f"{agent_person.name}: {date_message(human_message)}")
            write_file_history(agent_person.name, agent_person.name, date_message(human_message))
        else:
            # conversation with human (you)          
            human_message = input("Human: ")
            write_file_history(human_name, human_name, date_message(human_message))

    if not IS_HUMAN_CONVERSATION:
        # conversation_agent
        chat = 0
        while chat < 5:
            chat+=1
            # conversation_agent
            response1, success  = run_agent(conversation_agent.agent, date_message(human_message), history1)  
            print("========================================================")      
            if success:
                print(f"{conversation_agent.name}: {date_message(response1.content)}\n")
                write_file_history(agent_person.name,conversation_agent.name,date_message(response1.content))
                history1 += [HumanMessage(content=date_message(human_message)),AIMessage(content=date_message(response1.content))]
            else:
                print("(message not added to history because of the error)")
                break
            # agent_person
            response2, success  = run_agent(agent_person.agent, response1.content, history2)  
            print("========================================================")      
            if success:
                print(f"{agent_person.name}: {date_message(response2.content)}\n")
                write_file_history(agent_person.name,agent_person.name,date_message(response2.content))
                human_message = response2.content
                history2 += [HumanMessage(content=date_message(response1.content)),AIMessage(content=date_message(response2.content))]
            else:
                print("(message not added to history because of the error)")
                break
        print("========================================================")
    else:
        # Human conversation
        while True:
            # Agent
            response, success  = run_agent(conversation_agent.agent, human_message, history1)  
            print("========================================================")      
            if success:
                print(f"{conversation_agent.name}: {date_message(response.content)}\n")
                write_file_history(human_name,conversation_agent.name,date_message(response.content))
                history1 += [HumanMessage(content=date_message(human_message)),AIMessage(content=date_message(response.content))]
            else:
                print("(message not added to history because of the error)")
                break
            human_message = input("Human: ").strip()
            if not human_message:
                continue
            if human_message.lower() in {"exit", "quit", "q"}:
                print("EXIT")
                break
