from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

# Load .env into a dictionary
env_vars = dotenv_values(".env")

# model and llm definition
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=str(env_vars.get("OPENROUTER_API_KEY")),
    model=str(env_vars.get("LLM_MODEL")),
    temperature=0
)

# prompt definition
prompt = ChatPromptTemplate.from_messages(
    [
    ("system", "You are a linguistic expert. Analyze the mood of input text and translate it. OUTPUT ONLY THE TRANSLATED TEXT. DO NOT PROVIDE ANALYSIS OR EXPLANATIONS."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Translate this to {language}: {text}")
    ]
)

# string output parser
parser = StrOutputParser()

# LCEL
chain = prompt | llm | parser


# make response retain previous response. Making it stateful
memory = []

def translate(input_text, target_language):

    output = chain.invoke({
        "text": input_text,
        "language": target_language,
        "chat_history": memory
    })

    memory.append(HumanMessage(content=input_text))
    memory.append(AIMessage(content=output))

    return output

if __name__ == "__main__":
    # --- Testing the Memory ---
    print("First Run:", translate("I am feeling bad about tomorrow's Monday tonight.", "Marathi"))

    # Now the AI knows what we talked about previously
    print("Second Run:", translate("Now make it more formal.", "Marathi"))