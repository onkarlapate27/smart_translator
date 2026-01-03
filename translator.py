from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

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
    ("human", "Translate this to {language}: {text}")
    ]
)

# string output parser
parser = StrOutputParser()

# LCEL
chain = prompt | llm | parser

output = chain.invoke({
    "text": "I am in great mood today. Learning AI is tough and I feel exhausted for that! Also, the weather is lovely.",
    "language": "Hindi"
})


if __name__ == "__main__":
    print(output)