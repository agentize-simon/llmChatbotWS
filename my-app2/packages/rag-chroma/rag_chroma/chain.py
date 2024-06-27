from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
import requests
import re
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

WEATHER_API_KEY = '9f1906902d12e374d9f1a20f48b24023'
WEATHER_API_URL = 'https://api.openweathermap.org/data/2.5/weather'
GEOCODING_API_URL = 'http://api.openweathermap.org/geo/1.0/direct'

STOCK_API_KEY = '9E0JQT0KQ2UGWVPT'
STOCK_API_URL = 'https://www.alphavantage.co/query'

def getCoor(city_name):
    response = requests.get(f"{GEOCODING_API_URL}?q={city_name}&limit=1&appid={WEATHER_API_KEY}")
    if response.status_code == 200:
        data = response.json()
        if data:
            return data[0]['lat'], data[0]['lon']
        else:
            return None, None
    else:
        return None, None

def getWeather(city_name):
    lat, lon = getCoor(city_name)
    if lat is None or lon is None:
        return "Unable to find coordinates for the specified city."
    
    response = requests.get(f"{WEATHER_API_URL}?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric")
    if response.status_code == 200:
        data = response.json()
        print(data)
        return f"The current weather in {city_name} is {data['weather'][0]['description']} with a temperature of {data['main']['temp']}°C."
    else:
        return "Unable to fetch weather data."

def getPrice(symbol):
    response = requests.get(f"{STOCK_API_URL}?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={STOCK_API_KEY}")
    if response.status_code == 200:
        data = response.json()
        try:
            time_series = data['Time Series (Daily)']
            latest_date = next(iter(time_series))
            latest_data = time_series[latest_date]
            return (f"Stock price for {symbol} on {latest_date}: "
                    f"Open: {latest_data['1. open']}, "
                    f"High: {latest_data['2. high']}, "
                    f"Low: {latest_data['3. low']}, "
                    f"Close: {latest_data['4. close']}, "
                    f"Volume: {latest_data['5. volume']}")
        except KeyError:
            return "Unable to fetch stock data."
    else:
        return "Unable to fetch stock data."

model = ChatOpenAI()
# memory = ConversationBufferMemory()

# memory.chat_memory.add_user_message("hi!")
# memory.chat_memory.add_ai_message("what's up?")
# memory.load_memory_variables({})


route_chain = (
    PromptTemplate.from_template(
        """Given the user question below, classify it as either being about `Weather`, `Stock Price`, or `Other`.

Do not respond with more than one words.

<question>
{question}
</question>

Classification:"""
    )
    | model
    | StrOutputParser()
)

city_chain = (
    PromptTemplate.from_template(
        """Only return the city name in the question

<question>
{question}
</question>
"""
    )
    | model
    | StrOutputParser()
)

symbol_chain = (
    PromptTemplate.from_template(
        """Only return the stock symbol in the question

<question>
{question}
</question>
"""
    )
    | model
    | StrOutputParser()
)
weather_chain = (
    city_chain
    | RunnableLambda(lambda city: getWeather(city))
)

stock_chain = (
    symbol_chain
    | RunnableLambda(lambda symbol: getPrice(symbol))
)
# weather_chain = (PromptTemplate.from_template(
#     """You are an expert in weather report. \
# You will tell the user the detail of the weather today \
# The city will be in the question\
# Respond to the following question:

# Question: {question}
# Answer:"""
# ) | model | StrOutputParser())

# stock_chain = (PromptTemplate.from_template(
#     """You are an expert in stock price report. \
# You will tell the user the detail of the stock price today \
# The stock symbol will be in the question\
# Respond to the following question:

# Question: {question}
# Answer:"""
# ) | model | StrOutputParser() )

general_chain = (PromptTemplate.from_template(
    """Respond the following sentense only 'Sorry, I can only answer question about weather or stock price' """
) | model | StrOutputParser())

def route(info):
    print(info)
    print(re.search(r'\b\w+\b', info["topic"]).group(0))
    if "weather" in info["topic"].lower():
        return weather_chain
    elif "stock" in info["topic"].lower():
        return stock_chain
    else:
        return general_chain
    

chain = {"topic": route_chain, "question": lambda x: x["question"]} | RunnableLambda(
    route
) 
#getWeather("Toronto")
chain.invoke({"question": "weather in toronto?"})
chain.invoke({"question": "stock price of AAPL?"})

# # RAG prompt
# template = """Answer the question based only on the following context:
# {context}

# Question: {question}
# """
# prompt = ChatPromptTemplate.from_template(template)

# # LLM
# model = ChatOpenAI()

# # RAG chain
# # chain = (
# #     RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
# #     | prompt
# #     | model
# #     | StrOutputParser()
# # )

# # Add typing for input
# class Question(BaseModel):
#     question: str

# def getCityName(question):
#     match = re.search(r'\b(?:in|at|for|of)\s+([A-Za-z\s]+)', question)
#     if match:
#         return match.group(1).strip()
#     return None

# def getSymbol(question):
#     match = re.search(r'\b(?:symbol|stock)\s+([A-Z]+)', question)
#     if match:
#         return match.group(1).strip()
#     match = re.search(r'\b[A-Z]{1,5}\b', question)
#     if match:
#         return match.group(0).strip()
#     return None

# def route_question(question: str):
#     city = getCityName(question)
#     symbol = getSymbol(question)
    
#     if "weather" in question.lower() and city:
#         return getWeather(city)
#     elif "stock" in question.lower() and symbol:
#         return getPrice(symbol)
#     else:
#         return "I can only answer questions about today's weather and stock prices."

# class RoutedResponse(BaseModel):
#     __root__: str

# chain = (
#     RunnablePassthrough()
#     | (lambda question: route_question(question['question']))
#     | StrOutputParser()
# )
# chain = chain.with_types(input_type=Question, output_type=RoutedResponse)
