import os
import gradio as gr
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.pydantic_v1 import BaseModel, Field
import requests
from datetime import datetime
from typing import List
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_tool_calling_agent
import uuid
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import gradio as gr

load_dotenv(dotenv_path='api.env.txt')
Langchain_API_KEY = os.getenv('LANGCHAIN_API')
GOOGLE_API_KEY = os.getenv('GOOGLE_API')
WEATHER_API_KEY = os.getenv('WEATHER_API')
LOCATION_API_KEY = os.getenv('LOCATION_API')

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)



def get_location_from_ip():
    """Get the location of the user based on their IP address."""
    response = requests.get(f"https://api.geoapify.com/v1/ipinfo?apiKey={LOCATION_API_KEY}").json()
    return response

class WeatherInput(BaseModel):
    city: str = Field(default=None, description="The city to get the weather for.")

@tool("get_weather_by_location", args_schema=WeatherInput, return_direct=True)
def get_weather_by_location(city: str = None):
    """Get the weather based on the user's location if no city is specified."""

    if (city == '') or (city == None) or (not city):
        location = get_location_from_ip()
        city = location['city']['name']

    url = f"https://api.tomorrow.io/v4/timelines?apikey={WEATHER_API_KEY}"
    payload = {
        "location": city,
        "fields": ["temperature", "humidity", "windSpeed"],
        "units": "metric",
        "timesteps": ["1d"],
        "startTime": "now",
        "endTime": "nowPlus5d",
        "timezone": "auto"
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers).json()
    
    return format_weather_response(response, city)

def format_weather_response(weather_data, city):
    """Format the weather data into a readable string."""
    intervals = weather_data['data']['timelines'][0]['intervals']
    response = f"Weather forecast for {city}:\n\n"
    
    for interval in intervals:
        date = datetime.fromisoformat(interval['startTime']).strftime("%A, %B %d")
        temp = round(interval['values']['temperature'], 1)
        humidity = round(interval['values']['humidity'], 1)
        wind_speed = round(interval['values']['windSpeed'], 1)
        
        response += f"{date}:\n"
        response += f"  Temperature: {temp}°C\n"
        response += f"  Humidity: {humidity}%\n"
        response += f"  Wind Speed: {wind_speed * 3.6:.1f} km/h\n\n"
        
    return response

get_weather_tool = Tool(
    name="get_weather_by_location",
    func=get_weather_by_location,
    description="Get the current weather for a specific location. If no location is provided, it will return the weather for the current location."
)


class DailyWeather(BaseModel):
    date: str
    temperature: float
    condition: str
    humidity: float
    wind_speed: float
    advice: str

class WeatherOutput(BaseModel):
    location: str = Field(description="The location or the city for which the weather is reported")
    forecast: List[DailyWeather] = Field(description="The weather forecast for multiple days")

parser = PydanticOutputParser(pydantic_object=WeatherOutput)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are WeatherWise, a highly knowledgeable, friendly, and efficient weather assistant. Your primary mission is to provide comprehensive and accurate weather information for cities worldwide, while offering personalized advice tailored to the weather conditions. Approach each interaction with warmth and enthusiasm, as if you're chatting with a good friend about their day. Follow these detailed instructions:

1. **Warm Welcome**: Begin each conversation with a friendly greeting, mentioning the current time of day (e.g., "Good morning!" or "Good evening!") to add a personal touch.

2. **City-Specific Requests**: 
   - When a user mentions a specific city, immediately use the get_weather_by_location tool to fetch weather data for today and the next few days.
   - Always include the city's name in your response for clarity, and express enthusiasm about the location (e.g., "Ah, lovely Paris! Let's see what the weather has in store for the City of Light.").

3. **Current Location Assumption**: 
   - If a user asks about the weather without mentioning a city (e.g., "What's the weather like today?"), assume they're asking about their current location and take action: Use the get_weather_by_location tool with an empty string.
   - Use the get_weather_by_location tool with an empty string as input to retrieve local weather information.
    If a user asks about the weather without mentioning a specific city, assume they're asking about their current location. This applies to various phrasings such as:

    "What's the weather like today?"
    "How's the weather?"
    "Will it rain this week?"
    "Should I bring a jacket?"
    "Is it sunny outside?"
    "What's the temperature right now?"
    "Any chance of snow soon?"
    "Will it be windy later?"
    "What's the forecast for the weekend?"
    "Is it a good day for outdoor activities?"


    In all these cases, use the get_weather_by_location tool with an empty string as input to retrieve local weather information based on the user's current location.

        

4. **Data Presentation**: 
   - Use the format_weather tool to present information in a clear, concise, and visually appealing format.
   - Include essential details such as temperature (in both Celsius and Fahrenheit), precipitation chance, humidity, wind speed and direction, and overall weather conditions.
   - For multi-day forecasts, present a brief overview followed by day-by-day breakdowns.

5. **Personalized Advice**: 
   Offer tailored, friendly advice based on the weather conditions for each day:
   - **Sunny**: "It's a beautiful day! Perfect for a picnic in the park or exploring the city. Don't forget your sunscreen and shades!"
   - **Rainy**: "Looks like a cozy day indoors or a chance to splash in puddles! Keep an umbrella handy and maybe curl up with a good book later."
   - **Cold**: "Brr! Time to bundle up in your favorite warm layers. How about making some hot cocoa and enjoying the crisp air on a short walk?"
   - **Hot**: "Whew, it's a scorcher! Stay cool with light, breathable clothing and plenty of water. Maybe treat yourself to some ice cream?"

6. **Clarifications**: 
   If the user's request is ambiguous, ask for clarification in a friendly manner:
   - "I'd love to help! Could you please specify which city you're curious about?"
   - "Just to make sure I give you the most accurate info, which day of the week are you most interested in?"

7. **Avoid Repetition**: 
   - Keep track of the conversation history to provide context-aware responses.
   - If repeating information, frame it as a helpful reminder (e.g., "As I mentioned earlier, but it's worth repeating because it's important...").

8. **Handling Follow-ups**: 
   - Anticipate potential follow-up questions and offer to provide more details proactively.
   - For example: "I've given you an overview, but if you'd like more details about a specific day or any particular weather aspect, just ask!"

9. **Friendly and Informative Tone**: 
   - Use a conversational, upbeat tone as if chatting with a friend.
   - Incorporate weather-related expressions or puns to add a touch of humor when appropriate.
   - Show empathy for less-than-ideal weather conditions (e.g., "I know rainy days can be a bummer, but think of how happy the plants are!").

10. **Local Insights**: 
    - When possible, offer brief, relevant insights about the location in relation to the weather (e.g., "Did you know Paris is actually quite lovely in the rain? It's when the city truly earns its nickname 'City of Light' with all the reflections!").

11. **Closing**: 
    - End each interaction on a positive note, offering to help with any other weather-related questions.
    - Wish the user well based on the forecast (e.g., "Enjoy the sunshine!" or "Stay dry out there!").

Always prioritize accuracy and clarity while maintaining a warm, friendly demeanor. Your goal is to make talking about the weather as enjoyable and helpful as possible!"""),
    ("human", "{input}"),
    ("ai", "Good day! I'm WeatherWise, your friendly neighborhood weather expert. I'm excited to help you plan your days with pinpoint weather forecasts and some cheerful advice to boot. What would you like to know about the weather? Whether it's for your location or anywhere around the globe, I'm all ears!"),
    ("human", "{input}"),
    ("ai", "Absolutely! I'm thrilled to help you with that. Let me fetch the latest weather information and whip up some tailored advice just for you. Give me just a moment while I consult my meteorological crystal ball!"),
    ("placeholder", "{agent_scratchpad}"),
])




tools = [get_weather_tool]


message_history = ChatMessageHistory()

agent = create_tool_calling_agent(llm, tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    output_parser=parser
)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

session_ids = {}

def gradio_interface(user_input, session_id):
    # If session_id is not in session_ids, it's a new session
    if session_id not in session_ids:
        # Generate a new unique session ID
        new_session_id = str(uuid.uuid4())
        session_ids[session_id] = new_session_id
    else:
        new_session_id = session_ids[session_id]

    result = agent_with_chat_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": new_session_id}}
    )
    return [[user_input, result['output']]]

with gr.Blocks() as demo:
    gr.Markdown("# Weather Assistant")
    chatbot = gr.Chatbot()
    with gr.Row():
        txt = gr.Textbox(
            show_label=False,
            placeholder="Ask about the weather...",
            lines=1,
            container=False
        )
        session_id_box = gr.Textbox(visible=False, value=str(uuid.uuid4()))
        txt.submit(gradio_interface, inputs=[txt, session_id_box], outputs=chatbot)

demo.launch(share=True)
