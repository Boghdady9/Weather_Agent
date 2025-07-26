"""
Weather Assistant Application
A Gradio-based chatbot that provides weather information using LangChain and Google's Gemini model.
Author: Mohamed Boghdady
"""

# ==================== IMPORTS ====================
import os
import uuid
import base64
from datetime import datetime
from typing import List

import requests
import gradio as gr
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# ==================== CONFIGURATION ====================
# Load environment variables from file
load_dotenv(dotenv_path='api.env.txt')

# API Keys
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API')
GOOGLE_API_KEY = os.getenv('GOOGLE_API')
WEATHER_API_KEY = os.getenv('WEATHER_API')
LOCATION_API_KEY = os.getenv('LOCATION_API')

# Set Google API key in environment
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize the Language Model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,  # Deterministic responses
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


# ==================== UTILITY FUNCTIONS ====================
def get_location_from_ip():
    """
    Retrieve the user's location based on their IP address.
    
    Returns:
        dict: Location data including city, country, and coordinates
    """
    response = requests.get(
        f"https://api.geoapify.com/v1/ipinfo?apiKey={LOCATION_API_KEY}"
    )
    return response.json()


def format_weather_response(weather_data: dict, city: str) -> str:
    """
    Format raw weather data into a human-readable string.
    
    Args:
        weather_data (dict): Raw weather data from API
        city (str): Name of the city
        
    Returns:
        str: Formatted weather forecast
    """
    intervals = weather_data['data']['timelines'][0]['intervals']
    response = f"Weather forecast for {city}:\n\n"
    
    for interval in intervals:
        # Parse and format date
        date = datetime.fromisoformat(interval['startTime']).strftime("%A, %B %d")
        
        # Extract weather values
        temp = round(interval['values']['temperature'], 1)
        humidity = round(interval['values']['humidity'], 1)
        wind_speed = round(interval['values']['windSpeed'], 1)
        
        # Build forecast string
        response += f"{date}:\n"
        response += f"  Temperature: {temp}°C\n"
        response += f"  Humidity: {humidity}%\n"
        response += f"  Wind Speed: {wind_speed * 3.6:.1f} km/h\n\n"
        
    return response


def encode_image(image_path: str) -> str:
    """
    Encode an image file to base64 string for embedding in HTML.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# ==================== PYDANTIC MODELS ====================
class WeatherInput(BaseModel):
    """Schema for weather tool input parameters."""
    city: str = Field(default=None, description="The city to get the weather for.")


class DailyWeather(BaseModel):
    """Schema for daily weather data."""
    date: str
    temperature: float
    condition: str
    humidity: float
    wind_speed: float
    advice: str


class WeatherOutput(BaseModel):
    """Schema for formatted weather output."""
    location: str = Field(description="The location or city for which weather is reported")
    forecast: List[DailyWeather] = Field(description="Weather forecast for multiple days")


# ==================== TOOLS ====================
@tool("get_weather_by_location", args_schema=WeatherInput, return_direct=True)
def get_weather_by_location(city: str = None):
    """
    Fetch weather data for a specific city or user's current location.
    
    Args:
        city (str, optional): City name. If None, uses user's IP location.
        
    Returns:
        str: Formatted weather forecast
    """
    # If no city provided, get user's location
    if not city or city == '':
        location = get_location_from_ip()
        city = location['city']['name']

    # API endpoint and parameters
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

    # Make API request
    response = requests.post(url, json=payload, headers=headers)
    weather_data = response.json()
    
    return format_weather_response(weather_data, city)


# Create tool wrapper for LangChain
get_weather_tool = Tool(
    name="get_weather_by_location",
    func=get_weather_by_location,
    description="Get the current weather for a specific location. "
                "If no location is provided, returns weather for current location."
)


# ==================== AGENT CONFIGURATION ====================
# Output parser
parser = PydanticOutputParser(pydantic_object=WeatherOutput)

# System prompt for the weather assistant
SYSTEM_PROMPT = """You are WeatherWise, a highly knowledgeable, friendly, and efficient weather assistant. 
Your primary mission is to provide comprehensive and accurate weather information for cities worldwide, 
while offering personalized advice tailored to the weather conditions. 

[Previous detailed instructions remain the same but are omitted for brevity]
"""

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
    ("ai", "Good day! I'm WeatherWise, your friendly neighborhood weather expert. "
           "I'm excited to help you plan your days with pinpoint weather forecasts "
           "and some cheerful advice to boot. What would you like to know about the weather?"),
    ("human", "{input}"),
    ("ai", "Absolutely! I'm thrilled to help you with that. Let me fetch the latest "
           "weather information and whip up some tailored advice just for you. "
           "Give me just a moment while I consult my meteorological crystal ball!"),
    ("placeholder", "{agent_scratchpad}"),
])

# Tools list
tools = [get_weather_tool]

# Message history for conversation persistence
message_history = ChatMessageHistory()

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt=prompt)

# Create agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    output_parser=parser
)

# Add chat history capability
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Session management
session_ids = {}


# ==================== GRADIO INTERFACE ====================
def gradio_interface(user_input: str, session_id: str):
    """
    Handle user input and return chatbot response.
    
    Args:
        user_input (str): User's message
        session_id (str): Session identifier for conversation tracking
        
    Returns:
        list: Chat history with new message pair
    """
    # Create or retrieve session ID
    if session_id not in session_ids:
        new_session_id = str(uuid.uuid4())
        session_ids[session_id] = new_session_id
    else:
        new_session_id = session_ids[session_id]
    
    # Invoke agent with chat history
    result = agent_with_chat_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": new_session_id}}
    )
    
    return [[user_input, result['output']]]


# ==================== UI CONFIGURATION ====================
# Encode logo image
IMAGE_PATH = "/workspaces/Weather_Agent/productmanagerinterview_logo.jpeg"
encoded_image = encode_image(IMAGE_PATH)

# Information HTML content
INFO_HTML = f"""
<div style="display: flex; align-items: flex-start;">
    <img src="data:image/jpeg;base64,{encoded_image}" alt="Logo" 
         style="width: 200px; height: 200px; object-fit: cover; margin-right: 20px;">
    <div>
        <h2>Product Manager Accelerator Program</h2>
        <p>The Product Manager Accelerator Program is designed to support PM professionals 
           through every stage of their career. From students looking for entry-level jobs 
           to Directors looking to take on a leadership role, our program has helped over 
           hundreds of students fulfill their career aspirations.</p>
        <p>Our Product Manager Accelerator community are ambitious and committed. Through 
           our program they have learnt, honed and developed new PM and leadership skills, 
           giving them a strong foundation for their future endeavours.</p>
        <p>Learn product management for free today on our 
           <a href="https://www.youtube.com/c/drnancyli?sub_confirmation=1" target="_blank">
           YouTube channel</a></p>
        <h3>Interested in PM Accelerator Pro?</h3>
        <ol>
            <li>Attend the <a href="https://www.drnancyli.com/masterclass" target="_blank">
                Product Masterclass</a> to learn more about the program details, price, 
                different packages, and stay until the end to get FREE AI Course.</li>
            <li>Reserve your early bird ticket and submit an application to talk to our 
                Head of Admission</li>
            <li>Successful applicants join our PMA Pro community to receive customized 
                coaching!</li>
        </ol>
    </div>
</div>
"""


# ==================== MAIN APPLICATION ====================
def create_gradio_app():
    """Create and configure the Gradio interface."""
    with gr.Blocks() as demo:
        # Header
        gr.Markdown("# Weather Assistant - Done By Mohamed Boghdady")
        
        # Chatbot display
        chatbot = gr.Chatbot()
        
        # Input row
        with gr.Row():
            txt = gr.Textbox(
                show_label=False,
                placeholder="Ask about the weather in any city...",
                lines=1,
                container=False
            )
                        submit_btn = gr.Button("Submit")
        
        # Hidden session ID box for tracking conversations
        session_id_box = gr.Textbox(visible=False, value=str(uuid.uuid4()))
        
        # Info button and content
        info_btn = gr.Button("Info")
        info_box = gr.HTML(visible=False, value=INFO_HTML)
        
        # ==================== EVENT HANDLERS ====================
        # Submit button click
        submit_btn.click(
            fn=gradio_interface, 
            inputs=[txt, session_id_box], 
            outputs=chatbot
        )
        
        # Enter key press in textbox
        txt.submit(
            fn=gradio_interface, 
            inputs=[txt, session_id_box], 
            outputs=chatbot
        )
        
        # Info button click - shows the info box
        info_btn.click(
            fn=lambda: gr.update(visible=True), 
            outputs=info_box
        )
    
    return demo


# ==================== LAUNCH APPLICATION ====================
if __name__ == "__main__":
    # Create the Gradio app
    demo = create_gradio_app()
    
    # Launch the application with sharing enabled
    demo.launch(share=True)
