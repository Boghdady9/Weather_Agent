# Weather Assistant

## Overview

Weather Assistant is an interactive chat application that provides real-time weather information and personalized advice based on current and forecasted weather conditions. Built using Python, it leverages the power of large language models and external APIs to deliver accurate, context-aware weather updates for locations worldwide.

## Features

- Real-time weather data retrieval for any specified city
- Automatic location detection for local weather information
- Multi-day weather forecasts
- Personalized advice based on weather conditions
- User-friendly chat interface powered by Gradio
- Integration with Google's Generative AI for natural language processing

## Try it 

You can try the Weather Assistant application by visiting the following link:[Weather Assistant](https://huggingface.co/spaces/BoghdadyJR/Weather_Assistant)


## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7+
- pip (Python package manager)
- API keys for the following services:
  - [Langchain API](https://langchain.com/)
  - [Gemini API](https://aistudio.google.com/app/apikey)
  - [Tomorrow API](https://app.tomorrow.io/)
  - [Geoapify location API](https://www.geoapify.com/)

## API Integration

This application integrates with the following APIs:

- Google Generative AI API for natural language processing
- Tomorrow.io API for weather data
- Geoapify API for IP-based location detection

Ensure you have valid API keys for each service and they are correctly set in the `api.env.txt` file.


## Installation

1. Clone the repository:
   ```
   git clone https://github.com:Boghdady9/Weather_Agent.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a file named `api.env.txt` in the project root directory with the following content:
   ```
   LANGCHAIN_API=your_langchain_api_key
   GOOGLE_API=your_google_api_key
   WEATHER_API=your_weather_api_key
   LOCATION_API=your_location_api_key
   ```
   Replace `your_*_api_key` with your actual API keys.

## Usage

To start the Weather Assistant application, run:

```
python weather_assistant.py
```

This will launch a Gradio interface in your default web browser. You can interact with the Weather Assistant by typing your weather-related queries into the text box.


