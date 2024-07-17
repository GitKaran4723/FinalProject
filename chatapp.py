
import google.generativeai as g
from dotenv import load_dotenv
import os

load_dotenv()

# Configure generativeAI with your API key
g.configure(api_key=os.getenv("GENERATIVEAI_API_KEY"))

generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 5000,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

model = g.GenerativeModel(model_name="gemini-1.5-pro",
                          generation_config=generation_config,
                          safety_settings=safety_settings)

# Your generativeAI script


def generate_response(prompt):
    prompt_parts = [prompt]
    response = model.generate_content(prompt_parts).text
    return response


def generate_response_with_content(prompt, content):
    prompt_parts = [prompt,
                    "avoid reading the html tag extract the details in in it and explain",
                    "dont consider it as a code give text in the natural language",
                    content]
    response = model.generate_content(prompt_parts).text
    return response
