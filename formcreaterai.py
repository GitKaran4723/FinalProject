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

model = g.GenerativeModel(model_name="gemini-pro",
                          generation_config=generation_config,
                          safety_settings=safety_settings)

# Your generativeAI script
with open('dataformat.txt', 'r') as f:
    data = f.read();

def generate_response(prompt):
    prompt_parts = [
        prompt,
        "generate json code only do not give any other details json data shoould be like below",
        "the following is the data do not give spaces new line characters pure json is required",
        "strictly follow the structure only at any cost",
        data
    ]
    response = model.generate_content(prompt_parts).text
    return response
