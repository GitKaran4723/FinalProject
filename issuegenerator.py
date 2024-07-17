import json
import re
from flask import jsonify
import google.generativeai as g
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configure generative AI with your API key
g.configure(api_key=os.getenv("GENERATIVEAI_API_KEY_1"))

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

def generate_issues(prompt):
    prompt_parts = [
        prompt,
        "Generate JSON code only; do not give any other details.",
        "The JSON data should follow this format:",
        """
        {
            "screen": ["screen", "display"],
            "camera": ["camera", "lens"],
            "battery": ["battery", "charge"]
        }
        """,
        "This is simply a format for how the data should be, but you generate according to the prompt.",
        "Strictly follow the structure only at any cost.",
        "Read the whole text and identify all issues, and give it in the required format.",
        "Generate as many categories as necessary and create the JSON.",
        "All details of issues from the whole text have to be added, and the words have to be from the text.",
        "Do not use categories like positive, neutral, or negative.",
        "Focus on specific issues related to products or services mentioned in the text."
    ]
    
    # Generate the content
    response = model.generate_content(prompt_parts)
    
    # Extract JSON from the response
    issue_keywords = re.sub(r'^```json\n|\n```$', '', response.text)
    
    try:
        # Convert JSON string to dictionary
        issue_keywords_dict = json.loads(issue_keywords)
        
    except json.JSONDecodeError:
        print("Failed to decode JSON response")
        issue_keywords_dict = {}

    return issue_keywords_dict