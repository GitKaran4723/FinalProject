import formcreaterai
import os
import re
from flask import Flask, request, jsonify, send_from_directory
import json
from flask_cors import CORS

# API to save form structure


def save_form_structure():
    form_structure = request.json
    with open('forms/formStructure.json', 'w') as f:
        json.dump(form_structure, f)
    return jsonify({"message": "Form structure saved successfully!"})

# Route to serve static files from the React build


def static_files(path):
    return send_from_directory('frontend/build', path)

# API to save form structure
def save_form_structure(user):
    form_structure = request.json

    filename = form_structure.get('filename')
    form_structure = form_structure.get('formStructure')

    # Define the user-specific path
    path = f"forms/{user}"

    # Create the directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder '{path}' created.")

    # Save the form structure as JSON in the user-specific directory
    file_path = os.path.join(path, f"{filename}.json")
    with open(file_path, 'w') as f:
        json.dump(form_structure, f)
        print("Form structure saved:", form_structure)

    return jsonify({"message": "Form structure saved successfully!"})

# API to get form structure based on prompt
def get_form_structure():
    prompt = request.json.get('prompt')
    # Generate form structure based on the prompt (this is a placeholder example)
    form_structure = formcreaterai.generate_response(prompt)
    cleaned_json_str = re.sub(r'^```json\n|\n```$', '', form_structure)
    cleaned_json_str = json.loads(cleaned_json_str)
    return jsonify(cleaned_json_str)

# display forms
def get_forms(user):
    try:
        FORMS_FOLDER = f'forms/{user}'
        forms = os.listdir(FORMS_FOLDER)
        return jsonify({"forms": forms})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def get_data(user):
    try:
        FORMS_FOLDER = f'dataCollection/{user}'
        datas = os.listdir(FORMS_FOLDER)
        return jsonify({"datas": datas})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def delete_form(user, form_name):
    try:
        FORMS_FOLDER = f'forms/{user}'
        os.remove(os.path.join(FORMS_FOLDER, form_name))
        return jsonify({"message": "Form deleted successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# display form structure
def get_form(user, form_name):
    try:
        FORMS_FOLDER = f'forms/{user}'
        with open(os.path.join(FORMS_FOLDER, form_name), 'r') as f:
            form_data = json.load(f)
        return jsonify(form_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
import os
import csv
import json
from flask import jsonify

def get_data_from_csv(user, form_name):
    try:
        FORMS_FOLDER = f'dataCollection/{user}'
        csv_file_path = os.path.join(FORMS_FOLDER, form_name)

        # Read CSV file
        with open(csv_file_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Convert to JSON
        json_data = json.dumps(rows)

        return jsonify(json.loads(json_data))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def download_csv_file(user, form_name):
    try:
        FORMS_FOLDER = f'dataCollection/{user}'
        file_path = os.path.join(FORMS_FOLDER, form_name)
        return send_from_directory(FORMS_FOLDER, form_name, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def render_form(user, form_name):
    FORMS_FOLDER = f'forms/{user}'
    with open(os.path.join(FORMS_FOLDER, form_name), 'r') as f:
        form_data = json.load(f)
    return form_data
