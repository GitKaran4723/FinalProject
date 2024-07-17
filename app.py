from functools import wraps
import json
import os
import re
import pandas as pd
from flask import Flask, abort, jsonify, render_template, redirect, send_from_directory, url_for, request, session, flash
from flask_jwt_extended import JWTManager, create_access_token
import chatapp
from dbcollections.contact import Contact
import db
from dbcollections.User import User
import dataCollection
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import joblib
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['JWT_SECRET_KEY'] = 'your_jwt_secret_key'
jwt = JWTManager(app)

UPLOAD_FOLDER = 'uploads'
TREND_UPLOAD_FOLDER = 'trend_uploads'
MODELS_FOLDER = 'models'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(TREND_UPLOAD_FOLDER, exist_ok=True)
app.config['TREND_UPLOAD_FOLDER'] = TREND_UPLOAD_FOLDER

# if you feel vader_lexicon is not downloaded, uncomment the below line
# nltk.download('vader_lexicon')
nlp = spacy.load('en_core_web_sm')

issue_keywords = {
    'screen': ['screen', 'display'],
    'camera': ['camera', 'lens'],
    'battery': ['battery', 'charge'],
}

# Authentication check decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'is_logged_in' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route('/')
def index():
    return render_template('index.html')

# sentiment analysis module routes
@app.route('/sentimentAnalysis')
@login_required
def sentiment_analysis():
    return render_template('sentiment_analysis.html')

import issuegenerator
@app.route('/sentimentAnalysis/preprocessor', methods=['GET', 'POST'])
@login_required
def sentiment_analysis_preprocessor():
    new_columns = None
    file_name = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_name = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            df = pd.read_csv(filepath)

            if 'feedback' not in df.columns:
                flash('Please upload a file with a "feedback" column.', 'error')
                return redirect(url_for('sentiment_analysis_preprocessor'))

            # Ensure no missing values
            for column in df.columns:
                if df[column].dtype == 'object':
                    # Fill missing values in categorical columns with mode
                    df[column].fillna(df[column].mode()[0], inplace=True)
                else:
                    # Fill missing values in numerical columns with mean
                    df[column].fillna(df[column].mean(), inplace=True)

            # Sentiment Analysis
            sid = SentimentIntensityAnalyzer()
            df['sentiment'] = df['feedback'].apply(lambda x: sid.polarity_scores(x)['compound'])

            # Named Entity Recognition (NER)
            df['entities'] = df['feedback'].apply(lambda x: [ent.text for ent in nlp(x).ents])

            # Assuming df is your DataFrame
            all_feedback = ' '.join(df['feedback'].tolist())
            issue_keywords = issuegenerator.generate_issues(all_feedback)

            # Issue Identification
            def identify_issues(feedback):
                issues = []
                for issue, keywords in issue_keywords.items():
                    if any(keyword in feedback.lower() for keyword in keywords):
                        issues.append(issue)
                return issues

            df['issues'] = df['feedback'].apply(identify_issues)

            new_columns = df.columns.tolist()
            processed_filepath = os.path.join(PROCESSED_FOLDER, 'processed_' + file.filename)
            df.to_csv(processed_filepath, index=False)

            return render_template('sentiment_analysis/preprocessor.html', new_columns=new_columns, file_name=file_name)

    return render_template('sentiment_analysis/preprocessor.html', new_columns=new_columns, file_name=file_name)


@app.route('/sentimentAnalysis/analysis', methods=['GET', 'POST'])
@login_required
def sentiment_analysis_analysis():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.startswith('processed_'):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            df = pd.read_csv(filepath)

            required_columns = ['Feedback_ID', 'Customer_Name', 'Rating', 'feedback', 'Date', 'Response_Status', 'sentiment', 'entities', 'issues']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                flash(f'Missing columns: {", ".join(missing_columns)}', 'error')
                return redirect(url_for('sentiment_analysis_analysis'))

            # Convert entities and issues columns to lists
            df['entities'] = df['entities'].apply(lambda x: eval(x) if isinstance(x, str) else x)
            df['issues'] = df['issues'].apply(lambda x: eval(x) if isinstance(x, str) else x)

            # Sentiment Analysis
            sentiments = {
                'positive': int((df['sentiment'] > 0.1).sum()),
                'neutral': int(((df['sentiment'] >= -0.1) & (df['sentiment'] <= 0.1)).sum()),
                'negative': int((df['sentiment'] < -0.1).sum())
            }

            # Issue Identification
            issues = df['issues'].explode().value_counts().to_dict()
            issues = {k: int(v) for k, v in issues.items()}

            # Customer Satisfaction
            satisfaction = {
                'very_satisfied': int((df['Rating'] == 5).sum()),
                'satisfied': int((df['Rating'] == 4).sum()),
                'neutral': int((df['Rating'] == 3).sum()),
                'dissatisfied': int((df['Rating'] == 2).sum()),
                'very_dissatisfied': int((df['Rating'] == 1).sum())
            }

            analysis_data = {
                'sentiments': sentiments,
                'issues': issues,
                'satisfaction': satisfaction
            }

            # Store feedback data in the session as well
            feedback_data = df.to_dict(orient='records')
            session['analysis_data'] = analysis_data
            session['feedback_data'] = feedback_data

            return redirect(url_for('analysis_results'))
        else:
            flash('Please upload a preprocessed file.', 'error')
            return redirect(url_for('sentiment_analysis_analysis'))

    return render_template('sentiment_analysis/analysis.html')

@app.route('/sentimentAnalysis/results')
@login_required
def analysis_results():
    analysis_data = session.get('analysis_data')
    feedback_data = session.get('feedback_data')
    if not analysis_data or not feedback_data:
        flash('No analysis data found. Please upload and analyze a file first.', 'error')
        return redirect(url_for('sentiment_analysis_analysis'))
    return render_template('sentiment_analysis/results.html', analysis_data=analysis_data, feedback_data=feedback_data)

@app.route('/uploads/<filename>')
def download_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

# trend detection module routes
import trend_analysis
@app.route('/trend_detection')
@login_required
def trend_detection():
    return render_template('trend_detection.html')

@app.route('/trend', methods=['GET', 'POST'])
def trend_preprocess():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            data, new_columns = trend_analysis.preprocess_data(file)
            session['data'] = data.to_json(orient='split')
            session['file_name'] = file.filename
            processed_file_path = os.path.join(app.config['TREND_UPLOAD_FOLDER'], 'processed_' + file.filename)
            data.to_csv(processed_file_path, index=False)
            session['processed_file_path'] = processed_file_path
            return render_template('trend/preprocessor.html', new_columns=new_columns, file_name=file.filename)
    return render_template('trend/preprocessor.html')

@app.route('/trend/select_columns', methods=['GET', 'POST'])
def trend_select_columns():
    data = pd.read_json(session['data'], orient='split')
    if request.method == 'POST':
        selected_columns = request.form.getlist('columns')
        session['selected_columns'] = selected_columns
        return redirect(url_for('trend_results'))
    return render_template('trend/select_columns.html', columns=data.columns.tolist())

@app.route('/trend/results', methods=['GET'])
def trend_results():
    selected_columns = session['selected_columns']
    data = pd.read_json(session['data'], orient='split')
    trend_data = trend_analysis.detect_trends(data, selected_columns)
    return render_template('trend/results.html', data=trend_data.to_dict(orient='records'))

@app.route('/trend_uploads/<filename>')
def download_file_trend(filename):
    return send_from_directory(app.config['TREND_UPLOAD_FOLDER'], filename)


# customer segmentation module routes
import segmentationFun
@app.route('/segmentation')
@login_required
def segmentation():
    return render_template('segmentation.html')

@app.route('/segmentation/preprocess', methods=['GET', 'POST'])
def segmentation_preprocess():
    print("segmentation_preprocess")
    if request.method == 'POST':
        file = request.files['file']
        if file:
            data, error = segmentationFun.preprocess_data(file)
            if error:
                flash(error, 'danger')
                return redirect(url_for('segmentation_preprocess'))
            session['data'] = data.to_json(orient='split')
            session['file_name'] = file.filename
            processed_file_path = os.path.join(UPLOAD_FOLDER, 'processed_' + file.filename)
            data.to_csv(processed_file_path, index=False)
            session['processed_file_path'] = processed_file_path
            return render_template('segmentation/preprocessor.html', new_columns=data.columns.tolist(), file_name=file.filename)
    return render_template('segmentation/preprocessor.html')

@app.route('/segmentation/select_columns', methods=['GET', 'POST'])
def segmentation_select_columns():
    data = pd.read_json(StringIO(session['data']), orient='split')
    if request.method == 'POST':
        selected_columns = request.form.getlist('columns')  # Get selected columns from form
        session['selected_columns'] = selected_columns  # Store selected columns temporarily

        # Dynamically fit and save scaler and model based on selected columns
        customer_data = data[selected_columns].select_dtypes(include=[float, int]).dropna()
        
        scaler = StandardScaler()
        customer_data_scaled = scaler.fit_transform(customer_data)
        
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(customer_data_scaled)

        # Save the model and scaler
        joblib.dump(scaler, os.path.join(MODELS_FOLDER, 'segmentation_scaler.pkl'))
        joblib.dump(kmeans, os.path.join(MODELS_FOLDER, 'segmentation_kmeans_model.pkl'))

        return redirect(url_for('segmentation_results'))
    return render_template('segmentation/select_columns.html', columns=data.columns.tolist())

@app.route('/segmentation/results', methods=['GET'])
def segmentation_results():
    selected_columns = session['selected_columns']
    data = pd.read_json(StringIO(session['data']), orient='split')

    print(selected_columns)

    # Process data using selected columns
    customer_data = data[selected_columns].select_dtypes(include=[float, int]).dropna()
    
    # Load dynamically created scaler and model
    scaler = joblib.load(os.path.join(MODELS_FOLDER, 'segmentation_scaler.pkl'))
    kmeans = joblib.load(os.path.join(MODELS_FOLDER, 'segmentation_kmeans_model.pkl'))
    
    # Normalize the data
    customer_data_scaled = scaler.transform(customer_data)
    
    # Apply K-means clustering
    customer_data['Cluster'] = kmeans.predict(customer_data_scaled)

    print(customer_data.to_dict(orient='records'))

    return render_template('segmentation/results.html', data=customer_data.to_dict(orient='records'))

# data collection module routes
@app.route('/data_collection')
@login_required
def data_collection():
    return render_template('data_collection.html')

@app.route('/form_generator')
@login_required
def form_generator():
    return render_template('form_generator.html')


@app.route('/my_forms')
@login_required
def my_forms():
    user = session['user']
    return render_template('my_forms.html', user=user)

@app.route('/my_datas')
@login_required
def my_datas():
    user = session['user']
    return render_template('my_data.html', user=user)

@app.route('/api/get-forms', methods=['GET'])
def get_forms():
    user = session['user']
    print(dataCollection.get_forms(user))
    return dataCollection.get_forms(user)

@app.route('/api/get-datas', methods=['GET'])
def get_data():
    user = session['user']
    return dataCollection.get_data(user)

@app.route('/api/get-form/<form_name>', methods=['GET'])
def get_form(form_name):
    user = session['user']
    return dataCollection.get_form(user, form_name)

@app.route('/api/get-data/<file_name>', methods=['GET'])
def get_data_from_file(file_name):
    user = session['user']
    return dataCollection.get_data_from_csv(user, file_name)

@app.route('/api/download-file/<file_name>', methods=['GET'])  
def download_csv_file(file_name):
    user = session['user']
    return dataCollection.download_csv_file(user, file_name)

@app.route('/api/delete-form/<form_name>', methods=['DELETE'])
def delete_form(form_name):
    user = session['user']
    return dataCollection.delete_form(user, form_name)

@app.route('/form/<user_name>/<form_name>', methods=['GET'])
def serve_form(user_name, form_name):
    form_data = dataCollection.render_form(user_name, form_name)
    if form_data:
        return render_template('dynamic_form.html', form_data=form_data, user_name=user_name , form_name=form_name)
    else:
        abort(404, description="Form not found")

import csv
@app.route('/submit-form', methods=['POST'])
def submit_form():
    user_name = request.form.get('user_name')
    form_name = request.form.get('form_name')

    # Extract the data from the form
    form_data = {key: value for key, value in request.form.items() if key not in ['user_name', 'form_name']}
    
    # Ensure the directory exists
    user_directory = os.path.join('dataCollection', user_name)
    if not os.path.exists(user_directory):
        os.makedirs(user_directory)
    
    # Define the CSV file path
    form_name = form_name.split(".")[0]
    print(form_name)

    csv_file_path = os.path.join(user_directory, f"{form_name}.csv")
    
    # Write to the CSV file
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=form_data.keys())
        
        # Write the header if the file doesn't exist
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(form_data)
    
    return render_template('thank_you.html')

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirmPassword']
        company = request.form['company']

        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('signup.html')

        existing_user = User.get_user_by_email(email)
        if existing_user:
            flash('User already exists', 'error')
            return render_template('signup.html')

        new_user = User(email, password, company)
        try:
            new_user.save()
            flash('Signup successful! Redirecting to login...', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash(str(e), 'error')
            return render_template('signup.html')

    return render_template('signup.html')


@app.route('/about')
def about_us():
    return render_template('about_us.html')


@app.route('/features')
def features():
    features = [
        {
            'title': 'Advanced Data Analytics',
            'description': 'Our AI algorithms analyze customer data to provide valuable insights and trends, helping businesses make informed decisions.',
            'color': '#007bff'  # Blue
        },
        {
            'title': 'Customer Segmentation',
            'description': 'Automatically segment customers based on behavior, preferences, and demographics for targeted marketing and personalized experiences.',
            'color': '#28a745'  # Green
        },
        {
            'title': 'Real-Time Insights',
            'description': 'Access real-time data and analytics to stay ahead of market trends and respond to customer needs promptly.',
            'color': '#ffc107'  # Yellow
        },
        {
            'title': 'Custom Reports',
            'description': 'Generate custom reports tailored to your business needs, providing clear and actionable insights.',
            'color': '#17a2b8'  # Teal
        },
        {
            'title': 'Integration Capabilities',
            'description': 'Seamlessly integrate our service with your existing systems and tools for a smooth workflow.',
            'color': '#dc3545'  # Red
        }
    ]
    return render_template('features.html', features=features)


@app.route('/pricing')
def pricing():
    return render_template('pricing.html')


@app.route('/faq')
def faq():
    return render_template('faq.html')


@app.route('/contact')
def contact_us():
    return render_template('contact_us.html')


@app.route('/contactus', methods=['POST'])
def add_contact():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    message = data.get('message')

    if not name or not email or not message:
        return jsonify({"error": "Name, email, and message are required"}), 400

    contact = Contact(name=name, email=email, message=message)

    try:
        contact.save()
        flash('Thank you for your message. We will get back to you soon!', 'success')
        return jsonify({"message": "Message submitted successfully"}), 201
    except Exception as e:
        flash('An error occurred. Please try again.', 'error')
        return jsonify({"error": str(e)}), 500


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.get_user_by_email(email)
        if not user:
            flash('User does not exist', 'error')
            return render_template('login.html')
        if not user.check_password(password):
            flash('Invalid password', 'error')
            return render_template('login.html')

        access_token = create_access_token(
            identity={'email': user.email, 'company': user.company})

        session['is_logged_in'] = True
        session['user'] = user.email
        session['company'] = user.company
        session['access_token'] = access_token

        return redirect(url_for('dashboard'))

    return render_template('login.html')

from datetime import datetime
@app.route('/save_chat', methods=['POST'])
def save_chat():
    
    data = request.get_json()
    

    # Generate filename with current date and time
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Check if username is in session
    if 'username' in session:
        filename = f"{session['username']}_chat_{timestamp}.json"
    else:
        filename = f"chat_{timestamp}.json"

    # Ensure the directory exists
    if not os.path.exists('saved_chats'):
        os.makedirs('saved_chats')

    # Save data to a file on the server
    with open(os.path.join('saved_chats', filename), 'w') as file:
        json.dump(data, file)

    return jsonify({"message": "Chat saved successfully!", "filename": filename})

@app.route('/download_chat/<filename>', methods=['GET'])
def download_chat(filename):
    
    return send_from_directory('saved_chats', filename, as_attachment=True)


@app.route('/logout')
def logout():
    session.pop('is_logged_in', None)
    session.pop('user', None)
    session.pop('company', None)
    session.pop('access_token', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/dashboard')
def dashboard():
    if not session.get('is_logged_in'):
        return redirect(url_for('login'))

    return render_template('dashboard.html', company=session.get('company'), token=session.get('access_token'))

# Data collection module


@app.route('/get-form-structure', methods=['POST'])
def get_form_structure():
    return dataCollection.get_form_structure()


@app.route('/save-form-structure', methods=['POST'])
def save_form_structure():
    user = session['user']
    print('user', user)
    return dataCollection.save_form_structure(user)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    prompt = data['prompt']

    if 'content' in data:
        content = data['content']
        # Pass the content along with the prompt to your chatapp function if needed
        response = chatapp.generate_response_with_content(prompt, content)
    else:
        response = chatapp.generate_response(prompt)

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
