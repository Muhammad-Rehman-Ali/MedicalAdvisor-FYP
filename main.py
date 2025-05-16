"""
Medicine Recommendation System - Flask Application
Refined version with better coding style
"""

from flask import Flask, request, render_template, jsonify, session, redirect, url_for
import numpy as np
import pandas as pd
import pickle
from fuzzywuzzy import process
from datetime import datetime
import traceback
from dotenv import load_dotenv
import os
import warnings

# Initialize Flask app
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'


# Load datasets
def load_data():
    """Load all required datasets"""
    symptoms_severity = pd.read_csv('datasets/Symptom-severity.csv')
    training_data = pd.read_csv('datasets/Training.csv')
    sym_des = pd.read_csv("datasets/symtoms_df.csv")
    precautions = pd.read_csv("datasets/precautions_df.csv")
    workout = pd.read_csv("datasets/workout_df.csv")
    description = pd.read_csv("datasets/description.csv")
    medications = pd.read_csv('datasets/medications.csv')
    diets = pd.read_csv("datasets/diets.csv")

    return {
        'symptoms_severity': symptoms_severity,
        'training_data': training_data,
        'sym_des': sym_des,
        'precautions': precautions,
        'workout': workout,
        'description': description,
        'medications': medications,
        'diets': diets
    }


# Load data and model
data = load_data()
model = pickle.load(open('models/svc.pkl', 'rb'))

# Prepare symptom and disease lists
symptoms_dict = {symptom.lower(): idx for idx, symptom in enumerate(data['symptoms_severity']['Symptom'])}
symptoms_list = sorted(data['symptoms_severity']['Symptom'].unique().tolist())
diseases_list = sorted(data['training_data']['prognosis'].unique().tolist())


def correct_symptom(input_symptom):
    """Correct misspelled symptoms using fuzzy matching"""
    input_symptom = input_symptom.lower().strip()

    if input_symptom in symptoms_dict:
        return input_symptom

    matched = process.extractOne(
        input_symptom,
        symptoms_dict.keys(),
        score_cutoff=70
    )
    return matched[0] if matched else None


def get_recommendations(disease):
    """Get all recommendations for a disease"""
    try:
        clean_disease = disease.strip().lower()

        # Clean disease names in all datasets
        data['description']['Disease'] = data['description']['Disease'].str.strip().str.lower()
        data['precautions']['Disease'] = data['precautions']['Disease'].str.strip().str.lower()
        data['medications']['Disease'] = data['medications']['Disease'].str.strip().str.lower()
        data['diets']['Disease'] = data['diets']['Disease'].str.strip().str.lower()
        data['workout']['disease'] = data['workout']['disease'].str.strip().str.lower()

        # Get description
        matched_desc = data['description'][data['description']['Disease'] == clean_disease]
        desc = matched_desc['Description'].values[0] if not matched_desc.empty else "No description available"

        # Get precautions
        matched_prec = data['precautions'][data['precautions']['Disease'] == clean_disease]
        prec = matched_prec.iloc[0, 1:].dropna().tolist() if not matched_prec.empty else []

        # Get medications
        matched_meds = data['medications'][data['medications']['Disease'] == clean_disease]
        meds = [m.strip("[]'\"") for m in
                matched_meds['Medication'].dropna().tolist()] if not matched_meds.empty else []

        # Get diets
        matched_diet = data['diets'][data['diets']['Disease'] == clean_disease]
        diet = [d.strip("[]'\"") for d in matched_diet['Diet'].dropna().tolist()] if not matched_diet.empty else []

        # Get workouts
        matched_workout = data['workout'][data['workout']['disease'] == clean_disease]
        workouts = [w.strip("[]'\"") for w in
                    matched_workout['workout'].dropna().tolist()] if not matched_workout.empty else []

        return desc, prec, meds, diet, workouts

    except Exception as e:
        app.logger.error(f"Error getting recommendations: {str(e)}")
        return "No description available", [], [], [], []


def log_interaction(user_input, prediction=None):
    """Log user interactions to session"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'input': user_input,
            'prediction': prediction,
            'ip': request.remote_addr
        }

        if 'history' not in session:
            session['history'] = []
        session['history'].append(log_entry)
        session.modified = True
    except Exception as e:
        app.logger.error(f"Error logging interaction: {str(e)}")


@app.route('/')
def index():
    """Home page with recent searches"""
    if request.args.get('reload') == 'true':
        session.pop('prediction_data', None)
        session.modified = True
        return redirect(url_for('index'))

    recent_searches = []
    if 'history' in session:
        recent_searches = list({item['prediction'] for item in session['history']
                                if item['prediction'] in diseases_list})[:5]

    return render_template(
        'index.html',
        symptoms_list=symptoms_list,
        diseases_list=diseases_list,
        recent_searches=recent_searches
    )


@app.route('/clear_session', methods=['POST'])
def clear_session():
    """Clear session data"""
    session.clear()
    return jsonify({'status': 'success'})


@app.route('/predict', methods=['POST'])
def predict():
    """Handle symptom prediction"""
    try:
        symptoms_input = request.form.get('symptoms', '')

        if not symptoms_input:
            return render_template(
                'index.html',
                error="Please select symptoms",
                symptoms_list=symptoms_list,
                diseases_list=diseases_list
            )

        # Process symptoms
        user_symptoms = [s.strip() for s in symptoms_input.split(',') if s.strip()]
        valid_symptoms = []
        invalid_symptoms = []

        for symptom in user_symptoms:
            corrected = correct_symptom(symptom)
            if corrected:
                valid_symptoms.append(corrected)
            else:
                invalid_symptoms.append(symptom)

        if not valid_symptoms:
            return render_template(
                'index.html',
                error="No valid symptoms found",
                symptoms_list=symptoms_list,
                diseases_list=diseases_list
            )

        # Create input vector and predict (with warning suppression)
        input_vector = np.zeros(len(symptoms_dict))
        for symptom in valid_symptoms:
            input_vector[symptoms_dict[symptom]] = 1

        # Suppress sklearn warnings during prediction
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            disease_encoded = model.predict([input_vector])[0]
            predicted_disease = diseases_list[disease_encoded]

        # Get recommendations
        desc, prec, meds, diet, workouts = get_recommendations(predicted_disease)

        # Log interaction
        log_interaction(symptoms_input, predicted_disease)

        # Store in session
        session['prediction_data'] = {
            'prediction': predicted_disease,
            'description': desc,
            'precautions': prec,
            'medications': meds,
            'diets': diet,
            'workouts': workouts,
            'symptoms': valid_symptoms,
            'invalid_symptoms': invalid_symptoms if invalid_symptoms else None,
            'scroll_to_results': True
        }
        session.modified = True

        return redirect(url_for('results'))

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        traceback.print_exc()
        return render_template(
            'index.html',
            error="An error occurred. Please try again.",
            symptoms_list=symptoms_list,
            diseases_list=diseases_list
        )


@app.route('/results')
def results():
    """Show prediction results"""
    if 'prediction_data' not in session:
        return redirect(url_for('index'))

    data = session['prediction_data']
    return render_template(
        'index.html',
        **data,
        symptoms_list=symptoms_list,
        diseases_list=diseases_list
    )


@app.route('/api/search_symptoms')
def search_symptoms():
    """API endpoint for symptom search"""
    query = request.args.get('q', '').lower()

    if not query:
        return jsonify(symptoms_list[:20])

    exact_matches = [s for s in symptoms_list if query in s.lower()]
    fuzzy_matches = process.extract(query, symptoms_list, limit=10)
    fuzzy_matches = [m[0] for m in fuzzy_matches if m[1] > 60 and m[0] not in exact_matches]

    return jsonify(exact_matches + fuzzy_matches[:10])


@app.route('/api/disease_symptoms')
def disease_symptoms_api():
    """API endpoint for disease symptoms"""
    disease = request.args.get('disease')

    if not disease:
        return jsonify({'error': 'Disease parameter is required'}), 400

    try:
        disease_data = data['training_data'][data['training_data']['prognosis'] == disease].iloc[:, :-1]
        symptoms = disease_data.sum().nlargest(5).index.tolist()
        return jsonify({
            'status': 'success',
            'disease': disease,
            'symptoms': symptoms
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/about')
def about():
    """About page"""
    return render_template("about.html")


@app.route('/blog')
def blog():
    """Blog page"""
    return render_template("blog.html")


@app.route('/symptoms')
def symptoms():
    """Symptoms dictionary page"""
    return render_template('symptoms.html')


if __name__ == '__main__':
    app.run(debug=True)
