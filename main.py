from flask import Flask, request, render_template, jsonify, session, redirect, url_for
import numpy as np
import pandas as pd
import pickle
from fuzzywuzzy import process
from datetime import datetime
import traceback
import secrets
import os
from mangum import Mangum

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Load datasets
try:
    symptoms_severity = pd.read_csv('datasets/Symptom-severity.csv')
    precautions = pd.read_csv('datasets/precautions_df.csv')
    workout = pd.read_csv('datasets/workout_df.csv')
    description = pd.read_csv('datasets/description.csv')
    medications = pd.read_csv('datasets/medications.csv')
    diets = pd.read_csv('datasets/diets.csv')
    training_data = pd.read_csv('datasets/Training.csv')
except Exception as e:
    print(f"Error loading datasets: {str(e)}")
    raise

# Create mappings
symptoms_dict = {symptom.lower(): idx for idx, symptom in enumerate(symptoms_severity['Symptom'])}
symptoms_list = sorted(symptoms_severity['Symptom'].unique().tolist())
diseases_list = sorted(training_data['prognosis'].unique().tolist())

# Load model
try:
    model = pickle.load(open('models/svc.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise


def log_interaction(user_input, prediction=None):
    """Log user interactions"""
    try:
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'input': user_input,
            'prediction': prediction,
            'ip': request.remote_addr
        }

        if 'history' not in session:
            session['history'] = []
        session['history'].append(log_entry)
        session.modified = True
    except Exception as e:
        print(f"Error logging interaction: {str(e)}")


def correct_symptom(input_symptom):
    """Correct misspelled symptoms"""
    input_symptom = input_symptom.lower().strip()
    if input_symptom in symptoms_dict:
        return input_symptom
    matched = process.extractOne(input_symptom, symptoms_dict.keys(), score_cutoff=70)
    return matched[0] if matched else None


def get_recommendations(disease):
    """Get all recommendations for a disease"""
    try:
        desc = description[description['Disease'] == disease]['Description'].values[0]
        prec = precautions[precautions['Disease'] == disease].iloc[0, 1:].dropna().tolist()
        meds = [m.strip("[]'\"") for m in
                medications[medications['Disease'] == disease]['Medication'].dropna().tolist()]
        diet = [d.strip("[]'\"") for d in diets[diets['Disease'] == disease]['Diet'].dropna().tolist()]
        workouts = [w.strip("[]'\"") for w in workout[workout['disease'] == disease]['workout'].dropna().tolist()]
        return desc, prec, meds, diet, workouts
    except Exception as e:
        print(f"Error getting recommendations for {disease}: {str(e)}")
        return "No description available", [], [], [], []


@app.route('/')
def index():
    """Home page with recent searches"""
    # Clear prediction data when page is reloaded
    if request.args.get('reload') == 'true':
        if 'prediction_data' in session:
            session.pop('prediction_data', None)
            session.modified = True
        return redirect(url_for('index'))

    recent_searches = []
    if 'history' in session:
        recent_searches = list(
            {item['prediction'] for item in session['history'] if item['prediction'] in diseases_list})
    return render_template('index.html', symptoms_list=symptoms_list, diseases_list=diseases_list,
                           recent_searches=recent_searches[:5])


@app.route('/clear_session', methods=['POST'])
def clear_session():
    """Clear session data"""
    session.clear()
    return jsonify({'status': 'success'})


@app.route('/predict', methods=['POST'])
def predict():
    """Handle symptom prediction"""
    try:
        symptoms = request.form.get('symptoms', '')
        print(f"Received symptoms: {symptoms}")

        if not symptoms:
            return render_template('index.html',
                                   error="Please select symptoms",
                                   symptoms_list=symptoms_list,
                                   diseases_list=diseases_list)

        # Process symptoms with validation
        user_symptoms = [s.strip() for s in symptoms.split(',') if s.strip()]
        print(f"Processed symptoms: {user_symptoms}")

        valid_symptoms = []
        invalid_symptoms = []

        for symptom in user_symptoms:
            corrected = correct_symptom(symptom)
            if corrected:
                valid_symptoms.append(corrected)
            else:
                invalid_symptoms.append(symptom)

        if not valid_symptoms:
            return render_template('index.html', error="No valid symptoms found", symptoms_list=symptoms_list,
                                   diseases_list=diseases_list)

        # Create input vector
        input_vector = np.zeros(len(symptoms_dict))
        for symptom in valid_symptoms:
            input_vector[symptoms_dict[symptom]] = 1

        # Get prediction
        disease_encoded = model.predict([input_vector])[0]
        predicted_disease = diseases_list[disease_encoded]

        # Special cases handling for diseases with wrong predictions
        disease_mappings = {
            'typhoid': 'Typhoid',
            'dengue': 'Dengue',
            'vertigo': 'Paroymsal Positional Vertigo',
            'hepatitis e': 'Hepatitis E',
            'tuberculosis': 'Tuberculosis'
        }

        for keyword, disease in disease_mappings.items():
            if keyword in ' '.join(valid_symptoms).lower() and predicted_disease != disease:
                predicted_disease = disease
                break

        # Get recommendations
        desc, prec, meds, diet, workouts = get_recommendations(predicted_disease)

        # Log interaction
        log_interaction(symptoms, predicted_disease)

        # Store in session for back navigation
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
        print(f"Prediction error: {str(e)}")
        traceback.print_exc()
        return render_template('index.html', error="An error occurred. Please try again.", symptoms_list=symptoms_list,
                               diseases_list=diseases_list)


@app.route('/results')
def results():
    """Show prediction results"""
    if 'prediction_data' not in session:
        return redirect(url_for('index'))

    data = session['prediction_data']
    return render_template('index.html', **data, symptoms_list=symptoms_list, diseases_list=diseases_list)


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
        disease_data = training_data[training_data['prognosis'] == disease].iloc[:, :-1]
        symptoms = disease_data.sum().nlargest(5).index.tolist()
        return jsonify({'status': 'success', 'disease': disease, 'symptoms': symptoms})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/blog')
def blog():
    return render_template("blog.html")


if __name__ == '__main__':
    app.run(debug=True)

handler = Mangum(app)
