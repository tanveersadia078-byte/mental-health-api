from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import os

app = Flask(__name__)
CORS(app)

print("🚀 Starting Mental Health API...")

# Model Class
class MentalHealthRecommender:
    def __init__(self):
        self.rf_model = None
        self.knn_model = None
        self.label_encoders = {}
        self.cbt_encoder = LabelEncoder()
        self.wellness_encoder = LabelEncoder()
        self.lookup_table = None

    def prepare_data(self, df):
        self.lookup_table = df.copy()
        feature_columns = ['emotion', 'sub_emotion', 'condition', 'sub_condition']
        df_encoded = df.copy()

        for col in feature_columns:
            self.label_encoders[col] = LabelEncoder()
            df_encoded[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col])

        df_encoded['cbt_encoded'] = self.cbt_encoder.fit_transform(df['cbt_exercise'])
        df_encoded['wellness_encoded'] = self.wellness_encoder.fit_transform(df['wellness_challenge'])

        X = df_encoded[['emotion_encoded', 'sub_emotion_encoded',
                        'condition_encoded', 'sub_condition_encoded']]
        y = df_encoded[['cbt_encoded', 'wellness_encoded']]
        return X, y

    def train(self, df):
        print("🚀 Training model...")
        X, y = self.prepare_data(df)

        rf_cbt = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, verbose=0)
        rf_wellness = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, verbose=0)
        knn_cbt = KNeighborsClassifier(n_neighbors=min(5, len(X)), weights='distance')
        knn_wellness = KNeighborsClassifier(n_neighbors=min(5, len(X)), weights='distance')

        rf_cbt.fit(X, y['cbt_encoded'])
        rf_wellness.fit(X, y['wellness_encoded'])
        knn_cbt.fit(X, y['cbt_encoded'])
        knn_wellness.fit(X, y['wellness_encoded'])

        self.rf_model = {'cbt': rf_cbt, 'wellness': rf_wellness}
        self.knn_model = {'cbt': knn_cbt, 'wellness': knn_wellness}
        print("✅ Model trained!")
        return self

    def exact_lookup(self, emotion, sub_emotion, condition, sub_condition):
        result = self.lookup_table[
            (self.lookup_table['emotion'].str.lower().str.strip() == emotion.lower().strip()) &
            (self.lookup_table['sub_emotion'].str.lower().str.strip() == sub_emotion.lower().strip()) &
            (self.lookup_table['condition'].str.lower().str.strip() == condition.lower().strip()) &
            (self.lookup_table['sub_condition'].str.lower().str.strip() == sub_condition.lower().strip())
        ]
        if not result.empty:
            return {
                'method': 'Exact Match',
                'cbt': result.iloc[0]['cbt_exercise'],
                'wellness': result.iloc[0]['wellness_challenge'],
                'confidence': 1.0
            }
        return None

    def ensemble_predict(self, emotion, sub_emotion, condition, sub_condition):
        try:
            input_data = np.array([[
                self.label_encoders['emotion'].transform([emotion])[0],
                self.label_encoders['sub_emotion'].transform([sub_emotion])[0],
                self.label_encoders['condition'].transform([condition])[0],
                self.label_encoders['sub_condition'].transform([sub_condition])[0]
            ]])
            rf_cbt_pred = self.rf_model['cbt'].predict(input_data)[0]
            rf_wellness_pred = self.rf_model['wellness'].predict(input_data)[0]
            cbt_exercise = self.cbt_encoder.inverse_transform([rf_cbt_pred])[0]
            wellness_challenge = self.wellness_encoder.inverse_transform([rf_wellness_pred])[0]
            return {
                'method': 'ML Prediction',
                'cbt': cbt_exercise,
                'wellness': wellness_challenge,
                'confidence': 0.85
            }
        except:
            return None

    def predict(self, emotion, sub_emotion, condition, sub_condition):
        result = self.exact_lookup(emotion, sub_emotion, condition, sub_condition)
        if result:
            return result
        result = self.ensemble_predict(emotion, sub_emotion, condition, sub_condition)
        if result:
            return result
        return {
            'method': 'Default',
            'cbt': 'Practice deep breathing',
            'wellness': 'Take a mindful walk',
            'confidence': 0.60
        }

    def get_all_emotions(self):
        return sorted(self.lookup_table['emotion'].unique().tolist())

    def get_sub_emotions(self, emotion):
        return sorted(self.lookup_table[self.lookup_table['emotion'] == emotion]['sub_emotion'].unique().tolist())

    def get_conditions(self):
        return sorted(self.lookup_table['condition'].unique().tolist())

    def get_sub_conditions(self, condition):
        return sorted(self.lookup_table[self.lookup_table['condition'] == condition]['sub_condition'].unique().tolist())

# Load dataset
try:
    print("📂 Loading dataset...")
    df = pd.read_csv('DataSet.csv')
    print(f"✅ Dataset loaded: {len(df)} rows")
    
    recommender = MentalHealthRecommender()
    recommender.train(df)
    print("✅ API is ready!")
except Exception as e:
    print(f"❌ Error: {e}")
    recommender = None

# Routes
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'online',
        'message': 'Mental Health API'
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        result = recommender.predict(
            data['emotion'], 
            data['sub_emotion'],
            data['condition'], 
            data['sub_condition']
        )
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/emotions', methods=['GET'])
def get_emotions():
    return jsonify({'success': True, 'emotions': recommender.get_all_emotions()})

@app.route('/sub_emotions/<emotion>', methods=['GET'])
def get_sub_emotions(emotion):
    return jsonify({'success': True, 'sub_emotions': recommender.get_sub_emotions(emotion)})

@app.route('/conditions', methods=['GET'])
def get_conditions():
    return jsonify({'success': True, 'conditions': recommender.get_conditions()})

@app.route('/sub_conditions/<condition>', methods=['GET'])
def get_sub_conditions(condition):
    return jsonify({'success': True, 'sub_conditions': recommender.get_sub_conditions(condition)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)