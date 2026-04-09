from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import json
import xgboost as xgb

app = Flask(__name__)
CORS(app)

GB_MODEL_PATH = 'gradient_boosting_model.pkl'
XGB_MODEL_PATH = 'xgboost_model.pkl'
ENCODERS_PATH = 'label_encoders.pkl'
USER_DATA_PATH = 'train_users_2.csv'

XGB_MODEL_JSON_PATH = 'xgboost_model.json'
XGB_COLUMNS_PATH = 'xgb_feature_columns.json'
TARGET_ENCODER_PATH = 'country_label_encoder.pkl'

GB_FEATURES = [
    'age', 'gender', 'signup_method', 'signup_flow', 'language',
    'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked',
    'signup_app', 'first_device_type', 'first_browser'
]

GB_CATEGORICAL_COLS = [
    'gender', 'signup_method', 'language', 'affiliate_channel',
    'affiliate_provider', 'first_affiliate_tracked', 'signup_app',
    'first_device_type', 'first_browser', 'country_destination'
]

try:
    with open(GB_MODEL_PATH, 'rb') as f:
        gb_model = pickle.load(f)

    with open(XGB_MODEL_PATH, 'rb') as f:
        xgb_model = pickle.load(f)

    with open(ENCODERS_PATH, 'rb') as f:
        label_encoders = pickle.load(f)

except Exception as e:
    raise RuntimeError(f"Error loading Gradient Boost artifacts: {e}")

try:
    xgb_booster = xgb.Booster()
    xgb_booster.load_model(XGB_MODEL_JSON_PATH)

    with open(XGB_COLUMNS_PATH, 'r') as f:
        xgb_feature_columns = json.load(f)

    with open(TARGET_ENCODER_PATH, 'rb') as f:
        country_label_encoder = pickle.load(f)

except Exception as e:
    raise RuntimeError(f"Error loading XGBoost artifacts: {e}")


@app.route('/')
def home():
    return render_template('app_final.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_id = data.get('user_id')
    model_choice = data.get('model')

    if not user_id:
        return jsonify({"error": "User ID is required."}), 400

    try:
        user_data = pd.read_csv(USER_DATA_PATH)
        user_data = user_data[user_data['id'] == user_id]

        if user_data.empty:
            return jsonify({"result": "No data found for the given User ID."})

        print("Selected model:", model_choice)

        if model_choice == 'XGBoost':
            # Replicate training preprocessing: date features + get_dummies + column alignment
            xgb_raw_cols = [
                'timestamp_first_active', 'age', 'signup_flow',
                'date_account_created', 'date_first_booking',
                'gender', 'signup_method', 'language', 'affiliate_channel',
                'affiliate_provider', 'first_affiliate_tracked', 'signup_app',
                'first_device_type', 'first_browser'
            ]
            available_cols = [c for c in xgb_raw_cols if c in user_data.columns]
            user_features = user_data[available_cols].copy()

            user_features['date_first_booking'] = pd.to_datetime(
                user_features['date_first_booking'].fillna('Not Booked'), errors='coerce'
            )
            user_features['booking_year'] = user_features['date_first_booking'].dt.year
            user_features['booking_month'] = user_features['date_first_booking'].dt.month
            user_features['booking_day'] = user_features['date_first_booking'].dt.day
            user_features = user_features.drop(columns=['date_first_booking'], errors='ignore')

            user_features['age'] = user_features['age'].fillna(user_features['age'].median())
            user_features['first_affiliate_tracked'] = user_features['first_affiliate_tracked'].fillna('unknown')

            user_features = pd.get_dummies(user_features, drop_first=True)

            # Align to training feature columns
            user_features = user_features.reindex(columns=xgb_feature_columns, fill_value=0)

            dmatrix = xgb.DMatrix(user_features)
            y_pred_proba = xgb_booster.predict(dmatrix)

            if len(y_pred_proba.shape) == 1:
                y_pred_proba = y_pred_proba.reshape(1, -1)

            # Exclude NDF from predictions
            ndf_idx = list(country_label_encoder.classes_).index('NDF')
            y_pred_proba[0, ndf_idx] = 0
            y_pred_proba[0] = y_pred_proba[0] / y_pred_proba[0].sum()

            top_5_indices = np.argsort(y_pred_proba, axis=1)[0, -5:][::-1]
            top_5_countries = country_label_encoder.inverse_transform(top_5_indices)
            top_5_probs = y_pred_proba[0][top_5_indices]

        elif model_choice == 'Gradient Boost':
            user_features = user_data[GB_FEATURES].copy()

            for col in GB_CATEGORICAL_COLS:
                if col in user_features.columns:
                    user_features.loc[:, col] = label_encoders[col].transform(
                        user_features[col].fillna('Unknown')
                    )

            user_features = user_features.fillna(0)
            y_pred_proba = gb_model.predict_proba(user_features)

            top_5_indices = np.argsort(y_pred_proba, axis=1)[0, -5:][::-1]
            top_5_countries = label_encoders['country_destination'].inverse_transform(top_5_indices)
            top_5_probs = y_pred_proba[0][top_5_indices]

        else:
            return jsonify({"error": "Invalid model selected."}), 400

        result = [
            {
                "country": country,
                "probability": float(prob)
            }
            for country, prob in zip(top_5_countries, top_5_probs)
        ]

        return jsonify({"result": result})

    except KeyError as e:
        return jsonify({"error": f"Missing required feature: {e}"}), 400
    except Exception as e:
        print("BACKEND ERROR:", repr(e))
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)