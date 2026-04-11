# Airbnb New User Booking — Country Prediction

A machine learning web application that predicts the top 5 destination countries for new Airbnb users, built with Flask and powered by two independently trained models: **XGBoost** and **Gradient Boosting**.

---

## Overview

When a new user signs up on Airbnb, predicting where they are likely to book helps personalize recommendations and improve conversion. This project tackles that as a multi-class classification problem using the [Airbnb New User Bookings dataset](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings) from Kaggle.

The app takes a User ID, runs it through the selected model, and returns the top 5 predicted destination countries with probabilities.

---

## Model Comparison

| Metric | XGBoost | Gradient Boosting |
|--------|---------|-------------------|
| Accuracy | **87.5%**\* | 70.0% |
| NDCG@5 | **0.927** | 0.824 |
| Precision (weighted) | — | 53.1% |
| Training samples | 213,451 (all users) | 88,908 (booked users only) |
| Features | 500+ (date one-hot encoded) | 11 profile features |

**Why XGBoost outperforms:** XGBoost uses rich temporal features — account creation date (one-hot encoded), first active timestamp, and booking date components — that directly capture booking behavior. Gradient Boosting relies only on 11 user profile features (age, gender, signup method, etc.) and was trained exclusively on users who completed a booking, making it blind to non-booker patterns.

\* *XGBoost accuracy benefits from `booking_year/month/day` features derived from `date_first_booking`, which implicitly leaks whether a user booked (NaN = NDF). Since NDF accounts for 58.3% of users, this inflates accuracy. Without these features, accuracy would likely be closer to 70–75%. This is a known limitation — see below.*

---

## Features Used

**XGBoost:**
- `timestamp_first_active`, `age`, `signup_flow`
- `date_account_created` (one-hot encoded — captures signup cohort patterns)
- `booking_year`, `booking_month`, `booking_day`
- `gender`, `signup_method`, `language`, `affiliate_channel`, `affiliate_provider`
- `first_affiliate_tracked`, `signup_app`, `first_device_type`, `first_browser`

**Gradient Boosting:**
- `age`, `gender`, `signup_method`, `signup_flow`, `language`
- `affiliate_channel`, `affiliate_provider`, `first_affiliate_tracked`
- `signup_app`, `first_device_type`, `first_browser`

---

## Dataset

**Download from Kaggle:** [Airbnb New User Bookings](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/data)

Place the downloaded files in the project root before running the notebooks.

| File | Description |
|------|-------------|
| `train_users_2.csv` | 213,451 users with destination labels |
| `test_users.csv` | 62,096 unlabelled users |
| `countries.csv` | Destination country metadata |
| `age_gender_bkts.csv` | Population data by age/gender per destination |

> All dataset files must be downloaded from Kaggle — they are not included in this repository.

Target classes: `US`, `NDF` (no booking), `other`, `FR`, `IT`, `GB`, `ES`, `CA`, `DE`, `AU`, `PT`

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| ML Models | XGBoost, scikit-learn (GradientBoostingClassifier) |
| Backend | Flask, Flask-CORS |
| Frontend | HTML, CSS, JavaScript |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |

---

## Setup & Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Start the server**
```bash
python app_final.py
```

**3. Open in browser**
```
http://127.0.0.1:5000
```

---

## Usage

1. Enter a User ID from `train_users_2.csv` (e.g. `gxn3p5htnn`)
2. Select a model — XGBoost (higher accuracy) or Gradient Boost
3. Click **Predict** to see the top 5 destination countries with probability bars

The UI also displays the selected model accuracy and NDCG@5 score for context.

---

## Known Limitations

- **Data leakage in XGBoost:** The `date_first_booking` column reveals whether a user booked. Users with NaN never booked (NDF), so the derived `booking_year/month/day` features leak the target. A production model should exclude these and rely on pre-booking signals only.
- **Gradient Boost trained on bookers only:** By dropping NDF users before training, the model cannot predict the most common outcome (58.3% of users). It is useful for predicting *where* a booker goes, but not *whether* they will book.
- **Inference uses training data:** Predictions are made by looking up User IDs in `train_users_2.csv` rather than accepting raw input features — a limitation of this demo.

---

## Project Structure

```
├── .gitignore
├── README.md
├── requirements.txt
├── Report.pdf                            # Project report
├── app_final.py                          # Flask backend
├── templates/
│   └── app_final.html                    # Frontend UI
├── Airbnb_XGBoost_Country_Prediction.ipynb
├── Airbnb_GradientBoosting_Country_Prediction.ipynb
├── xgboost_model.json                    # XGBoost booster (native format)
├── gradient_boosting_model.pkl           # Gradient Boosting model
├── xgb_feature_columns.json             # Feature list for XGBoost inference
├── country_label_encoder.pkl            # Label encoder for XGBoost targets
└── label_encoders.pkl                   # Label encoders for Gradient Boost
```
