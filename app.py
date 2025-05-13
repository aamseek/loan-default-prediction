from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import pickle # For saving/loading the model and preprocessors
import os

app = Flask(__name__)

# Global variables for model and preprocessors (to be loaded once)
MODEL = None
IMPUTER = None
SCALER = None
LABEL_ENCODERS = {} # To store LabelEncoders for each column
PINCODES_DF = None
FEATURES = None # List of feature names used for training

def preprocess_data(df, pincodes_df, is_training=False):
    # --- Start of preprocessing steps ---
    df = df.copy() # Work on a copy to avoid SettingWithCopyWarning

    # 1. Merge with pincode data
    if 'c_address_pincode' in df.columns and pincodes_df is not None:
        # Ensure 'Pincode' in pincodes_df is string for mapping if c_address_pincode is object/string
        pincodes_df['Pincode'] = pincodes_df['Pincode'].astype(str)
        df['address_city'] = df['c_address_pincode'].astype(str).map(pincodes_df.set_index('Pincode')['City'])
        df['address_city'].fillna('NA', inplace=True)
    elif 'address_city' not in df.columns: # If c_address_pincode is not there, ensure address_city exists
        df['address_city'] = 'NA'

    # 2. Date handling and age calculation
    if 'age' not in df.columns: # If age is not directly provided for prediction
        if 'dob' in df.columns and 'created_at' in df.columns: # Expect these for training if age isn't there
            # Specific fix to handle potential errors if this index doesn't exist
            if 109538 in df.index and 'created_at' in df.columns:
                 df.loc[109538, 'created_at'] = '05-08-2020'
            
            df['dob'] = pd.to_datetime(df['dob'], format='%Y-%m-%d', errors='coerce')
            df['created_at'] = pd.to_datetime(df['created_at'], format='%d-%m-%Y', errors='coerce')
            df['age'] = (df['created_at'] - df['dob']).dt.days / 365
            df['age'].fillna(df['age'].median() if not df['age'].isnull().all() else 30, inplace=True)
            df['age'] = df['age'].astype(int)
        else: # Fallback if essential date columns are missing
            df['age'] = 30 # Default or raise error

    # 3. Outlier handling
    # For deployment, use saved medians/quantiles from the training phase.
    # Values from analysis done in notebook:
    
    # TODO: These values to be stored with preprocessors.
    salary_q01 = 9601.9
    salary_q99 = 207289.33
    salary_median = 30000.0 # median after filtering

    job_stability_q02 = 0.0
    job_stability_q98 = 312.0
    job_stability_median = 60.0

    if 'work_salary' in df.columns:
        if is_training:
            lower_limit_salary = df['work_salary'].quantile(0.01)
            upper_limit_salary = df['work_salary'].quantile(0.99)
            # Calculate median on non-outlier part for robustness
            median_salary_train = df.loc[(df['work_salary'] >= lower_limit_salary) & (df['work_salary'] <= upper_limit_salary), 'work_salary'].median()
            df['work_salary'] = np.where((df['work_salary'] < lower_limit_salary) | (df['work_salary'] > upper_limit_salary), median_salary_train, df['work_salary'])
        else:
            df['work_salary'] = np.where((df['work_salary'] < salary_q01) | (df['work_salary'] > salary_q99), salary_median, df['work_salary'])
        df['work_salary'].fillna(salary_median, inplace=True) # Fill any NaNs that might remain or appear
    elif 'work_salary' not in df.columns and FEATURES and 'work_salary' in FEATURES:
        df['work_salary'] = salary_median


    if 'work_job_stability' in df.columns:
        if is_training:
            lower_limit_stability = df['work_job_stability'].quantile(0.02)
            upper_limit_stability = df['work_job_stability'].quantile(0.98)
            median_stability_train = df.loc[(df['work_job_stability'] >= lower_limit_stability) & (df['work_job_stability'] <= upper_limit_stability), 'work_job_stability'].median()
            df['work_job_stability'] = np.where((df['work_job_stability'] < lower_limit_stability) | (df['work_job_stability'] > upper_limit_stability), median_stability_train, df['work_job_stability'])
        else:
            df['work_job_stability'] = np.where((df['work_job_stability'] < job_stability_q02) | (df['work_job_stability'] > job_stability_q98), job_stability_median, df['work_job_stability'])
        df['work_job_stability'].fillna(job_stability_median, inplace=True)
    elif 'work_job_stability' not in df.columns and FEATURES and 'work_job_stability' in FEATURES:
        df['work_job_stability'] = job_stability_median
        
    # Ensure all required columns for categorical engineering are present, fill with 'NA' for consistency
    cat_eng_cols = ['surgery_id', 'borrower_relationship', 'work_business_type', 'existing_loan_1', 
                    'city_wise_leads', 'address_city', 'work_company_category']
    for col in cat_eng_cols:
        if col not in df.columns:
            df[col] = 'NA'
        else: # Ensure they are string type for consistent replacement/mapping
            df[col] = df[col].astype(str).fillna('NA')


    # 4. Categorical feature engineering
    df['surgery_id'].replace('346.0', 'Hair Transplant', inplace=True) # Floats as strings due to prior astype(str)
    df['surgery_id'].replace('3.0', 'IVF', inplace=True)
    df.loc[~df['surgery_id'].isin(['Hair Transplant', 'IVF']), 'surgery_id'] = 'Others'
    
    df['borrower_relationship'].replace('husband', 'Husband', inplace=True) # Case consistency
    df['borrower_relationship'].replace('son', 'Son', inplace=True)
    df.loc[~df['borrower_relationship'].isin(['Husband', 'Son']), 'borrower_relationship'] = 'Others'
    
    business_type_cats = ['private_limited', 'business', 'public_limited', 'proprietorship', 'state_government', 'central_government', 'private'] # 'private' added
    df.loc[~df['work_business_type'].isin(business_type_cats), 'work_business_type'] = 'Others'
        
    existing_loan_cats = ['Personal Loan', 'Consumer Loan', 'Housing Loan', 'Auto Loan']
    df.loc[~df['existing_loan_1'].isin(existing_loan_cats), 'existing_loan_1'] = 'Others'
         
    city_leads_cats = ['delhi', 'bangalore', 'mumbai', 'pune']
    df.loc[~df['city_wise_leads'].isin(city_leads_cats), 'city_wise_leads'] = 'Others'
        
    address_city_cats = ['Delhi', 'Bengaluru', 'Mumbai', 'Ghaziabad', 'Pune']
    df.loc[~df['address_city'].isin(address_city_cats), 'address_city'] = 'Others'

    company_cat_cats = ['private_limited', 'business', 'public_limited', 'proprietorship', 'state_government', 'central_government']
    df.loc[~df['work_company_category'].isin(company_cat_cats), 'work_company_category'] = 'Others'
    
    # Additional columns that might need NA filling if not present
    simple_fill_na_cols = ['bucket', 'dmi_credit_decision', 'lender', 'type', 'loan_purpose',
                           'loan_to_be_disbursed', 'loan_type', 'work_employment_type',
                           'hospitalObject_type', 'gender', 'marital_status',
                           'hospitalObject_nature']
    for col in simple_fill_na_cols:
        if col not in df.columns:
            df[col] = 'NA'
        else:
            df[col] = df[col].astype(str).fillna('NA') # Ensure string and fill NA

    # Numeric columns that might need NA filled with a suitable value (e.g. 0 or median) before imputation
    numeric_cols_fill_zero = ['dmi_cibil_score', 'loan_amount', 'loan_tenure', 'loan_rate',
                              'subvention_percentage', 'processing_fee', 'down_payment',
                              'net_disbursal_amount', 'proposed_emi', 'advance_emi',
                              'existing_emi', 'foir', 'abb_3_months',
                              'actual_medical_bill', 'hospitalObject_status',
                              'work_current_job_duration'] # Already handled: work_salary, work_job_stability, age
    for col in numeric_cols_fill_zero:
        if col not in df.columns:
            df[col] = 0 # Or np.nan for imputer
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) # Coerce and fill NaNs with 0

    # 5. Select features
    df_selected = df.copy()
    if FEATURES:
        # Add any FEATURES columns missing in df_selected, fill with np.nan before imputation
        for f_col in FEATURES:
            if f_col not in df_selected.columns:
                df_selected[f_col] = np.nan
        df_selected = df_selected[FEATURES].copy()


    # 6. Label Encoding
    global LABEL_ENCODERS
    for col in df_selected.columns:
        if df_selected[col].dtype == 'object':
            df_selected[col] = df_selected[col].astype(str).fillna('NA') # Ensure string and fill NA
            if is_training:
                if len(list(df_selected[col].unique())) <= 7:
                    le = LabelEncoder()
                    df_selected[col] = le.fit_transform(df_selected[col])
                    LABEL_ENCODERS[col] = le
            else: # Prediction
                if col in LABEL_ENCODERS:
                    le = LABEL_ENCODERS[col]
                    # Handle unseen labels by mapping them to a new '<unknown>' category if not already part of encoder
                    # Or by assigning them to the most frequent class's encoded value or a specific placeholder
                    current_col_values = df_selected[col].astype(str)
                    
                    # Option 1: Map unknown to a specific known category (e.g., 'Others' if it exists and makes sense)
                    # This requires knowing which columns 'Others' is valid for.

                    # Option 2: More robustly, add '<unknown>' to encoder classes during training or handle here
                    known_classes = le.classes_
                    df_selected[col] = current_col_values.map(lambda s: s if s in known_classes else '<unknown>')
                    
                    if '<unknown>' not in le.classes_:
                        le.classes_ = np.append(le.classes_, '<unknown>')
                    
                    df_selected[col] = le.transform(df_selected[col])

    # Fill NaNs that might have been introduced for columns not label encoded or missing features.
    # Imputer will handle these numeric NaNs.
    # For object columns not label encoded (due to >7 unique values), they should be string type.
    for col in df_selected.columns:
        if df_selected[col].dtype == 'object':
            df_selected[col] = df_selected[col].astype(str).fillna('NA_imputed_placeholder')


    return df_selected

@app.route('/train', methods=['POST'])
def train_model_endpoint():
    global MODEL, IMPUTER, SCALER, LABEL_ENCODERS, PINCODES_DF, FEATURES

    if 'default_fields_1' not in request.files or \
       'default_additional_fields' not in request.files or \
       'all_pincodes' not in request.files:
        return jsonify({"error": "Missing one or more CSV files"}), 400

    try:
        raw_data_1 = pd.read_csv(request.files['default_fields_1'])
        raw_data_2 = pd.read_csv(request.files['default_additional_fields'])
        pincodes_df_train = pd.read_csv(request.files['all_pincodes'], encoding='ISO-8859-1')
        PINCODES_DF = pincodes_df_train.copy()
        
        target_column_name = request.form.get('target_column_name', 'default')

        data = pd.merge(raw_data_1, raw_data_2, on='lead_id', how='left') # Use left merge if raw_data_2 might not have all leads

        # Specific fix for 'created_at' before date conversion
        if 109538 in data.index: # Check if index exists to avoid KeyError
             data.loc[data.index == 109538, 'created_at'] = '05-08-2020'


        if 'dob' in data.columns and 'created_at' in data.columns:
            data['dob'] = pd.to_datetime(data['dob'], format='%Y-%m-%d', errors='coerce')
            data['created_at'] = pd.to_datetime(data['created_at'], format='%d-%m-%Y', errors='coerce')
            data['age'] = (data['created_at'] - data['dob']).dt.days / 365
            median_age = data['age'].median()
            data['age'].fillna(median_age if pd.notna(median_age) else 30, inplace=True)
            data['age'] = data['age'].astype(int)
        elif 'age' not in data.columns:
            return jsonify({"error": "Columns 'dob' and 'created_at' are required to calculate 'age', or 'age' must be provided."}), 400
        
        if 'disbursed_timestamp' in data.columns:
            data['disbursed_timestamp'] = pd.to_datetime(data['disbursed_timestamp'], format='%d-%m-%Y', errors='coerce')
            data_backup = data[ (data['disbursed_timestamp'] > pd.Timestamp('2019-06-01')) & 
                                (data['disbursed_timestamp'] < pd.Timestamp('2019-12-31')) ].copy() # Ensure it's a copy
            if data_backup.empty:
                 return jsonify({"error": "No data remains after filtering by 'disbursed_timestamp'. Check filter or data."}), 400
            data_for_features = data_backup # Use this for feature selection
        else:
            return jsonify({"error": "Missing 'disbursed_timestamp' column for data filtering"}), 400
            
        FEATURES = [
            'bucket', 'dmi_credit_decision', 'dmi_cibil_score', 'lender', 'type', 
            'loan_amount', 'loan_tenure', 'loan_rate', 'subvention_percentage', 
            'processing_fee', 'down_payment', 'net_disbursal_amount', 'loan_purpose', 
            'loan_to_be_disbursed', 'proposed_emi', 'advance_emi', 'surgery_id', 
            'borrower_relationship', 'loan_type', 'work_job_stability', 
            'work_business_type', 'work_salary', 'work_employment_type', 
            'hospitalObject_type', 'existing_emi', 'foir', 
            'abb_3_months', 'existing_loan_1', 'gender', 'work_company_category', 
            'work_current_job_duration', 'marital_status', 
            'actual_medical_bill', 'hospitalObject_nature', 'hospitalObject_status', 
            'city_wise_leads', 'address_city', 'age'
        ]
        
        missing_training_cols = [col for col in FEATURES if col not in data_for_features.columns]
        if missing_training_cols:
            return jsonify({"error": f"Following feature columns are missing from the training data: {', '.join(missing_training_cols)}"}), 400

        train_df_features = data_for_features[FEATURES].copy()
        
        if target_column_name not in data_for_features.columns:
            return jsonify({"error": f"Target column '{target_column_name}' not found in the filtered data_backup"}), 400
        train_labels = data_for_features[target_column_name].apply(lambda x: 1 if x in [1,2,3,4] else 0).values

        LABEL_ENCODERS = {} # Reset global LABEL_ENCODERS before training
        train_processed = preprocess_data(train_df_features, pincodes_df_train.copy(), is_training=True)

        IMPUTER = SimpleImputer(strategy='most_frequent')
        train_imputed = IMPUTER.fit_transform(train_processed)
        
        SCALER = MinMaxScaler(feature_range=(0, 1))
        train_scaled = SCALER.fit_transform(train_imputed)

        MODEL = LogisticRegression(C=0.0001, solver='liblinear', random_state=42)
        MODEL.fit(train_scaled, train_labels)

        # Save all artifacts
        artifacts_path = "./artifacts" # Define a subfolder for artifacts
        if not os.path.exists(artifacts_path):
            os.makedirs(artifacts_path)

        with open(os.path.join(artifacts_path, 'model.pkl'), 'wb') as f: pickle.dump(MODEL, f)
        with open(os.path.join(artifacts_path, 'imputer.pkl'), 'wb') as f: pickle.dump(IMPUTER, f)
        with open(os.path.join(artifacts_path, 'scaler.pkl'), 'wb') as f: pickle.dump(SCALER, f)
        with open(os.path.join(artifacts_path, 'label_encoders.pkl'), 'wb') as f: pickle.dump(LABEL_ENCODERS, f)
        with open(os.path.join(artifacts_path, 'pincodes_df.pkl'), 'wb') as f: pickle.dump(PINCODES_DF, f)
        with open(os.path.join(artifacts_path, 'features_list.pkl'), 'wb') as f: pickle.dump(FEATURES, f)

        return jsonify({"message": "Model trained and artifacts saved successfully in ./artifacts folder."})

    except FileNotFoundError as e:
        return jsonify({"error": f"A required CSV file was not found: {e.filename}"}), 400
    except KeyError as e:
        return jsonify({"error": f"A required column is missing from the data: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred during training: {str(e)}"}), 500


def load_model_and_preprocessors_on_startup():
    global MODEL, IMPUTER, SCALER, LABEL_ENCODERS, PINCODES_DF, FEATURES
    artifacts_path = "./artifacts"
    try:
        with open(os.path.join(artifacts_path,'model.pkl'), 'rb') as f: MODEL = pickle.load(f)
        with open(os.path.join(artifacts_path,'imputer.pkl'), 'rb') as f: IMPUTER = pickle.load(f)
        with open(os.path.join(artifacts_path,'scaler.pkl'), 'rb') as f: SCALER = pickle.load(f)
        with open(os.path.join(artifacts_path,'label_encoders.pkl'), 'rb') as f: LABEL_ENCODERS = pickle.load(f)
        with open(os.path.join(artifacts_path,'pincodes_df.pkl'), 'rb') as f: PINCODES_DF = pickle.load(f)
        with open(os.path.join(artifacts_path,'features_list.pkl'), 'rb') as f: FEATURES = pickle.load(f)
        print("Model and preprocessors loaded successfully from ./artifacts.")
    except FileNotFoundError:
        print("One or more .pkl artifact files not found in ./artifacts. Train the model using the /train endpoint or place them manually.")
        MODEL = IMPUTER = SCALER = LABEL_ENCODERS = PINCODES_DF = FEATURES = None
    except Exception as e:
        print(f"Error loading artifacts: {str(e)}")
        MODEL = IMPUTER = SCALER = LABEL_ENCODERS = PINCODES_DF = FEATURES = None


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if not all([MODEL, IMPUTER, SCALER, LABEL_ENCODERS, PINCODES_DF, FEATURES]):
        return jsonify({"error": "Model or preprocessors not loaded. Please train or load them first via /train or ensure artifacts are present."}), 500

    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        input_df = pd.DataFrame(json_data) 
        if input_df.empty:
             return jsonify({"error": "Empty JSON data provided"}), 400

        # Store original index if it's meaningful (e.g., lead_id)
        original_indices = input_df.index 
        if 'lead_id' in input_df.columns: # Prefer a lead_id column if present
            original_indices = input_df['lead_id']


        input_processed = preprocess_data(input_df.copy(), PINCODES_DF, is_training=False)
        
        input_imputed = IMPUTER.transform(input_processed)
        input_scaled = SCALER.transform(input_imputed)

        predictions_proba = MODEL.predict_proba(input_scaled)[:, 1]
        
        # Use original_indices for mapping results
        if len(original_indices) == len(predictions_proba):
            results = [{"identifier": idx, "default_probability": float(prob)} for idx, prob in zip(original_indices, predictions_proba)]
        else: # Fallback if index length mismatch
            results = [{"index": i, "default_probability": float(prob)} for i, prob in enumerate(predictions_proba)]

        return jsonify(results)

    except KeyError as e:
        return jsonify({"error": f"Missing expected feature in input JSON: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

@app.route('/status', methods=['GET'])
def status_endpoint():
    if all([MODEL, IMPUTER, SCALER, LABEL_ENCODERS, PINCODES_DF, FEATURES]):
        return jsonify({"status": "Model and preprocessors are loaded."})
    else:
        return jsonify({"status": "Model and/or preprocessors are NOT loaded. Please use /train or ensure .pkl files are present in ./artifacts folder."})

if __name__ == '__main__':
    load_model_and_preprocessors_on_startup()
    app.run(debug=True, host='0.0.0.0', port=5000)