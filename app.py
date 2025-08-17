import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and scaler
@st.cache_resource
def load_artifacts():
    return {
        'model': joblib.load('model/fraud_model.joblib'),
        'scaler': joblib.load('model/scaler.joblib'),
        'feature_names': joblib.load('model/feature_names.joblib')  # Save this during training
    }

artifacts = load_artifacts()
model = artifacts['model']
scaler = artifacts['scaler']
expected_features = artifacts['feature_names']

def format_dataframe(df):
    """Apply numeric formatting to float columns"""
    format_dict = {
        col: "{:,.2f}" for col in [
            'amount', 'oldbalanceOrg', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest',
            'errorBalanceOrg', 'errorBalanceDest',
            'fraudProbability'
        ] if col in df.columns
    }
    return df.style.format(format_dict)

def preprocess_input(input_df):
    """Complete preprocessing matching exact training setup"""
    try:
        # 1. Calculate error balances
        input_df['errorBalanceOrg'] = input_df['newbalanceOrig'] + input_df['amount'] - input_df['oldbalanceOrg']
        input_df['errorBalanceDest'] = input_df['oldbalanceDest'] + input_df['amount'] - input_df['newbalanceDest']
        
        # 2. Create ONLY the dummy variables the model expects
        # (Assuming model only needs CASH_OUT and TRANSFER)
        input_df['type_CASH_OUT'] = (input_df['type'] == 'CASH_OUT').astype(int)
        input_df['type_TRANSFER'] = (input_df['type'] == 'TRANSFER').astype(int)
        
        # 3. Calculate hour of day
        input_df['HourOfDay'] = input_df['step'] % 24
        
        # 4. Ensure we have EXACTLY the features the model expects
        # Add missing features with 0 values
        for feature in expected_features:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # 5. Select columns in EXACT order expected by model/scaler
        processed = input_df[expected_features]
        
        # 6. Standardize the features
        processed = scaler.transform(processed)
        
        return processed
    
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        return None

def predict_fraud(input_data):
    processed_data = preprocess_input(input_data)
    if processed_data is None:
        return None, None
    return model.predict(processed_data), model.predict_proba(processed_data)

def main():
    st.set_page_config(layout="wide", page_title="Fraud Detection System")
    
    # Sidebar navigation
    st.sidebar.title("Configuration")
    app_mode = st.sidebar.radio("Operation Mode", ["Single Transaction", "Batch Processing"])
    
    if app_mode == "Single Transaction":
        st.header("🔍 Transaction Fraud Analysis")
        
        with st.form(key='transaction_form'):
            col1, col2, col3 = st.columns([1,1,1])
            
            with col1:
                st.markdown("### Transaction Metadata")
                step = st.number_input("Step (1-744)", min_value=1, max_value=744, value=1)
                transaction_type = st.selectbox("Type", ["CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT"])
                amount = st.number_input("Amount", min_value=0.01, value=1000.0, format="%.2f")
                
            with col2:
                st.markdown("### Origin Account")
                oldbalance_org = st.number_input("Old Balance", min_value=0.0, value=5000.0, format="%.2f")
                newbalance_orig = st.number_input("New Balance", min_value=0.0, value=4000.0, format="%.2f")
                
            with col3:
                st.markdown("### Destination Account")
                oldbalance_dest = st.number_input("Dest. Old Balance", min_value=0.0, value=0.0, format="%.2f")
                newbalance_dest = st.number_input("Dest. New Balance", min_value=0.0, value=1000.0, format="%.2f")
            
            if st.form_submit_button("Analyze Transaction"):
                with st.spinner("Processing..."):
                    input_data = pd.DataFrame([{
                        'step': step,
                        'type': transaction_type,
                        'amount': amount,
                        'oldbalanceOrg': oldbalance_org,
                        'newbalanceOrig': newbalance_orig,
                        'oldbalanceDest': oldbalance_dest,
                        'newbalanceDest': newbalance_dest
                    }])
                    
                    # Display raw input
                    with st.expander("Transaction Details", expanded=True):
                        st.dataframe(format_dataframe(input_data))
                    
                    # Get prediction
                    prediction, proba = predict_fraud(input_data)
                    
                    if prediction is None:
                        st.error("Prediction failed - check error messages")
                    else:
                        # Show results
                        if prediction[0] == 1:
                            st.error(f"""
                            ## 🚨 Fraud Detected (Probability: {proba[0][1]*100:.2f}%)
                            **Recommendation:** Flag for manual review
                            """)
                        else:
                            st.success(f"""
                            ## ✅ Legitimate Transaction (Probability: {proba[0][0]*100:.2f}%)
                            **Recommendation:** Approve automatically
                            """)

    elif app_mode == "Batch Processing":
        st.header("📁 Batch Transaction Analysis")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        
        if uploaded_file:
            try:
                batch_data = pd.read_csv(uploaded_file)
                
                # Validate required columns
                required_cols = ['step', 'type', 'amount', 'oldbalanceOrg', 
                                'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
                
                if not all(col in batch_data.columns for col in required_cols):
                    missing = set(required_cols) - set(batch_data.columns)
                    st.error(f"Missing required columns: {', '.join(missing)}")
                else:
                    if st.button("Run Fraud Analysis"):
                        with st.spinner("Processing batch..."):
                            # Process in chunks
                            chunk_size = 10000
                            chunks = [batch_data.iloc[i:i+chunk_size] for i in range(0, len(batch_data), chunk_size)]
                            results = []
                            
                            progress_bar = st.progress(0)
                            for i, chunk in enumerate(chunks):
                                preds, probas = predict_fraud(chunk)
                                if preds is not None:
                                    chunk['isFraud'] = preds
                                    chunk['fraudProbability'] = [p[1] for p in probas]
                                    results.append(chunk)
                                progress_bar.progress((i + 1) / len(chunks))
                            
                            if results:
                                final_results = pd.concat(results)
                                
                                # Analysis summary
                                fraud_count = final_results['isFraud'].sum()
                                st.success(f"""
                                ### Analysis Complete
                                - **Total Transactions:** {len(final_results):,}
                                - **Fraudulent Detected:** {fraud_count:,} ({fraud_count/len(final_results)*100:.2f}%)
                                """)
                                
                                # Show sample
                                st.dataframe(format_dataframe(final_results.sample(min(5, len(final_results)))))
                                
                                # Download results
                                csv = final_results.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    "Download Full Results",
                                    data=csv,
                                    file_name="fraud_analysis_results.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.error("All chunks failed processing")
            
            except Exception as e:
                st.error(f"File processing error: {str(e)}")

if __name__ == '__main__':
    main()