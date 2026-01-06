import streamlit as st
import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import train

st.title("Custom Stock Price Predictor")

st.sidebar.header("Configuration")

# 1. Inputs
ticker = st.sidebar.text_input("Ticker Symbol", value="005930")

all_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change', '3MA', '5MA']
default_features = ['3MA', 'Volume', 'Open']
selected_features = st.sidebar.multiselect(
    "Select Features for Training",
    options=all_features,
    default=default_features
)

window_size = st.sidebar.number_input("Window Size", min_value=1, value=15)

today = datetime.date.today()
start_date = st.sidebar.date_input("Start Date", value=datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=today)

if st.button("Predict Price"):
    if not selected_features:
        st.error("Please select at least one feature.")
    elif 'Close' not in selected_features and 'Close' not in all_features: 
        # train.py code assumes 'Close' is always target, but it seems to handle features differently.
        # Let's check train.py logic again. 
        # logic: features_to_drop = list(set(features) - set(user_defined_features))
        # So 'Close' is needed in user_defined_features if we want it as input? 
        # Wait, the prompt says "user_defined_features: list of features to use for training (must include 'Close')"
        # Actually the docstring says "must include 'Close'". Let's enforce it or check.
        # But wait, predict_price signature has default: ['Open', 'High', 'Close', 'Volume', 'Change', '3MA', '5MA']
        # And inside: features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change', '3MA', '5MA']
        # features_to_drop = list(set(features) - set(user_defined_features))
        # X_train = train.drop(features_to_drop, axis=1)
        # So if 'Close' is NOT in user_defined_features, it is DROPPED from X.
        # But y is always 'Close'. So it's fine if it's not in X, unless we want autoregression.
        # The docstring in train.py says "(must include 'Close')". I'll trust the docstring.
        pass

    # Ensure Close is in features if required by docstring, but let's just pass what user selected.
    # Actually, let's look at the example call in train.py: predict_price(['3MA', 'Volume', 'Open'], window_size=30)
    # It does NOT include 'Close'. So the docstring might be slightly misleading or I misinterpreted "use for training".
    # It probably means "available features". But the code drops what is not in user_defined.
    
    with st.spinner("Training model and predicting..."):
        try:
            # Convert dates to string as expected by fdr (or just use what fdr accepts)
            # fdr.DataReader accepts objects that can be parsed.
            training_dates = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            prediction, confidence, history_df = train.predict_price(
                ticker=ticker,
                user_defined_features=selected_features,
                window_size=window_size,
                training_dates=training_dates
            )
            
            st.success(f"Predicted Price: **{prediction:,.0f} KRW**")
            st.info(f"Confidence (Std. Dev): **±{confidence:,.0f} KRW** (Range: {prediction - confidence:,.0f} - {prediction + confidence:,.0f})")
            
            st.subheader("Last 20 Days & Prediction")
            
            # Plotting with Seaborn
            # Create a dataframe for plotting
            plot_df = history_df.reset_index()
            # Assuming index is date
            plot_df.columns = ['Date', 'Close'] if 'Date' not in plot_df.columns else plot_df.columns
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=plot_df, x='Date', y='Close', ax=ax, label='Historical Close', marker='o')
            
            # Add prediction point
            last_date = plot_df['Date'].iloc[-1]
            try:
                next_date = last_date + datetime.timedelta(days=1)
            except:
                # If date is not datetime object (e.g. string), try to parse
                last_date = pd.to_datetime(last_date)
                next_date = last_date + datetime.timedelta(days=1)

            # Plot prediction as a separate point with error bar
            ax.errorbar([next_date], [prediction], yerr=[confidence], fmt='o', color='red', ecolor='salmon', capsize=5, label='Prediction ±1σ')
            
            # Add text annotation for prediction and confidence
            text_str = f"Pred: {prediction:,.0f}\nConf: ±{confidence:,.0f}"
            ax.text(next_date, prediction + confidence, text_str, ha='center', va='bottom', fontsize=9, color='red')

            ax.plot([last_date, next_date], [plot_df['Close'].iloc[-1], prediction], 'r--')
            
            ax.set_title(f"{ticker} Stock Price & Prediction") # + Confidence Interval
            plt.xticks(rotation=45)
            ax.legend()
            
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
