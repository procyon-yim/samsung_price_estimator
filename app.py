
import streamlit as st
import predictor

def main():
    st.title("Samsung Stock Price Predictor")
    
    st.write("Select a window size to predict the next closing price of Samsung Electronics.")
    
    # Radio button for window size
    window_size = st.radio(
        "Select Window Size:",
        options=[15, 30],
        index=0,
        horizontal=True
    )
    
    if st.button("Predict Price"):
        with st.spinner("Fetching data and predicting..."):
            try:
                price = predictor.predict_price(window_size)
                st.success(f"Predicted Price (Window Size {window_size}): **{price:,.0f} KRW**")
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
