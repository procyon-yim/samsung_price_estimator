# Stock Price Estimator

A Streamlit-based web application that predicts stock prices using Deep Learning (LSTM) and provides AI-powered feature recommendations.

## Features

- **Stock Price Prediction**: Uses an LSTM (Long Short-Term Memory) neural network to predict the next day's closing price.
- **Interactive Visualization**: Visualizes historical data and prediction trends using Seaborn and Matplotlib.
- **AI Ticker Search**: Search for any company's stock ticker (e.g., "Samsung Electronics", "Tesla") using natural language. (Powered by LangChain & OpenAI)
- **AI Feature Recommendation**: Describe your investment style (e.g., "Safety first", "High volatility"), and the AI will recommend the best technical indicators (Features) for training the model.
- **Customizable Training**: Select specific features (Open, High, Low, Volume, Moving Averages, etc.) and window size for the model.

## Technologies Used

- **Python 3.10+**
- **Streamlit**: Web interface
- **TensorFlow / Keras**: LSTM model implementation
- **LangChain & OpenAI**: LLM integration for search and recommendations
- **FinanceDataReader**: Stock market data acquisition
- **Seaborn & Matplotlib**: Data visualization

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/procyon-yim/samsung_price_estimator.git
    cd samsung_price_estimator
    ```

2.  **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

This project requires an OpenAI API Key for the AI features (Ticker Search and Feature Recommendation).

1.  Create a `.env` file in the root directory:
    ```bash
    touch .env
    ```

2.  Add your OpenAI API Key to `.env`:
    ```env
    OPENAI_API_KEY=your_sk_key_here
    ```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The app will open in your browser (usually at `http://localhost:8501`).

### How to use:
1.  **Find a Ticker**: Use the "종목 검색" (Ticker Search) in the sidebar to find a stock code by company name.
2.  **Select Features**:
    - Manually select features like `3MA` (3-day moving average), `Volume`, etc.
    - OR use the **AI Feature Recommender** by describing your investment style.
3.  **Predict**: Click "Predict Price" to train the model on real-time data and see the forecast.

## License

MIT License
