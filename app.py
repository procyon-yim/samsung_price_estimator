import streamlit as st
import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import train
import base64   
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize session state for ticker if not exists
if 'ticker' not in st.session_state:
    st.session_state.ticker = "005930"

st.title("주가 예측 서비스")

all_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change', '3MA', '5MA']

st.sidebar.header("설정")

# --- Ticker Search Feature ---
st.sidebar.subheader("종목 검색")
company_name = st.sidebar.text_input("회사 이름 (예: 삼성전자)", key="company_search")
if st.sidebar.button("종목 코드 찾기"):
    if not company_name:
        st.sidebar.warning("회사 이름을 입력해주세요.")
    else:
        try:
            # Check for API Key
            if "OPENAI_API_KEY" not in os.environ:
                 st.sidebar.error("OpenAI API Key가 설정되지 않았습니다.")
            else:
                with st.spinner("종목 코드를 찾는 중..."):
                    llm = ChatOpenAI(model="gpt-4o", temperature=0)
                    messages = [
                        SystemMessage(content="You are a helpful assistant that finds stock ticker symbols. Return ONLY the ticker symbol (e.g., 'AAPL', '005930'). Do not add any other text."),
                        HumanMessage(content=f"What is the stock ticker for {company_name}? If it's a Korean company, prefer the 6-digit code.")
                    ]
                    response = llm.invoke(messages)
                    found_ticker = response.content.strip()
                    st.session_state.ticker = found_ticker
                    st.sidebar.success(f"찾았습니다: {found_ticker}")
        except Exception as e:
            st.sidebar.error(f"오류 발생: {e}")

# --- AI Feature Recommender ---
st.sidebar.subheader("AI 투자 성향 분석 및 추천")
investment_style = st.sidebar.text_area("투자 성향을 알려주세요 (예: 안전 제일, 단타 위주)", key="style_input")
if st.sidebar.button("추천 기능 (Features) 받기"):
    if not investment_style:
         st.sidebar.warning("투자 성향을 입력해주세요.")
    else:
        try:
             with st.spinner("전문가가 분석 중입니다..."):
                llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
                all_feats_str = str(['Open', 'High', 'Low', 'Close', 'Volume', 'Change', '3MA', '5MA'])
                
                messages = [
                    SystemMessage(content=f"You are an expert stock trader. Available features: {all_feats_str}. '3MA' and '5MA' are moving averages. 'Change' is daily change. Return ONLY a Python list of 3 to 4 features that best match the user's investment style. Example: ['3MA', 'Volume']. Do not add any other text."),
                    HumanMessage(content=f"User's investment style: {investment_style}")
                ]
                response = llm.invoke(messages)
                
                # Parse response
                import ast
                recommended_features = ast.literal_eval(response.content.strip())
                
                # Validate
                valid_features = [f for f in recommended_features if f in all_features]
                
                if valid_features:
                    st.session_state['selected_features'] = valid_features
                    st.sidebar.success(f"추천된 기능: {valid_features}")
                    st.rerun()
                else:
                    st.sidebar.error("적절한 기능을 찾지 못했습니다.")
                    
        except Exception as e:
            st.sidebar.error(f"오류 발생: {e}")

# 1. Inputs
# Use session state for value
ticker = st.sidebar.text_input("종목 코드", value=st.session_state.ticker)
# Update session state if user manually changes input
if ticker != st.session_state.ticker:
    st.session_state.ticker = ticker

# Initialize defaults if not in session state
if 'selected_features' not in st.session_state:
    st.session_state['selected_features'] = ['3MA', 'Volume', 'Open']

selected_features = st.sidebar.multiselect(
    "학습할 데이터 (Features) 선택",
    options=all_features,
    key='selected_features'
)

window_size = st.sidebar.number_input("Window Size", min_value=1, value=15)


if st.button("Predict Price"):
    if not selected_features:
        st.error("Please select at least one feature.")

    with st.spinner("Training model and predicting..."):
        try:
            
            prediction, history_df = train.predict_price(
                ticker=ticker,
                user_defined_features=selected_features,
                window_size=window_size
            )

            import numpy as np
            if isinstance(prediction, (np.ndarray, list)):
                try:

                    prediction = float(prediction.flatten()[0])
                except:
                    prediction = float(prediction)
            
            st.success(f"Predicted Price: **{prediction:,.0f} KRW**")            
            st.subheader("Last 20 Days & Prediction")
            
            # Plotting with Seaborn
            sns.set_theme(style="white", context="talk") # Cleaner "talk" context for larger fonts, white background
            
            # Create a dataframe for plotting
            plot_df = history_df.reset_index()
            # Assuming index is date
            plot_df.columns = ['Date', 'Close'] if 'Date' not in plot_df.columns else plot_df.columns
            
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # 1. Historical Data - Smooth line, nice color
            # Use spline interpolation for smoother lines
            try:
                from scipy.interpolate import make_interp_spline
                import numpy as np
                
                # Convert dates to numbers for interpolation
                # Ensure dates are datetime
                plot_df['Date'] = pd.to_datetime(plot_df['Date'])
                date_nums = pd.to_numeric(plot_df['Date'])
                
                # Create smooth x axis
                date_nums_smooth = np.linspace(date_nums.min(), date_nums.max(), 300)
                
                # Spline interpolation
                spl = make_interp_spline(date_nums, plot_df['Close'], k=3)
                close_smooth = spl(date_nums_smooth)
                
                # Convert back to dates
                dates_smooth = pd.to_datetime(date_nums_smooth)
                
                # Plot smooth line
                sns.lineplot(x=dates_smooth, y=close_smooth, ax=ax, color='#4c72b0', linewidth=2.5, label='Historical')
                
                # Plot original points as scatter to show data
                sns.scatterplot(data=plot_df, x='Date', y='Close', ax=ax, color='#4c72b0', s=60, marker='o')
                
            except ImportError:
                # Fallback if scipy is not available
                sns.lineplot(data=plot_df, x='Date', y='Close', ax=ax, label='Historical', color='#4c72b0', linewidth=2.5, marker='o', markersize=8)
            
            # Add prediction point
            last_date = plot_df['Date'].iloc[-1]
            try:
                next_date = last_date + datetime.timedelta(days=1)
            except:
                last_date = pd.to_datetime(last_date)
                next_date = last_date + datetime.timedelta(days=1)

            # 2. Connection Line - Dashed, grey, subtle
            connector_df = pd.DataFrame({
                'Date': [last_date, next_date],
                'Close': [plot_df['Close'].iloc[-1], prediction]
            })
            sns.lineplot(data=connector_df, x='Date', y='Close', ax=ax, color='grey', linestyle='--', linewidth=2, label='예측 흐름')

            # 3. Prediction Point (No Uncertainty displayed)
            # Plot prediction point
            ax.scatter([next_date], [prediction], color='#c44e52', s=100, zorder=5, label='예측')
            
            # Add simple text annotation (Prediction only)
            text_str = f"{prediction:,.0f} KRW"
            ax.text(next_date, prediction * 1.002, text_str, ha='center', va='bottom', fontsize=12, color='#c44e52', fontweight='bold')

            # Polish Layout
            # Simple Title
            ax.set_title(f"{ticker} Price Prediction", fontsize=20, fontweight='bold', pad=20) 
            
            # Remove unnecessary labels
            ax.set_ylabel("")
            ax.set_xlabel("")
            
            # Remove top and right spines
            sns.despine(trim=True, offset=10)
            
            # Format X-axis dates cleaner
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.xticks(rotation=0, fontsize=12) # Horizontal dates if possible, or slight rotation
            plt.yticks(fontsize=12)
            
            # Remove legend for simplicity (Annotation explains the prediction)
            if ax.get_legend():
                ax.get_legend().remove()
            
            ax.grid(axis='y', linestyle='--', alpha=0.3) 
            
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
