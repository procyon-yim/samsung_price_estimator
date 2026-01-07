import streamlit as st
import datetime
import pandas as pd
import train
from plotting import plot_prediction
import base64   
import os
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 세션 상태에 티커가 없으면 초기화
if 'ticker' not in st.session_state:
    st.session_state.ticker = "005930"

st.title("주가 예측 서비스")

all_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change', '3MA', '5MA']

st.sidebar.header("설정")

# --- 종목 검색 기능 ---
st.sidebar.subheader("종목 검색")
company_name = st.sidebar.text_input("회사 이름 (예: 삼성전자)", key="company_search")
if st.sidebar.button("종목 코드 찾기"):
    if not company_name:
        st.sidebar.warning("회사 이름을 입력해주세요.")
    else:
        try:
            # API 키 확인
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

# --- AI 투자 성향 분석 및 추천 ---
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
                
                # 응답 파싱
                import ast
                recommended_features = ast.literal_eval(response.content.strip())
                
                # 검증
                valid_features = [f for f in recommended_features if f in all_features]
                
                if valid_features:
                    st.session_state['selected_features'] = valid_features
                    st.sidebar.success(f"추천된 기능: {valid_features}")
                    st.rerun()
                else:
                    st.sidebar.error("적절한 기능을 찾지 못했습니다.")
                    
        except Exception as e:
            st.sidebar.error(f"오류 발생: {e}")

# 1. 입력 설정
# 값 유지를 위해 세션 상태 사용
ticker = st.sidebar.text_input("종목 코드", value=st.session_state.ticker)
# 사용자가 수동으로 변경하면 세션 상태 업데이트
if ticker != st.session_state.ticker:
    st.session_state.ticker = ticker

# 세션 상태에 없으면 기본값 초기화
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
            
            # 새로운 함수를 사용하여 시각화
            fig = plot_prediction(history_df, prediction, ticker)
            
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
