# 주가 예측 서비스 (Stock Price Estimator)

딥러닝(LSTM)을 사용하여 주가를 예측하고 AI 기반의 투자 성향 분석 및 기능(Feature) 추천을 제공하는 Streamlit 웹 애플리케이션입니다.

## 주요 기능

- **주가 예측**: LSTM(Long Short-Term Memory) 신경망을 사용하여 다음 날의 종가를 예측합니다.
- **인터랙티브 시각화**: Seaborn과 Matplotlib을 사용하여 과거 데이터와 예측 추세를 시각화합니다.
- **AI 종목 검색**: 자연어로 회사 이름(예: "삼성전자", "테슬라")을 입력하여 주식 종목 코드를 검색합니다. (LangChain & OpenAI 기반)
- **AI 맞춤형 기능(Feature) 추천**: 투자 성향(예: "안전 제일", "변동성 큰 단타")을 입력하면 AI가 모델 학습에 가장 적합한 기술적 지표(Feature)를 추천해줍니다.
- **학습 옵션 사용자화**: 모델 학습에 사용할 기능(시가, 고가, 저가, 거래량, 이동평균선 등)과 윈도우 크기(Window Size)를 직접 선택할 수 있습니다.

## 사용 기술

- **Python 3.10+**
- **Streamlit**: 웹 인터페이스
- **TensorFlow / Keras**: LSTM 모델 구현
- **LangChain & OpenAI**: 검색 및 추천 기능을 위한 LLM 연동
- **FinanceDataReader**: 주식 시장 데이터 수집
- **Seaborn & Matplotlib**: 데이터 시각화

## 설치 방법

1.  **저장소 클론:**
    ```bash
    git clone https://github.com/procyon-yim/samsung_price_estimator.git
    cd samsung_price_estimator
    ```

2.  **가상 환경 생성 및 활성화 (필수):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # Windows의 경우: .venv\Scripts\activate
    ```

3.  **의존성 패키지 설치:**
    ```bash
    pip install -r requirements.txt
    ```

## 환경 설정

이 프로젝트는 AI 기능(종목 검색 및 기능 추천)을 위해 OpenAI API Key가 필요합니다.

1.  루트 디렉토리에 `.env` 파일을 생성합니다:
    ```bash
    touch .env
    ```

2.  `.env` 파일에 OpenAI API Key를 추가합니다:
    ```env
    OPENAI_API_KEY=your_sk_key_here
    ```

## 사용 방법

1.  **가상 환경 활성화 (이미 활성화되어 있다면 생략 가능):**
    ```bash
    source .venv/bin/activate
    ```

2.  **Streamlit 애플리케이션 실행:**
    ```bash
    streamlit run app.py
    ```

3.  브라우저에서 앱이 열립니다 (보통 `http://localhost:8501`).

### 사용 가이드:
1.  **종목 검색**: 사이드바의 "종목 검색" 기능을 사용하여 회사 이름으로 종목 코드를 찾습니다.
2.  **기능(Features) 선택**:
    - `3MA`(3일 이동평균), `Volume`(거래량) 등의 기능을 수동으로 선택합니다.
    - 또는 **AI 투자 성향 분석 및 추천** 기능을 사용하여 투자 성향에 맞는 기능을 추천받습니다.
3.  **주가 예측**: "Predict Price" 버튼을 클릭하여 실시간 데이터로 모델을 학습시키고 예측 결과를 확인합니다.

## 라이선스

쓸거면 1000000원
