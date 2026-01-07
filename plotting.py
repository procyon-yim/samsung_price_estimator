import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
from scipy.interpolate import make_interp_spline

def plot_prediction(history_df, prediction, ticker):
    """
    과거 주가와 예측된 미래 주가를 시각화합니다.
    
    Args:
        history_df (pd.DataFrame): 과거 주가 데이터를 담은 DataFrame. 
                                   'Date'와 'Close' 컬럼이 있어야 합니다 (또는 Date 인덱스와 'Close').
        prediction (float): 예측된 주가.
        ticker (str): 제목에 표시될 종목 코드.
        
    Returns:
        matplotlib.figure.Figure: 생성된 플롯 figure 객체.
    """
    
    # Seaborn으로 시각화
    sns.set_theme(style="white", context="talk") # 더 큰 폰트와 흰색 배경을 위한 깔끔한 "talk" 컨텍스트
    
    # 시각화를 위한 데이터프레임 생성
    plot_df = history_df.reset_index()
    # 인덱스가 날짜라고 가정
    plot_df.columns = ['Date', 'Close'] if 'Date' not in plot_df.columns else plot_df.columns
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 1. 과거 데이터 - 부드러운 선, 적절한 색상
    # 부드러운 선을 위해 스플라인 보간법 사용
    try:
        # 보간을 위해 날짜를 숫자로 변환
        # 날짜 형식 확인
        plot_df['Date'] = pd.to_datetime(plot_df['Date'])
        date_nums = pd.to_numeric(plot_df['Date'])
        
        # 부드러운 X축 생성
        date_nums_smooth = np.linspace(date_nums.min(), date_nums.max(), 300)
        
        # 스플라인 보간
        spl = make_interp_spline(date_nums, plot_df['Close'], k=3)
        close_smooth = spl(date_nums_smooth)
        
        # 다시 날짜로 변환
        dates_smooth = pd.to_datetime(date_nums_smooth)
        
        # 부드러운 선 그리기
        sns.lineplot(x=dates_smooth, y=close_smooth, ax=ax, color='#4c72b0', linewidth=2.5, label='Historical')
        
        # 원본 데이터를 산점도로 표시
        sns.scatterplot(data=plot_df, x='Date', y='Close', ax=ax, color='#4c72b0', s=60, marker='o')
        
    except ImportError:
        # scipy가 없는 경우 대체 방법
        sns.lineplot(data=plot_df, x='Date', y='Close', ax=ax, label='Historical', color='#4c72b0', linewidth=2.5, marker='o', markersize=8)
    except Exception as e:
         # 스무딩 중 다른 오류 발생 시 대체 방법
         sns.lineplot(data=plot_df, x='Date', y='Close', ax=ax, label='Historical', color='#4c72b0', linewidth=2.5, marker='o', markersize=8)

    
    # 예측 지점 추가
    last_date = plot_df['Date'].iloc[-1]
    try:
        next_date = last_date + datetime.timedelta(days=1)
    except:
        last_date = pd.to_datetime(last_date)
        next_date = last_date + datetime.timedelta(days=1)

    # 2. 연결선 - 점선, 회색, 은은하게
    connector_df = pd.DataFrame({
        'Date': [last_date, next_date],
        'Close': [plot_df['Close'].iloc[-1], prediction]
    })
    sns.lineplot(data=connector_df, x='Date', y='Close', ax=ax, color='grey', linestyle='--', linewidth=2, label='예측 흐름')

    # 3. 예측 지점 (불확실성 미표시)
    # 예측 지점 표시
    ax.scatter([next_date], [prediction], color='#c44e52', s=100, zorder=5, label='예측')
    
    # 간단한 텍스트 주석 추가 (예측값만)
    text_str = f"{prediction:,.0f} KRW"
    ax.text(next_date, prediction * 1.002, text_str, ha='center', va='bottom', fontsize=12, color='#c44e52', fontweight='bold')

    # 레이아웃 다듬기
    # 간단한 제목
    ax.set_title(f"{ticker} Price Prediction", fontsize=20, fontweight='bold', pad=20) 
    
    # 불필요한 라벨 제거
    ax.set_ylabel("")
    ax.set_xlabel("")
    
    # 상단 및 우측 테두리 제거
    sns.despine(trim=True, offset=10)
    
    # X축 날짜 형식 깔끔하게
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=0, fontsize=12) # 가능하면 날짜를 가로로, 아니면 약간 회전
    plt.yticks(fontsize=12)
    
    # 단순화를 위해 범례 제거 (주석이 예측 설명)
    if ax.get_legend():
        ax.get_legend().remove()
    
    ax.grid(axis='y', linestyle='--', alpha=0.3) 
    
    return fig
