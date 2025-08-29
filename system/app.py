import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from savings_calculator import SavingsCalculator
import locale

# 한글 폰트 설정
st.set_page_config(
    page_title="적금 추천 시스템",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일 추가
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .rank-badge {
        background-color: #ff6b6b;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 50%;
        font-weight: bold;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

def format_currency(amount):
    """금액을 한국 원화 형식으로 포맷팅"""
    try:
        return f"{amount:,.0f}원"
    except:
        return f"{amount}원"

def format_percentage(rate):
    """비율을 퍼센트 형식으로 포맷팅"""
    try:
        return f"{rate:.2f}%"
    except:
        return f"{rate}%"

def main():
    st.markdown('<h1 class="main-header">💰 적금 추천 시스템</h1>', unsafe_allow_html=True)
    
    # 사이드바 설정
    st.sidebar.markdown("## 📊 적금 조건 설정")
    
    monthly_amount = st.sidebar.number_input(
        "월 적금 금액 (원)",
        min_value=10000,
        max_value=10000000,
        value=500000,
        step=10000,
        help="매월 적금할 금액을 입력하세요"
    )
    
    period_months = st.sidebar.selectbox(
        "적금 기간",
        options=[12, 24, 36, 48, 60],
        format_func=lambda x: f"{x}개월 ({x//12}년)",
        help="적금할 기간을 선택하세요"
    )
    
    # CSV 파일 로드
    try:
        calculator = SavingsCalculator("naver-2025-08-29.csv")
        
        if calculator.df.empty:
            st.error("CSV 파일을 로드할 수 없습니다. 파일 경로를 확인해주세요.")
            return
        
        # 메인 대시보드
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("총 상품 수", f"{len(calculator.df):,}개")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            max_rate = calculator.df['최고금리_수치'].max()
            st.metric("최고 금리", format_percentage(max_rate))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            avg_rate = calculator.df['최고금리_수치'].mean()
            st.metric("평균 금리", format_percentage(avg_rate))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 적금 추천 결과
        st.markdown('<h2 class="sub-header">🏆 최고 수익률 상위 3개 추천</h2>', unsafe_allow_html=True)
        
        recommendations = calculator.get_top_recommendations(monthly_amount, period_months, 3)
        
        if recommendations:
            for rec in recommendations:
                with st.container():
                    st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
                    
                    with col1:
                        st.markdown(f'<div class="rank-badge">{rec["순위"]}</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"**{rec['상품명']}**")
                        st.markdown(f"🏦 {rec['은행명']}")
                        if rec['태그']:
                            tags = [tag for tag in rec['태그'] if pd.notna(tag) and tag != '']
                            if tags:
                                st.markdown(f"🏷️ {', '.join(tags)}")
                    
                    with col3:
                        st.markdown(f"**적용 금리:** {format_percentage(rec['적용금리'])}")
                        st.markdown(f"**최고 금리:** {format_percentage(rec['최고금리'])}")
                        st.markdown(f"**기본 금리:** {format_percentage(rec['기본금리'])}")
                    
                    with col4:
                        st.markdown(f"**총 납입:** {format_currency(rec['총 납입금액'])}")
                        st.markdown(f"**총 수령:** {format_currency(rec['총 수령금액'])}")
                        st.markdown(f"**이자 수익:** {format_currency(rec['이자 수익'])}")
                        st.markdown(f"**수익률:** {format_percentage(rec['수익률'])}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # 차트 섹션
        st.markdown('<h2 class="sub-header">📈 시각화 분석</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 금리 분포 파이 차트
            rate_dist = calculator.get_rate_distribution()
            if rate_dist['labels']:
                fig_pie = px.pie(
                    values=rate_dist['values'],
                    names=rate_dist['labels'],
                    title="금리 구간별 상품 분포",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # 은행별 평균 금리 바 차트
            bank_summary = calculator.get_bank_summary()
            if not bank_summary.empty:
                top_banks = bank_summary.head(10)  # 상위 10개 은행만 표시
                
                fig_bar = px.bar(
                    x=top_banks.index,
                    y=top_banks['평균최고금리'],
                    title="은행별 평균 최고 금리 (상위 10개)",
                    labels={'x': '은행명', 'y': '평균 최고 금리 (%)'},
                    color=top_banks['평균최고금리'],
                    color_continuous_scale='viridis'
                )
                fig_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_bar, use_container_width=True)
        
        # 상세 데이터 테이블
        st.markdown('<h2 class="sub-header">📋 전체 상품 목록</h2>', unsafe_allow_html=True)
        
        # 필터링 옵션
        col1, col2 = st.columns(2)
        
        with col1:
            selected_bank = st.selectbox(
                "은행 선택",
                options=['전체'] + sorted(calculator.df['은행명'].unique().tolist())
            )
        
        with col2:
            min_rate = st.slider(
                "최소 금리 (%)",
                min_value=float(calculator.df['최고금리_수치'].min()),
                max_value=float(calculator.df['최고금리_수치'].max()),
                value=float(calculator.df['최고금리_수치'].min()),
                step=0.1
            )
        
        # 필터링된 데이터
        filtered_df = calculator.df.copy()
        
        if selected_bank != '전체':
            filtered_df = filtered_df[filtered_df['은행명'] == selected_bank]
        
        filtered_df = filtered_df[filtered_df['최고금리_수치'] >= min_rate]
        
        # 표시할 컬럼 선택
        display_columns = ['상품명', '은행명', '최고금리_수치', '기본금리_수치', '태그1']
        
        st.dataframe(
            filtered_df[display_columns].rename(columns={
                '최고금리_수치': '최고금리(%)',
                '기본금리_수치': '기본금리(%)',
                '태그1': '주요태그'
            }).round(2),
            use_container_width=True,
            height=400
        )
        
        # 통계 정보
        st.markdown('<h2 class="sub-header">📊 통계 요약</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("필터링된 상품 수", f"{len(filtered_df):,}개")
        
        with col2:
            if not filtered_df.empty:
                st.metric("평균 최고 금리", format_percentage(filtered_df['최고금리_수치'].mean()))
        
        with col3:
            if not filtered_df.empty:
                st.metric("최고 금리", format_percentage(filtered_df['최고금리_수치'].max()))
        
        with col4:
            if not filtered_df.empty:
                st.metric("최저 금리", format_percentage(filtered_df['최고금리_수치'].min()))
        
    except Exception as e:
        st.error(f"오류가 발생했습니다: {str(e)}")
        st.info("CSV 파일이 올바른 형식인지 확인해주세요.")

if __name__ == "__main__":
    main()

