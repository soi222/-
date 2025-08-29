import streamlit as st
import pandas as pd
import plotly.express as px
from savings_calculator import SavingsCalculator

# 페이지 설정
st.set_page_config(
    page_title="💰 최고 적금 추천 시스템",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 아름다운 CSS 스타일
st.markdown("""
<style>
    /* 전체 페이지 스타일 */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    /* 헤더 스타일 */
    .main-header {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* 사이드바 스타일 */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50, #34495e);
        color: white;
        padding: 1rem;
        border-radius: 15px;
    }
    
    .sidebar .sidebar-content h2 {
        color: #ecf0f1;
        text-align: center;
        margin-bottom: 1.5rem;
        font-size: 1.5rem;
    }
    
    /* 입력 필드 스타일 */
    .stNumberInput > div > div > input {
        background: rgba(255,255,255,0.9);
        border: 2px solid #3498db;
        border-radius: 10px;
        padding: 0.5rem;
        font-size: 1.1rem;
    }
    
    .stSelectbox > div > div > div {
        background: rgba(255,255,255,0.9);
        border: 2px solid #3498db;
        border-radius: 10px;
    }
    
    /* 메트릭 카드 스타일 */
    .metric-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        flex: 1;
        margin: 0 0.5rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .metric-card .value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    /* 추천 카드 스타일 */
    .recommendations-container {
        margin: 2rem 0;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        border: none;
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .recommendation-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }
    
    .recommendation-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #ff6b6b, #ee5a24, #f39c12, #f1c40f);
    }
    
    .rank-badge {
        position: absolute;
        top: 1rem;
        right: 1rem;
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 5px 15px rgba(255,107,107,0.4);
    }
    
    .card-content {
        display: grid;
        grid-template-columns: 2fr 1fr 1fr;
        gap: 2rem;
        align-items: center;
    }
    
    .product-info h3 {
        color: #2c3e50;
        font-size: 1.5rem;
        margin: 0 0 0.5rem 0;
        font-weight: 700;
    }
    
    .bank-name {
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .tags {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    
    .tag {
        background: linear-gradient(135deg, #3498db, #2980b9);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .rate-info {
        text-align: center;
    }
    
    .rate-value {
        font-size: 2rem;
        font-weight: bold;
        color: #e74c3c;
        margin-bottom: 0.5rem;
    }
    
    .rate-label {
        color: #7f8c8d;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }
    
    .profit-info {
        text-align: center;
    }
    
    .profit-amount {
        font-size: 1.5rem;
        font-weight: bold;
        color: #27ae60;
        margin-bottom: 0.5rem;
    }
    
    .profit-label {
        color: #7f8c8d;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }
    
    /* 차트 컨테이너 스타일 */
    .charts-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .charts-container h2 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* 데이터 테이블 스타일 */
    .data-table-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .data-table-container h2 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        background: linear-gradient(135deg, #3498db, #2980b9);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(52,152,219,0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(52,152,219,0.6);
    }
    
    /* 애니메이션 */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* 반응형 디자인 */
    @media (max-width: 768px) {
        .card-content {
            grid-template-columns: 1fr;
            gap: 1rem;
        }
        
        .metric-container {
            flex-direction: column;
        }
        
        .metric-card {
            margin: 0.5rem 0;
        }
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
    # 아름다운 헤더
    st.markdown("""
    <div class="main-header fade-in-up">
        <h1>💰 최고 적금 추천 시스템</h1>
        <p>AI가 분석한 최고 수익률 적금 상품을 추천해드립니다</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 사이드바 설정
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-content">
            <h2>📊 적금 조건 설정</h2>
        </div>
        """, unsafe_allow_html=True)
        
        monthly_amount = st.number_input(
            "월 적금 금액 (원)",
            min_value=10000,
            max_value=10000000,
            value=500000,
            step=10000,
            help="매월 적금할 금액을 입력하세요"
        )
        
        period_months = st.selectbox(
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
        st.markdown("""
        <div class="metric-container fade-in-up">
            <div class="metric-card">
                <h3>총 상품 수</h3>
                <div class="value">{}</div>
            </div>
            <div class="metric-card">
                <h3>최고 금리</h3>
                <div class="value">{}</div>
            </div>
            <div class="metric-card">
                <h3>평균 금리</h3>
                <div class="value">{}</div>
            </div>
        </div>
        """.format(
            f"{len(calculator.df):,}개",
            format_percentage(calculator.df['최고금리_수치'].max()),
            format_percentage(calculator.df['최고금리_수치'].mean())
        ), unsafe_allow_html=True)
        
        # 적금 추천 결과
        st.markdown('<h2 style="text-align: center; color: #2c3e50; margin: 2rem 0;">🏆 최고 수익률 상위 3개 추천</h2>', unsafe_allow_html=True)
        
        recommendations = calculator.get_top_recommendations(monthly_amount, period_months, 3)
        
        if recommendations:
            for rec in recommendations:
                # 태그 정리
                tags = [tag for tag in rec['태그'] if pd.notna(tag) and tag != '']
                tags_html = ''.join([f'<span class="tag">{tag}</span>' for tag in tags])
                
                st.markdown(f"""
                <div class="recommendation-card fade-in-up">
                    <div class="rank-badge">{rec["순위"]}</div>
                    <div class="card-content">
                        <div class="product-info">
                            <h3>{rec['상품명']}</h3>
                            <div class="bank-name">🏦 {rec['은행명']}</div>
                            <div class="tags">{tags_html}</div>
                        </div>
                        <div class="rate-info">
                            <div class="rate-label">적용 금리</div>
                            <div class="rate-value">{format_percentage(rec['적용금리'])}</div>
                            <div class="rate-label">최고 금리: {format_percentage(rec['최고금리'])}</div>
                            <div class="rate-label">기본 금리: {format_percentage(rec['기본금리'])}</div>
                        </div>
                        <div class="profit-info">
                            <div class="profit-label">총 납입</div>
                            <div class="profit-amount">{format_currency(rec['총 납입금액'])}</div>
                            <div class="profit-label">총 수령</div>
                            <div class="profit-amount">{format_currency(rec['총 수령금액'])}</div>
                            <div class="profit-label">이자 수익</div>
                            <div class="profit-amount">{format_currency(rec['이자 수익'])}</div>
                            <div class="profit-label">수익률</div>
                            <div class="profit-amount">{format_percentage(rec['수익률'])}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # 차트 섹션
        st.markdown("""
        <div class="charts-container fade-in-up">
            <h2>📈 시각화 분석</h2>
        </div>
        """, unsafe_allow_html=True)
        
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
                fig_pie.update_layout(
                    title_font_size=16,
                    title_font_color='#2c3e50',
                    font=dict(size=12)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # 은행별 평균 금리 바 차트
            bank_summary = calculator.get_bank_summary()
            if not bank_summary.empty:
                top_banks = bank_summary.head(10)
                
                fig_bar = px.bar(
                    x=top_banks.index,
                    y=top_banks['평균최고금리'],
                    title="은행별 평균 최고 금리 (상위 10개)",
                    labels={'x': '은행명', 'y': '평균 최고 금리 (%)'},
                    color=top_banks['평균최고금리'],
                    color_continuous_scale='viridis'
                )
                fig_bar.update_layout(
                    xaxis_tickangle=-45,
                    title_font_size=16,
                    title_font_color='#2c3e50',
                    font=dict(size=12)
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        # 상세 데이터 테이블
        st.markdown("""
        <div class="data-table-container fade-in-up">
            <h2>📋 전체 상품 목록</h2>
        </div>
        """, unsafe_allow_html=True)
        
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
        st.markdown('<h2 style="text-align: center; color: #2c3e50; margin: 2rem 0;">📊 통계 요약</h2>', unsafe_allow_html=True)
        
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

