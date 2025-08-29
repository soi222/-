import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 페이지 설정
st.set_page_config(
    page_title="한국산업은행 예금정보 머신러닝 분석",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_and_process_data():
    """데이터 로드 및 전처리"""
    try:
        df = pd.read_csv('한국산업은행_예금정보.csv', encoding='utf-8')
        st.success("데이터 로드 성공! 🎉")
    except Exception as e:
        st.error(f"파일 읽기 오류: {e}")
        return None
    
    # 데이터 전처리
    df['가입대상_정리'] = df['jinTgtCone'].str.replace('[', '').str.replace(']', '')
    df['가입목적_정리'] = df['prdJinPpo']
    df['가입채널_정리'] = df['prdJinChnCone'].str.replace('[', '').str.replace(']', '')
    
    def extract_rate(rate_str):
        if pd.isna(rate_str) or rate_str == '':
            return 0.0
        rate_match = re.search(r'(\d+\.?\d*)', str(rate_str))
        if rate_match:
            return float(rate_match.group(1))
        return 0.0
    
    df['금리_숫자'] = df['hitIrtCndCone'].apply(extract_rate)
    
    # 특성 엔지니어링
    target_mapping = {
        '개인': 1, '기업': 2, '개인사업자': 3,
        '개인,기업': 4, '개인,기업,개인사업자': 5,
        '기업,개인사업자': 6, '개인,개인사업자': 7
    }
    df['가입대상_코드'] = df['가입대상_정리'].map(target_mapping)
    
    purpose_mapping = {
        '목돈모으기': 1, '목돈굴리기': 2, '입출금자유상품': 3,
        '외화예금': 4, '단기 운용상품': 5
    }
    df['가입목적_코드'] = df['가입목적_정리'].map(purpose_mapping)
    
    channel_mapping = {
        '영업점': 1, '인터넷': 2, '스마트폰': 3,
        '인터넷,영업점': 4, '인터넷,영업점,스마트폰': 5, '인터넷,스마트폰': 6
    }
    df['가입채널_코드'] = df['가입채널_정리'].map(channel_mapping)
    
    def categorize_period_code(period_str):
        if pd.isna(period_str) or period_str == '':
            return 0
        period_str = str(period_str)
        if '1개월' in period_str or '30일' in period_str:
            return 1
        elif '3개월' in period_str or '90일' in period_str:
            return 2
        elif '6개월' in period_str or '180일' in period_str:
            return 3
        elif '1년' in period_str:
            return 4
        elif '3년' in period_str:
            return 5
        elif '5년' in period_str:
            return 6
        elif '제한없음' in period_str:
            return 7
        else:
            return 8
    
    df['가입기간_코드'] = df['prdJinTrmCone'].apply(categorize_period_code)
    df['상품복잡도'] = df['prdOtl'].str.len()
    df['특수목적여부'] = df['prdNm'].str.contains('압류방지|지킴이|부가가치세|거래계좌|보증서|구조화', na=False).astype(int)
    df['KDB브랜드'] = df['prdNm'].str.contains('KDB', na=False).astype(int)
    df['Hi시리즈'] = df['prdNm'].str.contains('Hi', na=False).astype(int)
    df['Dream시리즈'] = df['prdNm'].str.contains('dream', case=False, na=False).astype(int)
    df['정기예금여부'] = df['prdNm'].str.contains('정기예금', na=False).astype(int)
    df['정기적금여부'] = df['prdNm'].str.contains('정기적금', na=False).astype(int)
    df['보통예금여부'] = df['prdNm'].str.contains('보통예금', na=False).astype(int)
    df['외화상품여부'] = df['prdNm'].str.contains('외화', na=False).astype(int)
    df['통장상품여부'] = df['prdNm'].str.contains('통장', na=False).astype(int)
    
    return df

def train_models(df):
    """머신러닝 모델 훈련"""
    # 모델링을 위한 특성 선택
    features_for_modeling = [
        '가입대상_코드', '가입목적_코드', '가입채널_코드', '가입기간_코드',
        '상품복잡도', '특수목적여부', 'KDB브랜드', 'Hi시리즈', 'Dream시리즈',
        '정기예금여부', '정기적금여부', '보통예금여부', '외화상품여부', '통장상품여부'
    ]
    
    df_modeling = df[features_for_modeling + ['금리_숫자']].fillna(0)
    X = df_modeling[features_for_modeling]
    y = df_modeling['금리_숫자']
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 특성 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 랜덤 포레스트 회귀
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train_scaled, y_train)
    y_pred_rf = rf_regressor.predict(X_test_scaled)
    
    # 모델 성능 평가
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    
    # 특성 중요도
    feature_importance = pd.DataFrame({
        '특성': features_for_modeling,
        '중요도': rf_regressor.feature_importances_
    }).sort_values('중요도', ascending=False)
    
    # 고금리 상품 분류 모델
    df['고금리여부'] = (df['금리_숫자'] >= 2.0).astype(int)
    y_class = df['고금리여부']
    
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    X_train_class_scaled = scaler.fit_transform(X_train_class)
    X_test_class_scaled = scaler.transform(X_test_class)
    
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_class_scaled, y_train_class)
    y_pred_class = rf_classifier.predict(X_test_class_scaled)
    
    return {
        'rf_regressor': rf_regressor,
        'rf_classifier': rf_classifier,
        'feature_importance': feature_importance,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred_rf': y_pred_rf,
        'y_test_class': y_test_class,
        'y_pred_class': y_pred_class,
        'mse_rf': mse_rf,
        'r2_rf': r2_rf,
        'X_train': X_train,
        'X_test_class_scaled': X_test_class_scaled,
        'features_for_modeling': features_for_modeling
    }

def main():
    st.title("🏦 한국산업은행 예금정보 머신러닝 분석 대시보드")
    st.markdown("---")
    
    # 사이드바
    st.sidebar.header("📊 분석 옵션")
    analysis_type = st.sidebar.selectbox(
        "분석 유형 선택",
        ["📈 데이터 개요", "🎯 모델 성능", "🔍 특성 분석", "👥 고객 분석", "📊 시각화"]
    )
    
    # 데이터 로드
    with st.spinner("데이터를 로드하고 있습니다..."):
        df = load_and_process_data()
    
    if df is None:
        st.error("데이터를 로드할 수 없습니다.")
        return
    
    # 모델 훈련
    with st.spinner("머신러닝 모델을 훈련하고 있습니다..."):
        model_results = train_models(df)
    
    # 고객 선호도 데이터
    customer_preference_data = df.groupby('가입대상_정리').agg({
        '정기예금여부': 'mean',
        '정기적금여부': 'mean',
        '보통예금여부': 'mean',
        '외화상품여부': 'mean',
        '통장상품여부': 'mean',
        '금리_숫자': 'mean'
    }).round(3)
    
    # 메인 컨텐츠
    if analysis_type == "📈 데이터 개요":
        st.header("📈 데이터 개요")
        
        # 메트릭 카드
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("총 상품 수", f"{len(df):,}개")
        
        with col2:
            st.metric("분석 특성 수", f"{len(model_results['features_for_modeling'])}개")
        
        with col3:
            st.metric("고객 유형 수", f"{df['가입대상_정리'].nunique()}개")
        
        with col4:
            st.metric("가입 목적 수", f"{df['가입목적_정리'].nunique()}개")
        
        # 데이터 구조
        st.subheader("📋 데이터 구조")
        st.dataframe(df.head(10))
        
        # 데이터 통계
        st.subheader("📊 데이터 통계")
        st.write(df.describe())
        
        # 금리 분포 시각화
        st.subheader("💰 금리 분포 분석")
        col1, col2 = st.columns(2)
        
        with col1:
            # 금리 히스토그램
            fig_rate_hist = px.histogram(
                df, 
                x='금리_숫자', 
                nbins=20,
                title="전체 금리 분포",
                labels={'x': '금리 (%)', 'y': '상품 수'},
                color_discrete_sequence=['#2196F3']
            )
            st.plotly_chart(fig_rate_hist, use_container_width=True)
        
        with col2:
            # 금리 박스플롯
            fig_rate_box = px.box(
                df, 
                y='금리_숫자',
                title="금리 분포 (박스플롯)",
                labels={'y': '금리 (%)'},
                color_discrete_sequence=['#4CAF50']
            )
            st.plotly_chart(fig_rate_box, use_container_width=True)
        
        # 금리 인사이트
        st.info(f"""
        **💰 금리 분포 인사이트:**
        - **평균 금리**: {df['금리_숫자'].mean():.3f}%
        - **중앙값 금리**: {df['금리_숫자'].median():.3f}%
        - **최고 금리**: {df['금리_숫자'].max():.3f}%
        - **최저 금리**: {df['금리_숫자'].min():.3f}%
        - **표준편차**: {df['금리_숫자'].std():.3f}%
        """)
        
        # 가입대상별 분석
        st.subheader("👥 가입대상별 분석")
        col1, col2 = st.columns(2)
        
        with col1:
            # 가입대상별 상품 수
            target_counts = df['가입대상_정리'].value_counts()
            fig_target_bar = px.bar(
                x=target_counts.index,
                y=target_counts.values,
                title="가입대상별 상품 수",
                labels={'x': '가입대상', 'y': '상품 수'},
                color=target_counts.values,
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_target_bar, use_container_width=True)
        
        with col2:
            # 가입대상별 평균 금리
            target_rates = df.groupby('가입대상_정리')['금리_숫자'].mean().sort_values(ascending=True)
            fig_target_rates = px.bar(
                x=target_rates.values,
                y=target_rates.index,
                orientation='h',
                title="가입대상별 평균 금리",
                labels={'x': '평균 금리 (%)', 'y': '가입대상'},
                color=target_rates.values,
                color_continuous_scale='plasma'
            )
            st.plotly_chart(fig_target_rates, use_container_width=True)
        
        # 가입대상 인사이트
        st.success(f"""
        **👥 가입대상 인사이트:**
        - **가장 많은 상품**: {target_counts.index[0]} ({target_counts.iloc[0]}개)
        - **가장 높은 평균 금리**: {target_rates.index[-1]} ({target_rates.iloc[-1]:.3f}%)
        - **가장 낮은 평균 금리**: {target_rates.index[0]} ({target_rates.iloc[0]:.3f}%)
        - **상품 다양성**: {len(target_counts)}가지 고객 유형 지원
        """)
        
        # 가입목적별 분석
        st.subheader("🎯 가입목적별 분석")
        col1, col2 = st.columns(2)
        
        with col1:
            # 가입목적별 상품 수
            purpose_counts = df['가입목적_정리'].value_counts()
            fig_purpose_bar = px.bar(
                x=purpose_counts.index,
                y=purpose_counts.values,
                title="가입목적별 상품 수",
                labels={'x': '가입목적', 'y': '상품 수'},
                color=purpose_counts.values,
                color_continuous_scale='inferno'
            )
            st.plotly_chart(fig_purpose_bar, use_container_width=True)
        
        with col2:
            # 가입목적별 평균 금리
            purpose_rates = df.groupby('가입목적_정리')['금리_숫자'].mean().sort_values(ascending=True)
            fig_purpose_rates = px.bar(
                x=purpose_rates.values,
                y=purpose_rates.index,
                orientation='h',
                title="가입목적별 평균 금리",
                labels={'x': '평균 금리 (%)', 'y': '가입목적'},
                color=purpose_rates.values,
                color_continuous_scale='magma'
            )
            st.plotly_chart(fig_purpose_rates, use_container_width=True)
        
        # 가입목적 인사이트
        st.warning(f"""
        **🎯 가입목적 인사이트:**
        - **가장 인기 있는 목적**: {purpose_counts.index[0]} ({purpose_counts.iloc[0]}개 상품)
        - **가장 높은 평균 금리**: {purpose_rates.index[-1]} ({purpose_rates.iloc[-1]:.3f}%)
        - **가장 낮은 평균 금리**: {purpose_rates.index[0]} ({purpose_rates.iloc[0]:.3f}%)
        - **목적별 상품 다양성**: {len(purpose_counts)}가지 가입 목적 지원
        """)
        
        # 상품 유형별 분석
        st.subheader("📦 상품 유형별 분석")
        
        # 상품 유형별 상품 수
        product_types = ['정기예금여부', '정기적금여부', '보통예금여부', '외화상품여부', '통장상품여부']
        product_labels = ['정기예금', '정기적금', '보통예금', '외화상품', '통장상품']
        
        product_counts = []
        product_rates = []
        
        for ptype in product_types:
            count = df[ptype].sum()
            avg_rate = df[df[ptype] == 1]['금리_숫자'].mean()
            product_counts.append(count)
            product_rates.append(avg_rate)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 상품 유형별 개수
            fig_product_counts = px.bar(
                x=product_labels,
                y=product_counts,
                title="상품 유형별 상품 수",
                labels={'x': '상품 유형', 'y': '상품 수'},
                color=product_counts,
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_product_counts, use_container_width=True)
        
        with col2:
            # 상품 유형별 평균 금리
            fig_product_rates = px.bar(
                x=product_labels,
                y=product_rates,
                title="상품 유형별 평균 금리",
                labels={'x': '상품 유형', 'y': '평균 금리 (%)'},
                color=product_rates,
                color_continuous_scale='plasma'
            )
            st.plotly_chart(fig_product_rates, use_container_width=True)
        
        # 상품 유형 인사이트
        st.info(f"""
        **📦 상품 유형 인사이트:**
        - **가장 많은 상품**: {product_labels[product_counts.index(max(product_counts))]} ({max(product_counts)}개)
        - **가장 높은 평균 금리**: {product_labels[product_rates.index(max(product_rates))]} ({max(product_rates):.3f}%)
        - **가장 낮은 평균 금리**: {product_labels[product_rates.index(min(product_rates))]} ({min(product_rates):.3f}%)
        - **상품 포트폴리오**: {len([c for c in product_counts if c > 0])}가지 상품 유형 제공
        """)
        
        # 상관관계 분석
        st.subheader("🔗 주요 특성 상관관계")
        
        # 수치형 특성만 선택
        numeric_features = ['가입대상_코드', '가입목적_코드', '가입채널_코드', '가입기간_코드', 
                          '상품복잡도', '특수목적여부', 'KDB브랜드', 'Hi시리즈', 'Dream시리즈',
                          '정기예금여부', '정기적금여부', '보통예금여부', '외화상품여부', '통장상품여부', '금리_숫자']
        
        correlation_matrix = df[numeric_features].corr()
        
        # 상관관계 히트맵
        fig_corr = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="특성 간 상관관계",
            labels=dict(x="특성", y="특성", color="상관계수"),
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # 상관관계 인사이트
        st.success(f"""
        **🔗 상관관계 인사이트:**
        - **금리와 가장 상관관계 높은 특성**: {correlation_matrix['금리_숫자'].abs().sort_values(ascending=False).index[1]} (상관계수: {correlation_matrix['금리_숫자'].abs().sort_values(ascending=False).iloc[1]:.3f})
        - **가장 강한 양의 상관관계**: {correlation_matrix.unstack().sort_values(ascending=False).index[1]} (상관계수: {correlation_matrix.unstack().sort_values(ascending=False).iloc[1]:.3f})
        - **가장 강한 음의 상관관계**: {correlation_matrix.unstack().sort_values(ascending=True).index[0]} (상관계수: {correlation_matrix.unstack().sort_values(ascending=True).iloc[0]:.3f})
        """)
        
    elif analysis_type == "🎯 모델 성능":
        st.header("🎯 모델 성능 지표")
        
        # 성능 메트릭
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R² 점수", f"{model_results['r2_rf']:.4f}")
        
        with col2:
            st.metric("분류 정확도", f"{model_results['rf_classifier'].score(model_results['X_test_class_scaled'], model_results['y_test_class']):.4f}")
        
        with col3:
            st.metric("MSE", f"{model_results['mse_rf']:.4f}")
        
        with col4:
            st.metric("훈련 데이터", f"{len(model_results['X_train']):,}개")
        
        # 성능 해석
        st.subheader("🚀 성능 해석")
        st.info(f"""
        - **R² 점수 {model_results['r2_rf']:.4f}**: {'매우 우수' if model_results['r2_rf'] > 0.9 else '우수' if model_results['r2_rf'] > 0.8 else '양호' if model_results['r2_rf'] > 0.7 else '보통'}한 예측 성능
        - **분류 정확도 {model_results['rf_classifier'].score(model_results['X_test_class_scaled'], model_results['y_test_class']):.4f}**: 고금리 상품을 {'매우 우수' if model_results['rf_classifier'].score(model_results['X_test_class_scaled'], model_results['y_test_class']) > 0.95 else '우수' if model_results['rf_classifier'].score(model_results['X_test_class_scaled'], model_results['y_test_class']) > 0.9 else '양호'}하게 분류
        - **MSE {model_results['mse_rf']:.4f}**: {'매우 낮은' if model_results['mse_rf'] < 0.01 else '낮은' if model_results['mse_rf'] < 0.05 else '보통' if model_results['mse_rf'] < 0.1 else '높은'} 오차
        """)
        
        # 상세 성능 분석
        st.subheader("📊 상세 모델 성능 분석")
        performance_data = {
            '모델 유형': ['금리 예측 모델', '금리 예측 모델', '고금리 분류 모델', '고금리 분류 모델'],
            '성능 지표': ['R² 점수', 'MSE', '정확도', '훈련 데이터'],
            '값': [
                model_results['r2_rf'],
                model_results['mse_rf'],
                model_results['rf_classifier'].score(model_results['X_test_class_scaled'], model_results['y_test_class']),
                len(model_results['X_train'])
            ],
            '해석': [
                '매우 우수' if model_results['r2_rf'] > 0.9 else '우수' if model_results['r2_rf'] > 0.8 else '양호' if model_results['r2_rf'] > 0.7 else '보통',
                '매우 낮음' if model_results['mse_rf'] < 0.01 else '낮음' if model_results['mse_rf'] < 0.05 else '보통' if model_results['mse_rf'] < 0.1 else '높음',
                '매우 우수' if model_results['rf_classifier'].score(model_results['X_test_class_scaled'], model_results['y_test_class']) > 0.95 else '우수' if model_results['rf_classifier'].score(model_results['X_test_class_scaled'], model_results['y_test_class']) > 0.9 else '양호',
                '충분한 데이터로 안정적인 학습'
            ]
        }
        st.dataframe(pd.DataFrame(performance_data))
        
    elif analysis_type == "🔍 특성 분석":
        st.header("🔍 특성 중요도 분석")
        
        # 특성 중요도 차트
        st.subheader("🔍 특성 중요도 (상위 10개)")
        top_features = model_results['feature_importance'].head(10)
        
        fig = px.bar(
            top_features,
            x='중요도',
            y='특성',
            orientation='h',
            title="특성 중요도 순위",
            color='중요도',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # 특성 설명
        feature_descriptions = {
            '가입대상_코드': '고객 유형 (개인/기업/개인사업자)',
            '가입목적_코드': '상품 가입 목적',
            '가입채널_코드': '가입 채널 (영업점/온라인)',
            '가입기간_코드': '상품 가입 기간',
            '상품복잡도': '상품 설명의 복잡도',
            '특수목적여부': '특수목적 상품 여부',
            'KDB브랜드': 'KDB 브랜드 상품 여부',
            'Hi시리즈': 'Hi 시리즈 상품 여부',
            'Dream시리즈': 'Dream 시리즈 상품 여부',
            '정기예금여부': '정기예금 상품 여부',
            '정기적금여부': '정기적금 상품 여부',
            '보통예금여부': '보통예금 상품 여부',
            '외화상품여부': '외화 상품 여부',
            '통장상품여부': '통장 상품 여부'
        }
        
        # 특성 중요도 테이블
        st.subheader("📋 특성 중요도 상세")
        importance_df = model_results['feature_importance'].head(10).copy()
        importance_df['설명'] = importance_df['특성'].map(feature_descriptions)
        importance_df['순위'] = range(1, len(importance_df) + 1)
        importance_df = importance_df[['순위', '특성', '중요도', '설명']]
        st.dataframe(importance_df)
        
        # 주요 발견사항
        st.subheader("🔍 주요 발견사항")
        st.success(f"""
        - **가장 중요한 특성**: {model_results['feature_importance'].iloc[0]['특성']} (중요도: {model_results['feature_importance'].iloc[0]['중요도']:.4f})
        - **상위 3개 특성**: {', '.join(model_results['feature_importance'].head(3)['특성'].tolist())}
        - **특성 중요도 분포**: 상위 3개 특성이 전체 중요도의 약 {sum(model_results['feature_importance'].head(3)['중요도'])*100:.1f}%를 차지
        """)
        
    elif analysis_type == "👥 고객 분석":
        st.header("👥 고객 유형별 상품 선호도")
        
        # 고객 선호도 차트
        st.subheader("👥 고객 유형별 상품 선호도")
        preference_features = ['정기예금여부', '정기적금여부', '보통예금여부', '외화상품여부', '통장상품여부']
        
        fig = go.Figure()
        
        for i, customer_type in enumerate(customer_preference_data.index):
            values = customer_preference_data.loc[customer_type, preference_features].values
            fig.add_trace(go.Bar(
                name=customer_type,
                x=['정기예금', '정기적금', '보통예금', '외화상품', '통장상품'],
                y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='auto',
            ))
        
        fig.update_layout(
            title="고객 유형별 상품 선호도",
            xaxis_title="상품 유형",
            yaxis_title="선호도",
            barmode='group',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 고객별 평균 금리
        st.subheader("💰 고객 유형별 평균 금리")
        customer_rates = df.groupby('가입대상_정리')['금리_숫자'].mean().sort_values(ascending=True)
        
        fig = px.bar(
            x=customer_rates.values,
            y=customer_rates.index,
            orientation='h',
            title="고객 유형별 평균 금리",
            labels={'x': '평균 금리 (%)', 'y': '고객 유형'},
            color=customer_rates.values,
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # 고객 선호도 테이블
        st.subheader("📊 고객 선호도 상세")
        preference_table = customer_preference_data.copy()
        preference_table.columns = ['정기예금', '정기적금', '보통예금', '외화상품', '통장상품', '평균 금리']
        st.dataframe(preference_table)
        
        # 고객별 특성 분석
        st.subheader("👥 고객별 특성 분석")
        st.info(f"""
        - **개인 고객**: 통장상품 선호도가 높고, 평균 금리가 {customer_preference_data.loc['개인', '금리_숫자']:.3f}%
        - **기업 고객**: 정기예금과 통장상품을 고르게 이용하며, 평균 금리가 {customer_preference_data.loc['기업', '금리_숫자']:.3f}%
        - **개인사업자**: 다양한 상품을 활용하는 경향이 있음
        """)
        
    elif analysis_type == "📊 시각화":
        st.header("📊 종합 시각화")
        
        # 1. 특성 중요도
        st.subheader("🔍 특성 중요도")
        top_features = model_results['feature_importance'].head(10)
        fig1 = px.bar(
            top_features,
            x='중요도',
            y='특성',
            orientation='h',
            color='중요도',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # 2. 실제 vs 예측 금리 비교
        st.subheader("🎯 실제 vs 예측 금리 비교")
        fig2 = px.scatter(
            x=model_results['y_test'],
            y=model_results['y_pred_rf'],
            title="실제 vs 예측 금리 비교",
            labels={'x': '실제 금리 (%)', 'y': '예측 금리 (%)'}
        )
        fig2.add_trace(go.Scatter(
            x=[model_results['y_test'].min(), model_results['y_test'].max()],
            y=[model_results['y_test'].min(), model_results['y_test'].max()],
            mode='lines',
            name='완벽한 예측선',
            line=dict(color='red', dash='dash')
        ))
        st.plotly_chart(fig2, use_container_width=True)
        
        # 3. 예측 오차 분포
        st.subheader("📊 예측 오차 분포")
        errors = model_results['y_test'] - model_results['y_pred_rf']
        fig3 = px.histogram(
            x=errors,
            nbins=25,
            title="예측 오차 분포",
            labels={'x': '예측 오차', 'y': '빈도'}
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # 4. 고객 유형별 상품 선호도
        st.subheader("👥 고객 유형별 상품 선호도")
        preference_features = ['정기예금여부', '정기적금여부', '보통예금여부', '외화상품여부', '통장상품여부']
        
        fig4 = go.Figure()
        for i, customer_type in enumerate(customer_preference_data.index):
            values = customer_preference_data.loc[customer_type, preference_features].values
            fig4.add_trace(go.Bar(
                name=customer_type,
                x=['정기예금', '정기적금', '보통예금', '외화상품', '통장상품'],
                y=values
            ))
        
        fig4.update_layout(
            title="고객 유형별 상품 선호도",
            xaxis_title="상품 유형",
            yaxis_title="선호도",
            barmode='group'
        )
        st.plotly_chart(fig4, use_container_width=True)
        
        # 5. 분류 모델 혼동 행렬
        st.subheader("🎯 고금리 상품 분류 혼동 행렬")
        cm = confusion_matrix(model_results['y_test_class'], model_results['y_pred_class'])
        
        fig5 = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title="고금리 상품 분류 혼동 행렬",
            labels=dict(x="예측", y="실제", color="개수"),
            x=['일반금리', '고금리'],
            y=['일반금리', '고금리']
        )
        st.plotly_chart(fig5, use_container_width=True)
        
        # 6. 상품 유형별 금리 분포
        st.subheader("📦 상품 유형별 금리 분포")
        product_types = ['정기예금여부', '정기적금여부', '보통예금여부', '외화상품여부', '통장상품여부']
        labels = ['정기예금', '정기적금', '보통예금', '외화상품', '통장상품']
        
        fig6 = go.Figure()
        for i, ptype in enumerate(product_types):
            type_data = df[df[ptype] == 1]['금리_숫자']
            if len(type_data) > 0:
                fig6.add_trace(go.Box(
                    y=type_data,
                    name=labels[i],
                    boxpoints='outliers'
                ))
        
        fig6.update_layout(
            title="상품 유형별 금리 분포",
            yaxis_title="금리 (%)"
        )
        st.plotly_chart(fig6, use_container_width=True)
        
        # 7. 전체 금리 분포
        st.subheader("📈 전체 금리 분포")
        fig7 = px.histogram(
            df,
            x='금리_숫자',
            nbins=30,
            title="전체 금리 분포",
            labels={'x': '금리 (%)', 'y': '상품 수'}
        )
        st.plotly_chart(fig7, use_container_width=True)
        
        # 8. 가입대상별 상품 수
        st.subheader("👥 가입대상별 상품 수")
        target_counts = df['가입대상_정리'].value_counts()
        fig8 = px.bar(
            x=target_counts.index,
            y=target_counts.values,
            title="가입대상별 상품 수",
            labels={'x': '가입대상', 'y': '상품 수'},
            color=target_counts.values,
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig8, use_container_width=True)
        
        # 9. 가입목적별 상품 수
        st.subheader("🎯 가입목적별 상품 수")
        purpose_counts = df['가입목적_정리'].value_counts()
        fig9 = px.bar(
            x=purpose_counts.index,
            y=purpose_counts.values,
            title="가입목적별 상품 수",
            labels={'x': '가입목적', 'y': '상품 수'},
            color=purpose_counts.values,
            color_continuous_scale='plasma'
        )
        st.plotly_chart(fig9, use_container_width=True)
    
    # 사이드바에 추가 정보
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 데이터 정보")
    st.sidebar.info(f"""
    - **총 상품 수**: {len(df):,}개
    - **고객 유형**: {df['가입대상_정리'].nunique()}개
    - **가입 목적**: {df['가입목적_정리'].nunique()}개
    - **평균 금리**: {df['금리_숫자'].mean():.2f}%
    """)
    
    st.sidebar.markdown("### 🎯 모델 정보")
    st.sidebar.success(f"""
    - **R² 점수**: {model_results['r2_rf']:.4f}
    - **분류 정확도**: {model_results['rf_classifier'].score(model_results['X_test_class_scaled'], model_results['y_test_class']):.4f}
    - **MSE**: {model_results['mse_rf']:.4f}
    """)

if __name__ == "__main__":
    main()
