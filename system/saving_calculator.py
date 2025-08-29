import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

class SavingsCalculator:
    def __init__(self, csv_file: str):
        """적금 계산기 초기화"""
        self.df = self._load_and_clean_data(csv_file)
    
    def _load_and_clean_data(self, csv_file: str) -> pd.DataFrame:
        """CSV 파일을 로드하고 데이터를 정리합니다"""
        # 여러 인코딩을 시도하여 파일 로드
        encodings = ['euc-kr', 'cp949', 'utf-8', 'latin1']  # euc-kr을 첫 번째로 시도
        
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_file, encoding=encoding)
                print(f"✅ CSV 파일 로드 성공 (인코딩: {encoding})")
                break
            except UnicodeDecodeError:
                print(f"❌ 인코딩 {encoding} 실패, 다음 시도...")
                continue
            except Exception as e:
                print(f"❌ 기타 오류 (인코딩 {encoding}): {e}")
                continue
        else:
            print("❌ 모든 인코딩 시도 실패")
            return pd.DataFrame()
        
        try:
            print(f"원본 데이터: {len(df)}행, {len(df.columns)}컬럼")
            print("원본 컬럼명:", df.columns.tolist())
            
            # 컬럼명 정리
            df.columns = ['상품명', '은행명', '최고금리', '최고금리_수치', '기본금리', '태그1', '태그2', '태그3', 'blind']
            
            # 금리 컬럼을 숫자로 변환
            df['최고금리_수치'] = pd.to_numeric(df['최고금리_수치'], errors='coerce')
            df['기본금리_수치'] = df['기본금리'].str.extract(r'(\d+\.?\d*)').astype(float)
            
            print(f"금리 변환 후: 최고금리_수치 NaN: {df['최고금리_수치'].isna().sum()}, 기본금리_수치 NaN: {df['기본금리_수치'].isna().sum()}")
            
            # NaN 값 제거
            df = df.dropna(subset=['최고금리_수치', '기본금리_수치'])
            
            print(f"✅ 데이터 정리 완료: {len(df)}개 상품")
            print(f"금리 범위: {df['최고금리_수치'].min():.2f}% ~ {df['최고금리_수치'].max():.2f}%")
            return df
        except Exception as e:
            print(f"❌ 데이터 정리 오류: {e}")
            return pd.DataFrame()
    
    def calculate_savings_return(self, monthly_amount: int, period_months: int, 
                               interest_rate: float) -> Dict[str, float]:
        """적금 수익을 계산합니다"""
        total_principal = monthly_amount * period_months
        
        # 복리 계산 (월 복리)
        monthly_rate = interest_rate / 100 / 12
        total_amount = 0
        
        for month in range(1, period_months + 1):
            months_invested = period_months - month + 1
            monthly_interest = monthly_amount * monthly_rate * months_invested
            
            total_amount += monthly_amount + monthly_interest
        
        interest_earned = total_amount - total_principal
        
        return {
            '총 납입금액': total_principal,
            '총 수령금액': total_amount,
            '이자 수익': interest_earned,
            '수익률': (interest_earned / total_principal) * 100
        }
    
    def get_top_recommendations(self, monthly_amount: int, period_months: int, 
                               top_n: int = 3) -> List[Dict]:
        """최고 수익률 상위 N개 적금 상품을 추천합니다"""
        if self.df.empty:
            return []
        
        recommendations = []
        
        for _, row in self.df.iterrows():
            # 최고금리와 기본금리 중 높은 것을 사용
            best_rate = max(row['최고금리_수치'], row['기본금리_수치'])
            
            # 적금 수익 계산
            result = self.calculate_savings_return(monthly_amount, period_months, best_rate)
            
            recommendation = {
                '순위': 0,
                '상품명': row['상품명'],
                '은행명': row['은행명'],
                '적용금리': best_rate,
                '최고금리': row['최고금리_수치'],
                '기본금리': row['기본금리_수치'],
                '총 납입금액': result['총 납입금액'],
                '총 수령금액': result['총 수령금액'],
                '이자 수익': result['이자 수익'],
                '수익률': result['수익률'],
                '태그': [row['태그1'], row['태그2'], row['태그3']] if pd.notna(row['태그1']) else []
            }
            
            recommendations.append(recommendation)
        
        # 수익률 기준으로 정렬하고 상위 N개 선택
        recommendations.sort(key=lambda x: x['수익률'], reverse=True)
        
        for i, rec in enumerate(recommendations[:top_n]):
            rec['순위'] = i + 1
        
        return recommendations[:top_n]
    
    def get_bank_summary(self) -> pd.DataFrame:
        """은행별 적금 상품 요약 정보를 제공합니다"""
        if self.df.empty:
            return pd.DataFrame()
        
        bank_summary = self.df.groupby('은행명').agg({
            '상품명': 'count',
            '최고금리_수치': ['mean', 'max'],
            '기본금리_수치': ['mean', 'max']
        }).round(2)
        
        bank_summary.columns = ['상품수', '평균최고금리', '최고금리', '평균기본금리', '최고기본금리']
        bank_summary = bank_summary.sort_values('평균최고금리', ascending=False)
        
        return bank_summary
    
    def get_rate_distribution(self) -> Dict[str, List]:
        """금리 분포 정보를 제공합니다"""
        if self.df.empty:
            return {'labels': [], 'values': []}
        
        # 금리 구간별 상품 수 계산
        rate_bins = [0, 1.5, 2.0, 2.5, 3.0, 10]
        rate_labels = ['~1.5%', '1.5~2.0%', '2.0~2.5%', '2.5~3.0%', '3.0%+']
        
        rate_counts = pd.cut(self.df['최고금리_수치'], bins=rate_bins, labels=rate_labels).value_counts()
        
        return {
            'labels': rate_counts.index.tolist(),
            'values': rate_counts.values.tolist()
        }
