import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import time

# 下载VIX和油价数据，从2000年开始
def download_with_retry(ticker, start, end, retries=3, delay=5):
    for i in range(retries):
        try:
            data = yf.download(ticker, start=start, end=end)
            if not data.empty:
                return data
        except Exception as e:
            print(f"尝试 {i+1} 次失败: {e}")
            time.sleep(delay)
    return pd.DataFrame()  # 返回空DataFrame

vix_data = download_with_retry('^VIX', start='2000-01-01', end='2023-01-01')
oil_data = download_with_retry('CL=F', start='2000-01-01', end='2023-01-01')

# 检查数据是否下载成功
if vix_data.empty or oil_data.empty:
    print("数据下载失败，请稍后重试。")
    exit()

# 选择收盘价
vix_data = vix_data[['Close']].rename(columns={'Close': 'VIX'})
oil_data = oil_data[['Close']].rename(columns={'Close': 'Oil_Price'})

# 合并数据
merged_data = pd.merge(vix_data, oil_data, left_index=True, right_index=True)

# 检查合并后的数据是否有效
if merged_data.empty:
    print("合并后的数据为空，无法计算相关性。")
    exit()

# 计算皮尔逊相关系数
correlation, p_value = pearsonr(merged_data['VIX'], merged_data['Oil_Price'])
print(f"皮尔逊相关系数: {correlation}, p值: {p_value}")

# 可视化
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Oil_Price', y='VIX', data=merged_data)
plt.title('VIX vs 油价 (2000-2023)')
plt.xlabel('油价')
plt.ylabel('VIX')
plt.show()
