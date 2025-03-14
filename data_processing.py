import pandas as pd

# 1. 读取vix.xlsx中的数据
df = pd.read_excel('vix.xlsx')

# 2. 填充空白数据（使用前一日数据填充）
df.ffill(inplace=True)

# 3. 将补充后的数据重新写回vix.xlsx
df.to_excel('vix.xlsx', index=False)
