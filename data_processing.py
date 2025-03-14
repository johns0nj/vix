import pandas as pd

def process_vix_data(file_path):
    """
    读取 Excel 文件，填充空白数据，并返回处理后的 DataFrame。
    
    参数:
    file_path (str): Excel 文件的路径
    
    返回:
    pd.DataFrame: 处理后的 DataFrame
    """
    # 1. 读取 Excel 文件中的数据
    df = pd.read_excel(file_path)
    
    # 2. 填充空白数据（使用前一日数据填充）
    df.ffill(inplace=True)
    
    # 3. 返回处理后的 DataFrame
    return df

# 示例用法
if __name__ == '__main__':
    # 处理数据并返回 DataFrame
    processed_df = process_vix_data('vix.xlsx')
    
    # 打印处理后的 DataFrame
    print(processed_df)
    
    # 将补充后的数据重新写回 Excel 文件（可选）
    processed_df.to_excel('vix_processed.xlsx', index=False)
