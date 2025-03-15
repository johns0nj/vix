import dash
from dash import dcc, html
import plotly.graph_objects as go
from data_processing import process_vix_data
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression

# 从 data_processing.py 中读取处理后的 DataFrame
processed_df = process_vix_data('vix.xlsx')

# 确保日期列是日期类型并设置为索引
processed_df['Date'] = pd.to_datetime(processed_df['Date'])
processed_df.set_index('Date', inplace=True)

def find_support_levels(df, window=20, threshold=0.02):
    """识别支撑位"""
    prices = df['CO1 Comdty  (L1)'].values
    local_mins = argrelextrema(prices, np.less, order=window)[0]
    
    support_levels = {}
    for i in local_mins:
        price = prices[i]
        # 将价格四舍五入到最近的整数
        rounded_price = round(price)
        
        if rounded_price not in support_levels:
            support_levels[rounded_price] = {
                'count': 1,
                'dates': [df.index[i]],
                'duration': 0,
                'importance': 0
            }
        else:
            support_levels[rounded_price]['count'] += 1
            support_levels[rounded_price]['dates'].append(df.index[i])
    
    # 计算支撑位的持续时间和重要性
    for price, info in support_levels.items():
        if len(info['dates']) > 1:
            duration = (info['dates'][-1] - info['dates'][0]).days
            info['duration'] = duration
            # 重要性 = 触及次数 * 持续时间（天）/ 1000
            info['importance'] = info['count'] * duration / 1000
    
    return support_levels

def analyze_vix_breakout(df, support_levels):
    """分析支撑位跌破后VIX的变化"""
    breakout_analysis = []
    
    for price, info in support_levels.items():
        if info['importance'] < 1:  # 忽略不重要的支撑位
            continue
            
        for i in range(len(df) - 1):
            if df['CO1 Comdty  (L1)'].iloc[i] >= price and df['CO1 Comdty  (L1)'].iloc[i+1] < price:
                # 找到跌破支撑位的点
                breakout_date = df.index[i+1]
                # 计算VIX在跌破后30天内的最大变化幅度
                if i + 31 < len(df):
                    vix_before = df['VIX Index  (R1)'].iloc[i+1]
                    vix_after_max = df['VIX Index  (R1)'].iloc[i+1:i+31].max()
                    vix_change = ((vix_after_max - vix_before) / vix_before) * 100
                    
                    breakout_analysis.append({
                        'support_price': price,
                        'importance': info['importance'],
                        'breakout_date': breakout_date,
                        'vix_change': vix_change
                    })
    
    return pd.DataFrame(breakout_analysis)

# 分析支撑位和VIX变化
support_levels = find_support_levels(processed_df)
breakout_df = analyze_vix_breakout(processed_df, support_levels)

# 在创建 Dash 布局之前，添加单日价格变化的回归分析
# 计算单日价格变化
processed_df['VIX_daily_change'] = processed_df['VIX Index  (R1)'].pct_change() * 100
processed_df['CO1_daily_change'] = processed_df['CO1 Comdty  (L1)'].pct_change() * 100

# 删除缺失值
daily_changes_df = processed_df.dropna(subset=['VIX_daily_change', 'CO1_daily_change'])

# 创建单日变化回归分析图表
fig3 = go.Figure()

# 添加散点图
fig3.add_trace(go.Scatter(
    x=daily_changes_df['CO1_daily_change'],
    y=daily_changes_df['VIX_daily_change'],
    mode='markers',
    name='日变化率',
    marker=dict(
        size=6,
        color='purple',
        opacity=0.5
    ),
    hovertemplate='CO1变化率: %{x:.2f}%<br>VIX变化率: %{y:.2f}%<br>日期: %{text}<extra></extra>',
    text=daily_changes_df.index.strftime('%Y年%m月%d日')
))

# 添加回归线
X = daily_changes_df['CO1_daily_change'].values.reshape(-1, 1)
y = daily_changes_df['VIX_daily_change'].values
daily_reg = LinearRegression().fit(X, y)
y_pred = daily_reg.predict(X)

fig3.add_trace(go.Scatter(
    x=daily_changes_df['CO1_daily_change'],
    y=y_pred,
    mode='lines',
    name=f'回归线 (R² = {daily_reg.score(X, y):.3f})',
    line=dict(color='red')
))

# 设置回归分析图表布局
fig3.update_layout(
    title='VIX与CO1日变化率的关系',
    xaxis_title='CO1日变化率 (%)',
    yaxis_title='VIX日变化率 (%)',
    plot_bgcolor='white',
    paper_bgcolor='white',
    showlegend=True,
    hovermode='closest'
)

# 在创建 Dash 布局之前，添加周度价格变化的回归分析
# 计算周度价格变化
processed_df['VIX_weekly_change'] = processed_df['VIX Index  (R1)'].pct_change(periods=5) * 100  # 5个交易日
processed_df['CO1_weekly_change'] = processed_df['CO1 Comdty  (L1)'].pct_change(periods=5) * 100

# 删除缺失值
weekly_changes_df = processed_df.dropna(subset=['VIX_weekly_change', 'CO1_weekly_change'])

# 创建周度变化回归分析图表
fig4 = go.Figure()

# 添加散点图
fig4.add_trace(go.Scatter(
    x=weekly_changes_df['CO1_weekly_change'],
    y=weekly_changes_df['VIX_weekly_change'],
    mode='markers',
    name='周变化率',
    marker=dict(
        size=6,
        color='green',
        opacity=0.5
    ),
    hovertemplate='CO1周变化率: %{x:.2f}%<br>VIX周变化率: %{y:.2f}%<br>日期: %{text}<extra></extra>',
    text=weekly_changes_df.index.strftime('%Y年%m月%d日')
))

# 添加回归线
X_weekly = weekly_changes_df['CO1_weekly_change'].values.reshape(-1, 1)
y_weekly = weekly_changes_df['VIX_weekly_change'].values
weekly_reg = LinearRegression().fit(X_weekly, y_weekly)
y_weekly_pred = weekly_reg.predict(X_weekly)

fig4.add_trace(go.Scatter(
    x=weekly_changes_df['CO1_weekly_change'],
    y=y_weekly_pred,
    mode='lines',
    name=f'回归线 (R² = {weekly_reg.score(X_weekly, y_weekly):.3f})',
    line=dict(color='red')
))

# 设置回归分析图表布局
fig4.update_layout(
    title='VIX与CO1周变化率的关系',
    xaxis_title='CO1周变化率 (%)',
    yaxis_title='VIX周变化率 (%)',
    plot_bgcolor='white',
    paper_bgcolor='white',
    showlegend=True,
    hovermode='closest'
)

# 在创建 Dash 布局之前，添加VIX大波动分析
# 筛选VIX日变化超过20%的数据
large_volatility_df = daily_changes_df[abs(daily_changes_df['VIX_daily_change']) > 20].copy()

# 创建VIX大波动回归分析图表
fig5 = go.Figure()

# 添加散点图
fig5.add_trace(go.Scatter(
    x=large_volatility_df['CO1_daily_change'],
    y=large_volatility_df['VIX_daily_change'],
    mode='markers',
    name='VIX大波动日',
    marker=dict(
        size=8,
        color='orange',
        opacity=0.7,
        symbol='diamond'
    ),
    hovertemplate='日期: %{text}<br>CO1变化率: %{x:.2f}%<br>VIX变化率: %{y:.2f}%<extra></extra>',
    text=large_volatility_df.index.strftime('%Y年%m月%d日')
))

# 添加回归线
if len(large_volatility_df) > 1:  # 确保有足够的数据点进行回归
    X_large = large_volatility_df['CO1_daily_change'].values.reshape(-1, 1)
    y_large = large_volatility_df['VIX_daily_change'].values
    large_reg = LinearRegression().fit(X_large, y_large)
    y_large_pred = large_reg.predict(X_large)

    fig5.add_trace(go.Scatter(
        x=large_volatility_df['CO1_daily_change'],
        y=y_large_pred,
        mode='lines',
        name=f'回归线 (R² = {large_reg.score(X_large, y_large):.3f})',
        line=dict(color='red', width=2)
    ))

# 设置图表布局
fig5.update_layout(
    title='VIX大波动日(>20%)与CO1变化率的关系',
    xaxis_title='CO1日变化率 (%)',
    yaxis_title='VIX日变化率 (%)',
    plot_bgcolor='white',
    paper_bgcolor='white',
    showlegend=True,
    hovermode='closest'
)

# 创建 Dash 应用
app = dash.Dash(__name__)

# 创建主图表
fig1 = go.Figure()

# 添加 VIX 数据
fig1.add_trace(go.Scatter(
    x=processed_df.index,
    y=processed_df['VIX Index  (R1)'],
    name='VIX',
    line=dict(color='blue'),
    hovertemplate='%{x|%Y年%m月%d日}<br>VIX: %{y:.2f}<extra></extra>'
))

# 添加 CO1 数据（倒轴显示）
fig1.add_trace(go.Scatter(
    x=processed_df.index,
    y=processed_df['CO1 Comdty  (L1)'],
    name='CO1',
    line=dict(color='red'),
    yaxis='y2',
    hovertemplate='%{x|%Y年%m月%d日}<br>CO1: %{y:.2f}<extra></extra>'
))

# 设置主图表布局
fig1.update_layout(
    title='VIX vs CO1',
    xaxis=dict(
        title='日期',
        type='date',
        tickformat='%Y',
        tickangle=45,
        dtick='M12',
        showgrid=True,
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikecolor='grey',
        spikethickness=1,
        hoverformat='%Y年%m月%d日'
    ),
    yaxis=dict(
        title='VIX',
        side='left',
        gridcolor='lightgrey',
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikecolor='grey',
        spikethickness=1
    ),
    yaxis2=dict(
        title='CO1',
        side='right',
        overlaying='y',
        autorange="reversed",
        gridcolor='lightgrey',
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikecolor='grey',
        spikethickness=1
    ),
    legend=dict(x=0.1, y=0.9),
    plot_bgcolor='white',
    paper_bgcolor='white',
    hovermode='x unified',
    hoverdistance=100,
    spikedistance=1000
)

# 更新 Dash 布局
app.layout = html.Div([
    dcc.Graph(
        figure=fig1,
        style={'height': '600px'},
        config={
            'displayModeBar': True,
            'scrollZoom': True,
            'showTips': True
        }
    ),
    dcc.Graph(
        figure=fig3,
        style={'height': '400px'},
        config={
            'displayModeBar': True,
            'scrollZoom': True,
            'showTips': True
        }
    ),
    dcc.Graph(
        figure=fig4,
        style={'height': '400px'},
        config={
            'displayModeBar': True,
            'scrollZoom': True,
            'showTips': True
        }
    ),
    dcc.Graph(
        figure=fig5,
        style={'height': '400px'},
        config={
            'displayModeBar': True,
            'scrollZoom': True,
            'showTips': True
        }
    ),
    # 更新统计表格，添加大波动分析结果
    html.Div([
        html.H3('回归分析统计结果', style={
            'textAlign': 'center',
            'color': '#2c3e50',
            'marginBottom': '20px'
        }),
        html.Table([
            html.Thead(
                html.Tr([
                    html.Th('统计指标', style={'backgroundColor': '#f8f9fa', 'padding': '12px'}),
                    html.Th('日度回归', style={'backgroundColor': '#f8f9fa', 'padding': '12px'}),
                    html.Th('周度回归', style={'backgroundColor': '#f8f9fa', 'padding': '12px'}),
                    html.Th('大波动日回归', style={'backgroundColor': '#f8f9fa', 'padding': '12px'})
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td('样本数量', style={'padding': '10px', 'borderBottom': '1px solid #dee2e6'}),
                    html.Td(f'{len(daily_changes_df)}', style={'padding': '10px', 'borderBottom': '1px solid #dee2e6'}),
                    html.Td(f'{len(weekly_changes_df)}', style={'padding': '10px', 'borderBottom': '1px solid #dee2e6'}),
                    html.Td(f'{len(large_volatility_df)}', style={'padding': '10px', 'borderBottom': '1px solid #dee2e6'})
                ]),
                html.Tr([
                    html.Td('R²值', style={'padding': '10px', 'borderBottom': '1px solid #dee2e6'}),
                    html.Td(f'{daily_reg.score(X, y):.3f}', style={'padding': '10px', 'borderBottom': '1px solid #dee2e6'}),
                    html.Td(f'{weekly_reg.score(X_weekly, y_weekly):.3f}', style={'padding': '10px', 'borderBottom': '1px solid #dee2e6'}),
                    html.Td(f'{large_reg.score(X_large, y_large):.3f}' if len(large_volatility_df) > 1 else 'N/A', 
                           style={'padding': '10px', 'borderBottom': '1px solid #dee2e6'})
                ]),
                html.Tr([
                    html.Td('斜率 (β)', style={'padding': '10px', 'borderBottom': '1px solid #dee2e6'}),
                    html.Td(f'{daily_reg.coef_[0]:.3f}', style={'padding': '10px', 'borderBottom': '1px solid #dee2e6'}),
                    html.Td(f'{weekly_reg.coef_[0]:.3f}', style={'padding': '10px', 'borderBottom': '1px solid #dee2e6'}),
                    html.Td(f'{large_reg.coef_[0]:.3f}' if len(large_volatility_df) > 1 else 'N/A',
                           style={'padding': '10px', 'borderBottom': '1px solid #dee2e6'})
                ]),
                html.Tr([
                    html.Td('截距 (α)', style={'padding': '10px', 'borderBottom': '1px solid #dee2e6'}),
                    html.Td(f'{daily_reg.intercept_:.3f}', style={'padding': '10px', 'borderBottom': '1px solid #dee2e6'}),
                    html.Td(f'{weekly_reg.intercept_:.3f}', style={'padding': '10px', 'borderBottom': '1px solid #dee2e6'}),
                    html.Td(f'{large_reg.intercept_:.3f}' if len(large_volatility_df) > 1 else 'N/A',
                           style={'padding': '10px', 'borderBottom': '1px solid #dee2e6'})
                ])
            ])
        ], style={
            'margin': 'auto',
            'border-collapse': 'collapse',
            'width': '80%',
            'textAlign': 'center',
            'boxShadow': '0 0 20px rgba(0,0,0,0.1)',
            'backgroundColor': 'white',
            'borderRadius': '8px',
            'overflow': 'hidden'
        })
    ], style={
        'margin': '20px',
        'padding': '20px',
        'backgroundColor': '#f8f9fa',
        'borderRadius': '10px'
    })
])

# 添加CSS样式
app.layout.style = {
    'font-family': 'Arial, sans-serif'
}

# 运行应用
if __name__ == '__main__':
    app.run_server(debug=True)
