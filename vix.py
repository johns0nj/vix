import dash
from dash import dcc, html
import plotly.graph_objects as go
from data_processing import process_vix_data
import pandas as pd

# 从 data_processing.py 中读取处理后的 DataFrame
processed_df = process_vix_data('vix.xlsx')

# 确保日期列是日期类型并设置为索引
processed_df['Date'] = pd.to_datetime(processed_df['Date'])
processed_df.set_index('Date', inplace=True)

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
    )
])

# 添加CSS样式
app.layout.style = {
    'font-family': 'Arial, sans-serif'
}

# 运行应用
if __name__ == '__main__':
    app.run_server(debug=True)
