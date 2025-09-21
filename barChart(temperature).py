import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime

def fetch_html(url):
	headers = {
		'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'
	}
	response = requests.get(url, headers=headers, timeout=15)
	response.encoding = response.apparent_encoding
	return response.text

def parse_weather(html):
	soup = BeautifulSoup(html, 'html.parser')
	data = []
	for row in soup.find_all('tr'):
		tds = row.find_all('td')
		if len(tds) < 6:
			continue
		date_text = tds[0].get_text(strip=True)
		max_temp = tds[1].get_text(strip=True).replace('℃','')
		min_temp = tds[2].get_text(strip=True).replace('℃','')
		sunshine = tds[5].get_text(strip=True).replace('小时','')
		try:
			max_temp = float(max_temp)
			min_temp = float(min_temp)
			sunshine = float(sunshine)
			date_part = date_text[:10]
			date_obj = datetime.strptime(date_part, '%Y-%m-%d')
			date_en = date_obj.strftime('%b %d, %Y')
			data.append({'date': date_en, 'max': max_temp, 'min': min_temp, 'sun': sunshine})
		except Exception:
			continue
	return pd.DataFrame(data)

def plot_weather(df):
	import matplotlib.patches as patches
	from matplotlib.colors import LinearSegmentedColormap
	# 参数
	dates = df['date']
	min_temps = df['min']
	max_temps = df['max']
	n = len(df)
	# 归一化高度
	min_h = (min_temps - min_temps.min()) / (max_temps.max() - min_temps.min() + 1e-6)
	max_h = (max_temps - min_temps.min()) / (max_temps.max() - min_temps.min() + 1e-6)
	# 画布
	fig, ax = plt.subplots(figsize=(18,8))
	# 整个背景填充为深紫色
	fig.patch.set_facecolor('#7B3FA0')
	ax.set_facecolor('#7B3FA0')
	# 柱体参数
	bar_width = 0.6
	base_y = 0.05
	max_bar_height = 0.8
	# 柱体循环
	for i in range(n):
		x = i
		ice_height = min_h.iloc[i]*max_bar_height
		magma_height = max_h.iloc[i]*max_bar_height - ice_height
		# 最低温柱体（普通矩形）
		ice_rect = patches.Rectangle((x-bar_width/2, base_y), bar_width, ice_height, color='#00F0FF', alpha=0.7, zorder=2)
		ax.add_patch(ice_rect)
		# 最高温柱体（普通矩形）
		magma_rect = patches.Rectangle((x-bar_width/2, base_y+ice_height), bar_width, magma_height, color='#FF3B3B', alpha=0.7, zorder=3)
		ax.add_patch(magma_rect)
		# 温差白线为普通直线
		delta = max_temps.iloc[i] - min_temps.iloc[i]
		lw = 2 + 3*(delta/(max_temps.max()-min_temps.min()+1e-6))
		lightning_x = x
		lightning_y = base_y + ice_height + magma_height/2
		ax.plot([lightning_x, lightning_x, lightning_x],
				[lightning_y, lightning_y+0.04, lightning_y-0.04],
				color='white', lw=lw, zorder=4)
		# 温差字号减小
		ax.text(lightning_x, lightning_y+0.05, f'{delta:.1f}', color='white', ha='center', va='bottom', fontsize=9, alpha=0.8)
		# 日期只显示“几号”
		day_num = dates.iloc[i][4:6].replace(',','').replace(' ','')
		if not day_num.isdigit():
			day_num = dates.iloc[i][5:7].replace(',','').replace(' ','')
		ax.text(lightning_x, base_y+max_bar_height+0.03, day_num, color='white', ha='center', va='bottom', fontsize=11, alpha=0.9)
	# 交互气温图（Plotly）
	try:
		import plotly.graph_objects as go
		import plotly.io as pio
		plotly_dates = [d[4:6].replace(',','').replace(' ','') if d[4:6].replace(',','').replace(' ','').isdigit() else d[5:7].replace(',','').replace(' ','') for d in dates]
		hover_text = [f"8月{plotly_dates[i]}日<br>最低温：{min_temps.iloc[i]}℃<br>最高温：{max_temps.iloc[i]}℃<br>温差：{max_temps.iloc[i]-min_temps.iloc[i]:.1f}℃" for i in range(n)]
		fig2 = go.Figure()
		fig2.add_trace(go.Bar(
			x=plotly_dates,
			y=max_temps,
			name='最高温',
			marker_color='#FF6B00',
			opacity=0.7,
			hovertext=hover_text,
			hoverinfo='text',
		))
		fig2.add_trace(go.Bar(
			x=plotly_dates,
			y=min_temps,
			name='最低温',
			marker_color='#00F0FF',
			opacity=0.7,
			hovertext=hover_text,
			hoverinfo='text',
		))
		fig2.update_layout(
			barmode='overlay',
			title='深圳8月气温对抗艺术化交互图',
			plot_bgcolor='rgba(42,10,58,1)',
			paper_bgcolor='rgba(42,10,58,1)',
			font_color='white',
			xaxis_title='日期',
			yaxis_title='温度(℃)',
		)
		pio.write_html(fig2, file='artistic_weather_aug2025_interactive.html', auto_open=False)
	except Exception as e:
		print('Plotly 交互图生成失败:', e)
	# 美化y轴
	ax.set_xlim(-1, n)
	ax.set_ylim(0, 1)
	ax.spines['left'].set_color('#6C3483')
	ax.spines['left'].set_linewidth(2)
	ax.spines['bottom'].set_color('none')
	ax.spines['top'].set_color('none')
	ax.spines['right'].set_color('none')
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('none')
	# y轴刻度与标签
	temp_min = min_temps.min()
	temp_max = max_temps.max()
	yticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
	ylabels = [f'{temp_min + (temp_max-temp_min)*y:.1f}' for y in yticks]
	ax.set_yticks(yticks)
	ax.set_yticklabels(ylabels, fontsize=12, color='#E6D6F7')
	# 去除x轴刻度
	ax.set_xticks([])
	plt.title('Temperature: Day vs Night (August 2025)', fontsize=22, color='white', pad=30)
	plt.tight_layout()
	# 导出图片
	plt.savefig('artistic_weather_aug2025.png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
	plt.show()

if __name__ == '__main__':
	url = 'https://mirror-earth.com/wea_history/440300/2025-08'
	html = fetch_html(url)
	df = parse_weather(html)
	if df.empty:
		print('未采集到数据！')
	else:
		plot_weather(df)
