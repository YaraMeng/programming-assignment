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
	import matplotlib.animation as animation
	import numpy as np
	import matplotlib.patches as patches
	# 参数
	dates = df['date']
	min_temps = df['min']
	max_temps = df['max']
	n = len(df)
	# 柱体参数
	bar_width = 0.6
	base_y = 0.05
	max_bar_height = 0.8
	# 归一化高度
	min_h = (min_temps - min_temps.min()) / (max_temps.max() - min_temps.min() + 1e-6)
	max_h = (max_temps - min_temps.min()) / (max_temps.max() - min_temps.min() + 1e-6)
	# 画布（提前定义，动画和静态绘图共用）
	fig, ax = plt.subplots(figsize=(18,8))
	# 动画参数
	frames_per_bar = 25
	total_frames = n * frames_per_bar
	# 预计算每根柱子的冰晶/岩浆高度
	ice_targets = min_h * max_bar_height
	magma_targets = max_h * max_bar_height - ice_targets
	# 锯齿参数
	def draw_zigzag(ax, x, y, width, height, color, lw=2, zorder=5):
		num_zig = 6
		zig_x = np.linspace(x-width/2, x+width/2, num_zig*2)
		zig_y = np.array([y + (height/2 if i%2==0 else -height/2) for i in range(num_zig*2)])
		ax.plot(zig_x, zig_y, color=color, lw=lw, zorder=zorder)
	# 动画更新函数
	def update(frame):
		ax.clear()
		fig.patch.set_facecolor('#7B3FA0')
		ax.set_facecolor('#7B3FA0')
		ax.set_xlim(-1, n)
		ax.set_ylim(0, 1)
		ax.spines['left'].set_color('#6C3483')
		ax.spines['left'].set_linewidth(2)
		ax.spines['bottom'].set_color('none')
		ax.spines['top'].set_color('none')
		ax.spines['right'].set_color('none')
		ax.yaxis.set_ticks_position('left')
		ax.xaxis.set_ticks_position('none')
		temp_min = min_temps.min()
		temp_max = max_temps.max()
		yticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
		ylabels = [f'{temp_min + (temp_max-temp_min)*y:.1f}' for y in yticks]
		ax.set_yticks(yticks)
		ax.set_yticklabels(ylabels, fontsize=12, color='#E6D6F7')
		ax.set_xticks([])
		plt.title('Temperature: Day vs Night (August 2025)', fontsize=22, color='white', pad=30)
		# 当前激活柱子
		bar_idx = frame // frames_per_bar
		bar_progress = (frame % frames_per_bar) / frames_per_bar
		for i in range(n):
			x = i
			# 柱体动画进度
			if i < bar_idx:
				ice_h = ice_targets.iloc[i]
				magma_h = magma_targets.iloc[i]
			elif i == bar_idx:
				# 4阶段动画
				if bar_progress < 0.25:
					# 冰晶生长，岩浆退缩
					ice_h = ice_targets.iloc[i] * (bar_progress/0.25)
					magma_h = magma_targets.iloc[i] * (1-bar_progress/0.25)
				elif bar_progress < 0.5:
					# 岩浆喷发，冰晶被压
					ice_h = ice_targets.iloc[i] * (1-(bar_progress-0.25)/0.25)
					magma_h = magma_targets.iloc[i] * ((bar_progress-0.25)/0.25)
				elif bar_progress < 0.75:
					# 午后对抗，锯齿交界
					ice_h = ice_targets.iloc[i]*0.5
					magma_h = magma_targets.iloc[i]*0.5
				else:
					# 傍晚冰晶反攻
					ice_h = ice_targets.iloc[i] * ((bar_progress-0.75)/0.25)
					magma_h = magma_targets.iloc[i] * (1-(bar_progress-0.75)/0.25)
			else:
				ice_h = 0
				magma_h = 0
			# 冰晶层
			ice_rect = patches.Rectangle((x-bar_width/2, base_y), bar_width, ice_h, color='#00F0FF', alpha=0.7, zorder=2)
			ax.add_patch(ice_rect)
			# 岩浆层
			magma_rect = patches.Rectangle((x-bar_width/2, base_y+ice_h), bar_width, magma_h, color='#FF3B3B', alpha=0.7, zorder=3)
			ax.add_patch(magma_rect)
			# 锯齿交界线（仅午后对抗阶段）
			if i == bar_idx and 0.5 <= bar_progress < 0.75:
				draw_zigzag(ax, x, base_y+ice_h, bar_width, 0.04, color='white', lw=2, zorder=5)
			# 温差白线
			delta = max_temps.iloc[i] - min_temps.iloc[i]
			lw_ = 2 + 3*(delta/(max_temps.max()-min_temps.min()+1e-6))
			lightning_x = x
			lightning_y = base_y + ice_h + magma_h/2
			ax.plot([lightning_x, lightning_x, lightning_x],
					[lightning_y, lightning_y+0.04, lightning_y-0.04],
					color='white', lw=lw_, zorder=4)
			# 温差数值
			ax.text(lightning_x, lightning_y+0.05, f'{delta:.1f}', color='white', ha='center', va='bottom', fontsize=9, alpha=0.8)
			# 日期
			day_num = dates.iloc[i][4:6].replace(',','').replace(' ','')
			if not day_num.isdigit():
				day_num = dates.iloc[i][5:7].replace(',','').replace(' ','')
			ax.text(lightning_x, base_y+max_bar_height+0.03, day_num, color='white', ha='center', va='bottom', fontsize=11, alpha=0.9)
	# 创建动画
	ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=400, blit=False)
	ani.save('weather_bars_animation.mp4', writer='ffmpeg', dpi=180)
	plt.show()
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

def main():
	url = 'https://www.mirror-earth.com/wea_history/440300/2025-08'
	html = fetch_html(url)
	df = parse_weather(html)
	plot_weather(df)

if __name__ == '__main__':
	main()

	temp_max = max_temps.max()
	yticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
	ylabels = [f'{temp_min + (temp_max-temp_min)*y:.1f}' for y in yticks]

	ax.set_yticks(yticks)
	ax.set_yticklabels(ylabels, fontsize=12, color='#E6D6F7')
	ax.set_xticks([])

	plt.title('Temperature: Day vs Night (August 2025)', fontsize=22, color='white', pad=30)
	# 当前激活柱子
	bar_idx = frame // frames_per_bar

		bar_progress = (frame % frames_per_bar) / frames_per_bar
		for i in range(n):
			x = i

			# 柱体动画进度
			if i < bar_idx:
				ice_h = ice_targets.iloc[i]

				# 4阶段动画
				if bar_progress < 0.25:
					# 冰晶生长，岩浆退缩

					# 岩浆喷发，冰晶被压
					ice_h = ice_targets.iloc[i] * (1-(bar_progress-0.25)/0.25)
					magma_h = magma_targets.iloc[i] * ((bar_progress-0.25)/0.25)

				elif bar_progress < 0.75:
					# 午后对抗，锯齿交界
					ice_h = ice_targets.iloc[i]*0.5

				else:
					# 傍晚冰晶反攻
					ice_h = ice_targets.iloc[i] * ((bar_progress-0.75)/0.25)

				ice_h = 0
				magma_h = 0
			# 冰晶层

			ice_rect = patches.Rectangle((x-bar_width/2, base_y), bar_width, ice_h, color='#00F0FF', alpha=0.7, zorder=2)
			ax.add_patch(ice_rect)
			# 岩浆层

			magma_rect = patches.Rectangle((x-bar_width/2, base_y+ice_h), bar_width, magma_h, color='#FF3B3B', alpha=0.7, zorder=3)
			ax.add_patch(magma_rect)
			# 锯齿交界线（仅午后对抗阶段）

			if i == bar_idx and 0.5 <= bar_progress < 0.75:
				draw_zigzag(ax, x, base_y+ice_h, bar_width, 0.04, color='white', lw=2, zorder=5)
			# 温差白线

			delta = max_temps.iloc[i] - min_temps.iloc[i]
			lw_ = 2 + 3*(delta/(max_temps.max()-min_temps.min()+1e-6))
			lightning_x = x

	    lightning_y = base_y + ice_h + magma_h/2
        ax.plot([lightning_x, lightning_x, lightning_x],
		    [lightning_y, lightning_y+0.04, lightning_y-0.04],

		    color='white', lw=lw_, zorder=4)
	    # 温差数值
	    ax.text(lightning_x, lightning_y+0.05, f'{delta:.1f}', color='white', ha='center', va='bottom', fontsize=9, alpha=0.8)

			# 日期
			day_num = dates.iloc[i][4:6].replace(',','').replace(' ','')
			if not day_num.isdigit():

				day_num = dates.iloc[i][5:7].replace(',','').replace(' ','')
			ax.text(lightning_x, base_y+max_bar_height+0.03, day_num, color='white', ha='center', va='bottom', fontsize=11, alpha=0.9)
	# 创建动画

	ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=400, blit=False)
	ani.save('weather_bars_animation.mp4', writer='ffmpeg', dpi=180)
	plt.show()

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

def main():
	url = 'https://www.mirror-earth.com/wea_history/440300/2025-08'
	html = fetch_html(url)

	df = parse_weather(html)
	plot_weather(df)

if __name__ == '__main__':
	main()
def main():
	url = 'https://www.mirror-earth.com/wea_history/440300/2025-08'
	html = fetch_html(url)
	df = parse_weather(html)
	plot_weather(df)

if __name__ == '__main__':
	main()
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
	import matplotlib.animation as animation
	import numpy as np
	import matplotlib.patches as patches
	# 参数
	dates = df['date']
	min_temps = df['min']
	max_temps = df['max']
	n = len(df)
	# 柱体参数
	bar_width = 0.6
	base_y = 0.05
	max_bar_height = 0.8
	# 归一化高度
	min_h = (min_temps - min_temps.min()) / (max_temps.max() - min_temps.min() + 1e-6)
	max_h = (max_temps - min_temps.min()) / (max_temps.max() - min_temps.min() + 1e-6)
	# 画布（提前定义，动画和静态绘图共用）
	fig, ax = plt.subplots(figsize=(18,8))
	# 动画参数
	frames_per_bar = 25
	total_frames = n * frames_per_bar
	# 预计算每根柱子的冰晶/岩浆高度
	ice_targets = min_h * max_bar_height
	magma_targets = max_h * max_bar_height - ice_targets
	# 锯齿参数
	def draw_zigzag(ax, x, y, width, height, color, lw=2, zorder=5):
		# 生成锯齿线
		num_zig = 6
		zig_x = np.linspace(x-width/2, x+width/2, num_zig*2)
		zig_y = np.array([y + (height/2 if i%2==0 else -height/2) for i in range(num_zig*2)])
		ax.plot(zig_x, zig_y, color=color, lw=lw, zorder=zorder)
	# 动画更新函数
	def update(frame):
		ax.clear()
		fig.patch.set_facecolor('#7B3FA0')
		ax.set_facecolor('#7B3FA0')
		ax.set_xlim(-1, n)
		ax.set_ylim(0, 1)
		ax.spines['left'].set_color('#6C3483')
		ax.spines['left'].set_linewidth(2)
		ax.spines['bottom'].set_color('none')
		ax.spines['top'].set_color('none')
		ax.spines['right'].set_color('none')
		ax.yaxis.set_ticks_position('left')
		ax.xaxis.set_ticks_position('none')
		temp_min = min_temps.min()
		temp_max = max_temps.max()
		yticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
		ylabels = [f'{temp_min + (temp_max-temp_min)*y:.1f}' for y in yticks]
		ax.set_yticks(yticks)
		ax.set_yticklabels(ylabels, fontsize=12, color='#E6D6F7')
		ax.set_xticks([])
		plt.title('Temperature: Day vs Night (August 2025)', fontsize=22, color='white', pad=30)
		# 当前激活柱子
		bar_idx = frame // frames_per_bar
		bar_progress = (frame % frames_per_bar) / frames_per_bar
		for i in range(n):
			x = i
			# 柱体动画进度
			if i < bar_idx:
				ice_h = ice_targets.iloc[i]
				magma_h = magma_targets.iloc[i]
			elif i == bar_idx:
				# 4阶段动画
				if bar_progress < 0.25:
					# 冰晶生长，岩浆退缩
					ice_h = ice_targets.iloc[i] * (bar_progress/0.25)
					magma_h = magma_targets.iloc[i] * (1-bar_progress/0.25)
				elif bar_progress < 0.5:
					# 岩浆喷发，冰晶被压
					ice_h = ice_targets.iloc[i] * (1-(bar_progress-0.25)/0.25)
					magma_h = magma_targets.iloc[i] * ((bar_progress-0.25)/0.25)
				elif bar_progress < 0.75:
					# 午后对抗，锯齿交界
					ice_h = ice_targets.iloc[i]*0.5
					magma_h = magma_targets.iloc[i]*0.5
				else:
					# 傍晚冰晶反攻
					ice_h = ice_targets.iloc[i] * ((bar_progress-0.75)/0.25)
					magma_h = magma_targets.iloc[i] * (1-(bar_progress-0.75)/0.25)
			else:
				ice_h = 0
				magma_h = 0
			# 冰晶层
			ice_rect = patches.Rectangle((x-bar_width/2, base_y), bar_width, ice_h, color='#00F0FF', alpha=0.7, zorder=2)
			ax.add_patch(ice_rect)
			# 岩浆层
			magma_rect = patches.Rectangle((x-bar_width/2, base_y+ice_h), bar_width, magma_h, color='#FF3B3B', alpha=0.7, zorder=3)
			ax.add_patch(magma_rect)
			# 锯齿交界线（仅午后对抗阶段）
			if i == bar_idx and 0.5 <= bar_progress < 0.75:
				draw_zigzag(ax, x, base_y+ice_h, bar_width, 0.04, color='white', lw=2, zorder=5)
			# 温差白线
			delta = max_temps.iloc[i] - min_temps.iloc[i]
			lw_ = 2 + 3*(delta/(max_temps.max()-min_temps.min()+1e-6))
			lightning_x = x
			lightning_y = base_y + ice_h + magma_h/2
			ax.plot([lightning_x, lightning_x, lightning_x],
					[lightning_y, lightning_y+0.04, lightning_y-0.04],
					color='white', lw=lw_, zorder=4)
			# 温差数值
			ax.text(lightning_x, lightning_y+0.05, f'{delta:.1f}', color='white', ha='center', va='bottom', fontsize=9, alpha=0.8)
			# 日期
			day_num = dates.iloc[i][4:6].replace(',','').replace(' ','')
			if not day_num.isdigit():
				day_num = dates.iloc[i][5:7].replace(',','').replace(' ','')
			ax.text(lightning_x, base_y+max_bar_height+0.03, day_num, color='white', ha='center', va='bottom', fontsize=11, alpha=0.9)
	# 创建动画
	ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=400, blit=False)
	ani.save('weather_bars_animation.mp4', writer='ffmpeg', dpi=180)
	plt.show()
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
