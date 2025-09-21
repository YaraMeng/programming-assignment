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
	# 假设每一天是一个表格行，温度在某一列
	rows = soup.find_all('tr')
	data = []
	for row in rows:
		tds = row.find_all('td')
		if len(tds) < 5:
			continue
		# 假设第1列为日期，第2列为最高温，第3列为最低温
		date_text = tds[0].get_text(strip=True)
		max_temp = tds[1].get_text(strip=True)
		min_temp = tds[2].get_text(strip=True)
		# 只处理数字
		try:
			max_temp = float(re.findall(r'-?\d+\.?\d*', max_temp)[0])
			min_temp = float(re.findall(r'-?\d+\.?\d*', min_temp)[0])
			# 日期格式如 '2025-08-01 周五'，只取前10位
			date_part = date_text[:10]
			date_obj = datetime.strptime(date_part, '%Y-%m-%d')
			date_en = date_obj.strftime('%b %d, %Y')
			data.append({'date': date_en, 'max': max_temp, 'min': min_temp})
		except Exception:
			continue
	return pd.DataFrame(data)

def plot_weather(df):
	plt.figure(figsize=(12,6))
	plt.plot(df['date'], df['max'], label='Max Temp', marker='o', color='red')
	plt.plot(df['date'], df['min'], label='Min Temp', marker='o', color='blue')
	plt.xticks(rotation=45)
	plt.xlabel('Date')
	plt.ylabel('Temperature (°C)')
	plt.title('Daily Max/Min Temperature (August 2025)')
	plt.legend()
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	url = 'https://mirror-earth.com/wea_history/440300/2025-08'
	html = fetch_html(url)
	df = parse_weather(html)
	if df.empty:
		print('未采集到数据！')
	else:
		plot_weather(df)
