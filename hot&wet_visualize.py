import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

def plot_weather_multiview():
    # 获取深圳过去24小时天气数据
    url = "https://meteo.agrodigits.com/v1/forecast?format=json&out_format=json&model=gfs&longitude=113.932482&latitude=22.521626&hourly=temperature_2m,windspeed_10m,dewpoint_2m,rain,cloudcover,apparent_temperature,winddirection_10m&daily=&past_days=1&forecast_days=1&current_weather=false&timezone=Asia/Shanghai&windspeed_unit=ms&api_key="
    resp = requests.get(url, timeout=15)
    data = resp.json()
    hourly = data['hourly']
    times = pd.to_datetime(hourly['time'])
    hours = times.hour[:24]
    temperature = np.array(hourly['temperature_2m'][:24])
    dewpoint = np.array(hourly['dewpoint_2m'][:24])
    apparent = np.array(hourly['apparent_temperature'][:24])
    rain = np.array(hourly['rain'][:24])
    cloudcover = np.array(hourly['cloudcover'][:24])
    windspeed = np.array(hourly['windspeed_10m'][:24])
    winddir = np.array(hourly.get('winddirection_10m', [0]*24)[:24])

    fig = plt.figure(figsize=(14,7))
    gs = gridspec.GridSpec(2, 2, width_ratios=[3,1], height_ratios=[3,1], wspace=0.18, hspace=0.22)

    # 主折线图（气温/露点/体感温度）
    ax_main = fig.add_subplot(gs[0,0])
    ax_main.plot(hours, temperature, label='Temperature (°C)', color='#d7263d', lw=2.5)
    ax_main.plot(hours, dewpoint, label='Dew Point (°C)', color='#3bbfae', lw=2.2, alpha=0.7)
    ax_main.plot(hours, apparent, label='Apparent Temp (°C)', color='#ffd95a', lw=2.2, alpha=0.7)
    ax_main.set_ylabel('Temperature (°C)', fontsize=13)
    # 只显示每3小时的时间刻度，标签自动旋转
    xtick_pos = [h for h in hours if h % 3 == 0]
    ax_main.set_xticks(xtick_pos)
    ax_main.set_xticklabels([f'{h}:00' for h in xtick_pos], rotation=30, fontsize=12)
    ax_main.grid(True, alpha=0.18)
    ax_main.legend(fontsize=11, loc='upper left')
    ax_main.set_title('Shenzhen Weather (Past 24h)', fontsize=18, color='#d7263d', pad=10)

    # 云量背景色带
    cloud_norm = (cloudcover - cloudcover.min()) / (cloudcover.max() - cloudcover.min() + 1e-6)
    for i in range(len(hours)-1):
        ax_main.axvspan(hours[i], hours[i+1], color=plt.cm.Blues(cloud_norm[i]), alpha=0.18)

    # 底部降雨柱状图
    ax_rain = fig.add_subplot(gs[1,0], sharex=ax_main)
    ax_rain.bar(hours, rain, color='#3b9cff', alpha=0.8, width=0.8)
    ax_rain.set_ylabel('Rain (mm)', fontsize=13)
    ax_rain.set_xticks(xtick_pos)
    ax_rain.set_xticklabels([f'{h}:00' for h in xtick_pos], rotation=30, fontsize=12)
    ax_rain.grid(True, alpha=0.13)

    # 右侧风玫瑰图
    ax_wind = fig.add_subplot(gs[:,1], polar=True)
    wind_dir_rad = np.deg2rad(winddir)
    bins = np.linspace(0,2*np.pi,9)
    hist, _ = np.histogram(wind_dir_rad, bins)
    mean_speed = [windspeed[(wind_dir_rad>=bins[i])&(wind_dir_rad<bins[i+1])].mean() if hist[i]>0 else 0 for i in range(len(bins)-1)]
    theta = (bins[:-1]+bins[1:])/2
    ax_wind.bar(theta, mean_speed, width=np.pi/4, color='#3bbfae', alpha=0.7, edgecolor='#fff', linewidth=2)
    ax_wind.set_theta_zero_location('N')
    ax_wind.set_theta_direction(-1)
    ax_wind.set_yticklabels([])
    ax_wind.set_xticks(np.deg2rad([0,45,90,135,180,225,270,315]))
    ax_wind.set_xticklabels(['N','NE','E','SE','S','SW','W','NW'], fontsize=12)
    ax_wind.set_title('Wind Rose', fontsize=15, color='#3bbfae', pad=12)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_weather_multiview()
