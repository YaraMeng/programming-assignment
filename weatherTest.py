# 可视化香港机场历史天气页面24小时每小时的地面2米气温

# --- 新增 selenium 自动化采集 ---
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import re
import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

def parse_weather_table(html):
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table', id='archiveTable')
    if not table:
        print('No table found!')
        return []
    rows = table.find_all('tr')
    data = []
    current_date = None

    for row in rows:
        tds = row.find_all('td', recursive=False)
        if not tds:
            continue
        # 主行（日期行）
        if 'rowspan' in tds[0].attrs:
            current_date = tds[0].get_text(strip=True)
            continue
        # 小行（小时数据）
        if current_date and len(tds) > 2:
            hour = tds[0].get_text(strip=True)
            temp_td = tds[1]
            temp_div = temp_td.find('div', class_='t_0 dfs')
            if temp_div:
                try:
                    temp = float(temp_div.get_text(strip=True))
                    data.append({'date': current_date, 'hour': hour, 'temp': temp})
                except Exception:
                    pass
    return data

def daily_extremes(data):
    df = pd.DataFrame(data)
    grouped = df.groupby('date')['temp']
    daily_max = grouped.max()
    daily_min = grouped.min()
    return daily_max, daily_min


def plot_daily_extremes(daily_max, daily_min):
    import re
    import datetime
    def zh_date_to_en(date_str):
        # 例：2025年 9月 21日 星期日
        m = re.match(r'(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日', date_str)
        if m:
            y, mon, d = m.groups()
            dt = datetime.date(int(y), int(mon), int(d))
            return dt.strftime('%b %d, %Y')  # Sep 21, 2025
        return date_str

    en_dates = [zh_date_to_en(d) for d in daily_max.index]
    plt.figure(figsize=(12,6))
    plt.plot(en_dates, daily_max.values, label='Max Temp', marker='o', color='red')
    plt.plot(en_dates, daily_min.values, label='Min Temp', marker='o', color='blue')
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.title('Daily Max/Min Temperature')
    plt.legend()
    plt.tight_layout()
    plt.show()

# 主流程：自动抓取网页并可视化
if __name__ == "__main__":
    # Selenium 自动化采集 30天数据
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # 无界面模式
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(options=chrome_options)
    url = "https://rp5.ru/%E9%A6%99%E6%B8%AF(%E6%9C%BA%E5%9C%BA)%E5%8E%86%E5%8F%B2%E5%A4%A9%E6%B0%94_"
    driver.get(url)
    try:
        # 等待“30天”按钮出现并点击
        wait = WebDriverWait(driver, 20)
        btn_30 = wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@type='radio' and @value='30']")))
        btn_30.click()
        # 等待页面刷新，archiveTable 出现
        wait.until(EC.presence_of_element_located((By.ID, "archiveTable")))
        time.sleep(2)
        html = driver.page_source
        with open("rp5_30days_selenium.html", "w", encoding="utf-8") as f:
            f.write(html)
        print("已用selenium采集并保存30天源码到 rp5_30days_selenium.html")
        all_data = parse_weather_table(html)
        print(f"已采集30天数据，共 {len(all_data)} 条")
        if not all_data:
            print("未采集到任何温度数据！")
        else:
            daily_max, daily_min = daily_extremes(all_data)
            plot_daily_extremes(daily_max, daily_min)
    except Exception as e:
        print(f"selenium采集失败: {e}")
    finally:
        driver.quit()
# daily_max, daily_min = daily_extremes(data)
# plot_daily_extremes(daily_max, daily_min)
# plt.tight_layout()
# plt.show()


