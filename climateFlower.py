import os
import calendar
import datetime
from typing import Dict, Any, Tuple, Optional

import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

#############################################
# Utilities
#############################################

def _to_np(arr) -> np.ndarray:
    """Convert a generic iterable to float numpy array, preserving NaN for invalid entries."""
    if arr is None:
        return np.array([], dtype=float)
    out = []
    for v in arr:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append(np.nan)
        elif isinstance(v, (int, float, np.integer, np.floating)):
            out.append(float(v))
        else:
            try:
                out.append(float(v))
            except Exception:
                out.append(np.nan)
    return np.array(out, dtype=float)


def fetch_daily_basic(year: int,
                      use_cache: bool = True,
                      cache_dir: str = '.cache',
                      csv: bool = False) -> Dict[str, Any]:
    """Fetch daily temperature/rain/wind for a year (365 days) with caching.

    Drops Feb 29 if leap year to ensure 365 petals.
    Falls back to synthetic seasonal data on failure.
    Returns dict: temp_mean, rain_sum, wind_max, dates(list[date]), meta.
    """
    start_date = datetime.date(year, 1, 1)
    end_date = datetime.date(year, 12, 31)
    cache_file = os.path.join(cache_dir, f"meteo_{year}.npz")
    csv_file = os.path.join(cache_dir, f"meteo_{year}.csv")

    if use_cache and os.path.exists(cache_file):
        try:
            data = np.load(cache_file, allow_pickle=True)
            t = data['temp_mean']
            r = data['rain_sum']
            w = data['wind_max']
            dates = [datetime.date.fromisoformat(d) for d in data['dates']]
            if len(dates) == 366:  # drop Feb 29
                idx = [i for i,d in enumerate(dates) if not (d.month == 2 and d.day == 29)]
                t, r, w = t[idx], r[idx], w[idx]
                dates = [dates[i] for i in idx]
            if len(dates) == 365:
                return {
                    'temp_mean': t,
                    'rain_sum': r,
                    'wind_max': w,
                    'dates': dates,
                    'meta': {'source': 'cache(npz)', 'message': 'loaded'}
                }
            else:
                print(f"[cache] invalid npz length={len(dates)} -> refetch")
        except Exception as e:
            print(f"[cache] npz load failed: {e}; refetch")

    # Try CSV (secondary cache)
    if use_cache and os.path.exists(csv_file):
        try:
            rows = []
            with open(csv_file, 'r', encoding='utf-8') as f:
                next(f)  # header
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 5:
                        rows.append(parts)
            if 360 <= len(rows) <= 366:
                dates = [datetime.date.fromisoformat(r[0]) for r in rows]
                t = np.array([float(r[1]) for r in rows])
                rs = np.array([float(r[2]) for r in rows])
                w = np.array([float(r[3]) for r in rows])
                if len(dates) == 366:
                    idx = [i for i,d in enumerate(dates) if not (d.month == 2 and d.day == 29)]
                    dates = [dates[i] for i in idx]
                    t, rs, w = t[idx], rs[idx], w[idx]
                if len(dates) == 365:
                    if use_cache:
                        try:
                            os.makedirs(cache_dir, exist_ok=True)
                            np.savez_compressed(cache_file,
                                                temp_mean=t,
                                                rain_sum=rs,
                                                wind_max=w,
                                                dates=np.array([d.isoformat() for d in dates], dtype=object))
                            print(f"[cache] rebuilt npz from csv -> {cache_file}")
                        except Exception as re:
                            print(f"[cache] rebuild npz failed: {re}")
                    return {
                        'temp_mean': t,
                        'rain_sum': rs,
                        'wind_max': w,
                        'dates': dates,
                        'meta': {'source': 'cache(csv)', 'message': 'csv loaded'}
                    }
                else:
                    print(f"[cache] csv invalid length={len(rows)} -> fetch network")
        except Exception as ce:
            print(f"[cache] csv load failed: {ce}; fetch network")

    # Network fetch
    url = (
        "https://meteo.agrodigits.com/v1/archive?format=json"
        "&longitude=113.92554&latitude=22.5364"
        "&hourly="
        "&daily=temperature_2m_mean,rain_sum,windspeed_10m_max"
        "&timezone=Asia/Shanghai"
        f"&start_date={start_date.isoformat()}&end_date={end_date.isoformat()}"
        "&windspeed_unit=ms&api_key="
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        js = resp.json()
        if 'daily' not in js:
            raise KeyError('daily missing')
        d = js['daily']
        t = _to_np(d.get('temperature_2m_mean', []))
        r = _to_np(d.get('rain_sum', []))
        w = _to_np(d.get('windspeed_10m_max', []))
        times = d.get('time', [])
        dates = [datetime.date.fromisoformat(x) for x in times]
        if len(dates) == 366:  # drop Feb 29
            idx = [i for i, dt in enumerate(dates) if not (dt.month == 2 and dt.day == 29)]
            t, r, w = t[idx], r[idx], w[idx]
            dates = [dates[i] for i in idx]
        mask = ~np.isnan(t) & ~np.isnan(r) & ~np.isnan(w)
        t, r, w = t[mask], r[mask], w[mask]
        dates = [dates[i] for i,m in enumerate(mask) if m]
        if len(dates) != 365:
            raise ValueError(f"expected 365 days after cleaning, got {len(dates)}")
        meta = {'source': 'api', 'message': 'ok'}
        if use_cache:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                np.savez_compressed(cache_file,
                                    temp_mean=t,
                                    rain_sum=r,
                                    wind_max=w,
                                    dates=np.array([d.isoformat() for d in dates], dtype=object))
                print(f"[cache] saved -> {cache_file}")
                if csv:
                    with open(csv_file, 'w', encoding='utf-8') as f:
                        f.write('date,temp_mean,rain_sum,wind_max,source\n')
                        for i,dte in enumerate(dates):
                            f.write(f"{dte.isoformat()},{t[i]:.4f},{r[i]:.4f},{w[i]:.4f},api\n")
            except Exception as se:
                print(f"[cache] save failed: {se}")
        return {'temp_mean': t, 'rain_sum': r, 'wind_max': w, 'dates': dates, 'meta': meta}
    except Exception as e:
        print(f"[fetch] network failed -> synthetic ({type(e).__name__}: {e})")
        # Synthetic fallback
        N = 365
        days = np.arange(N)
        t = 22 + 9 * np.sin(2 * np.pi * (days - 200) / N)
        r = np.where((days > 130) & (days < 260),
                     np.random.gamma(0.9, 2.2, N),
                     np.random.gamma(0.25, 1.2, N)) * 0.5
        w = np.clip(1.5 + 2.0 * np.sin(2 * np.pi * (days - 40) / N) + np.random.randn(N) * 0.8, 0, None)
        dates = [start_date + datetime.timedelta(days=int(i)) for i in range(N)]
        if use_cache:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                np.savez_compressed(cache_file,
                                    temp_mean=t,
                                    rain_sum=r,
                                    wind_max=w,
                                    dates=np.array([d.isoformat() for d in dates], dtype=object))
                if csv:
                    with open(csv_file, 'w', encoding='utf-8') as f:
                        f.write('date,temp_mean,rain_sum,wind_max,source\n')
                        for i,dte in enumerate(dates):
                            f.write(f"{dte.isoformat()},{t[i]:.4f},{r[i]:.4f},{w[i]:.4f},synthetic\n")
                print(f"[cache] synthetic saved -> {cache_file}")
            except Exception as se:
                print(f"[cache] synthetic save failed: {se}")
        return {'temp_mean': t, 'rain_sum': r, 'wind_max': w, 'dates': dates,
                'meta': {'source': 'synthetic', 'message': 'fallback'}}


#############################################
# Glassmorphism Climate Flower
#############################################

def _color_from_rain(rain: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r = rain.copy()
    r[r < 0] = 0
    p90 = np.nanpercentile(r, 90) if r.size else 1.0
    scale = p90 if p90 > 1e-6 else (r.max() + 1e-6)
    rn = np.clip(r / (scale + 1e-9), 0, 1)
    c_dry = np.array([1.0, 0.85, 0.40])
    c_wet = np.array([0.05, 0.22, 0.65])
    cols = (1 - rn)[:, None] * c_dry + rn[:, None] * c_wet
    cols = cols ** (1 + rn[:, None] * 0.6)
    alphas = 0.25 + 0.55 * rn
    return cols, alphas


def _wind_streak_intensity(wind: np.ndarray) -> np.ndarray:
    if wind.size == 0:
        return np.array([])
    w = wind.copy(); w[w < 0] = 0
    p80 = np.nanpercentile(w, 80) if w.size else 1.0
    scale = p80 if p80 > 1e-6 else (w.max() + 1e-6)
    wn = np.clip(w / (scale + 1e-9), 0, 1)
    return wn ** 1.3


def plot_glass_flower(year: int = 2024,
                      save_path: Optional[str] = None,
                      dpi: int = 250,
                      use_cache: bool = True,
                      save_svg: bool = False,
                      cache_dir: str = '.cache',
                      show: bool = True,
                      enable_hover: bool = True) -> None:
    data = fetch_daily_basic(year, use_cache=use_cache, cache_dir=cache_dir, csv=True)
    temp = data['temp_mean']; rain = data['rain_sum']; wind = data['wind_max']; dates = data['dates']
    N = len(temp)
    assert N == 365, f"Expected 365 days, got {N}"

    # Normalize temperature
    tmin, tmax = float(np.nanmin(temp)), float(np.nanmax(temp))
    tnorm = (temp - tmin) / (tmax - tmin + 1e-6)
    r_base, r_max = 0.05, 1.0
    colors, alphas = _color_from_rain(rain)
    wind_intensity = _wind_streak_intensity(wind)
    theta = np.linspace(0, 2*np.pi, N+1)

    fig, ax = plt.subplots(figsize=(11,11), subplot_kw={'polar': True})
    ax.set_facecolor((0.04, 0.05, 0.10)); fig.patch.set_facecolor((0.04,0.05,0.10))
    ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
    ax.set_yticklabels([]); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)

    # Glow
    for g in range(6):
        radius = r_max * (1.0 - g*0.14)
        alpha_glow = 0.08 * (1 - g/6)
        circ = plt.Circle((0,0), radius, transform=ax.transData, color=(0.15,0.20,0.45), alpha=alpha_glow, zorder=0)
        ax.add_artist(circ)

    dtheta = 2*np.pi / N
    heights = tnorm * (r_max - r_base)
    glow_heights = heights * 1.04
    glow_rgba = [(c[0],c[1],c[2], a*0.20) for c,a in zip(colors, alphas)]
    ax.bar(theta[:-1], glow_heights, width=dtheta, bottom=r_base, color=glow_rgba, linewidth=0, zorder=1, align='edge')
    main_rgba = [(c[0],c[1],c[2], a*0.85) for c,a in zip(colors, alphas)]
    bars = ax.bar(theta[:-1], heights, width=dtheta, bottom=r_base, color=main_rgba, linewidth=0, zorder=2, align='edge')
    for b,(col,a) in zip(bars, zip(colors, alphas)):
        b.set_path_effects([pe.withStroke(linewidth=0.6, foreground=(col[0],col[1],col[2], a*0.55))])

    # Wind streaks (subtle)
    if wind_intensity.size:
        thr = max(0.08, float(np.nanmedian(wind_intensity))*0.85)
        mask = wind_intensity > thr
        if np.any(mask):
            wi = wind_intensity[mask]
            streak_alpha = 0.06 + 0.22 * (wi ** 0.8)
            base_cols = colors[mask]
            mix = 0.08 + 0.92 * base_cols
            streak_rgba = [(c[0],c[1],c[2], a) for c,a in zip(mix, streak_alpha)]
            wind_bars = ax.bar(theta[:-1][mask], heights[mask]*0.97, width=dtheta*0.42, bottom=r_base,
                               color=streak_rgba, linewidth=0, zorder=3, align='edge')
            for wb in wind_bars:
                wb.set_path_effects([pe.withStroke(linewidth=0.25, foreground=(1,1,1,0.08))])

    # Month guides
    month_starts = [i for i,d in enumerate(dates) if d.day==1]
    if month_starts:
        guide_col = (1,1,1,0.12)
        for idx in month_starts:
            ang = theta[idx]
            ax.plot([ang, ang], [r_base*0.9, r_max], color=guide_col, linewidth=0.7, zorder=2.6, solid_capstyle='round')
    ax.set_xticks([theta[i] for i in month_starts])
    ax.set_xticklabels([calendar.month_abbr[dates[i].month].upper() for i in month_starts],
                       color=(0.9,0.92,0.96), fontsize=10)

    ax.set_title(f"Climate Flower of {year}", fontsize=18, pad=18, color=(0.92,0.95,1))
    source_txt = f"Data: open-meteo (cached)  •  Year: {year}"
    fig.text(0.995, 0.01, source_txt, ha='right', va='bottom', fontsize=8.5, color='#9aa4b5', alpha=0.9)

    # Legend & encoding info
    from matplotlib.lines import Line2D
    color_legend = [
        Line2D([0],[0], marker='s', markersize=11, linestyle='None', markerfacecolor=(1.0,0.85,0.40,0.70), markeredgecolor='none', label='Dry (low rain)'),
        Line2D([0],[0], marker='s', markersize=11, linestyle='None', markerfacecolor=(0.05,0.22,0.65,0.90), markeredgecolor='none', label='Wet (high rain)')
    ]
    leg = ax.legend(handles=color_legend, loc='upper left', bbox_to_anchor=(1.08,0.90), frameon=False, fontsize=10, labelcolor='#d0d5df', borderaxespad=0.)
    if leg:
        for txt in leg.get_texts():
            txt.set_color('#d0d5df')
    info_lines = ["Length: Temperature mean", "Color/Alpha: Rain", "Narrow Streak: Stronger Wind"]
    base_y = 0.80; line_h = 0.055
    for k,line in enumerate(info_lines):
        ax.text(1.08, base_y - k*line_h, line, transform=ax.transAxes, ha='left', va='top', fontsize=9.2, color='#cfd4dd')

    # Manual hover (transient only)
    if show and enable_hover:
        temp_arr = temp; rain_arr = rain; wind_arr = wind; date_list = dates
        original_facecolors = [b.get_facecolor() for b in bars]
        highlighted = {'i': None}
        annot = ax.annotate('', xy=(0,0), xytext=(10,10), textcoords='offset points',
                             bbox=dict(boxstyle='round,pad=0.45', fc='#f5f7fa', ec='#5b78c8', lw=1.0, alpha=0.96),
                             fontsize=9, color='#1c2333')
        annot.set_visible(False)

        def brighten(rgba):
            r,g,b,a = rgba
            return (min(r*1.22,1), min(g*1.22,1), min(b*1.22,1), min(a*1.18,1))

        def update_annot(i, event):
            d = date_list[i]
            text = (f"Date: {d.strftime('%Y-%m-%d')}\n"
                    f"Temp: {temp_arr[i]:.2f} °C\n"
                    f"Rain: {rain_arr[i]:.2f} mm\n"
                    f"Max Wind: {wind_arr[i]:.2f} m/s")
            annot.set_text(text)
            annot.xy = (event.xdata, event.ydata)
            annot.set_visible(True)

        def restore_prev():
            if highlighted['i'] is not None:
                j = highlighted['i']
                bars[j].set_facecolor(original_facecolors[j])
                bars[j].set_linewidth(0)
                highlighted['i'] = None

        def on_move(event):
            if event.inaxes != ax:
                restore_prev(); annot.set_visible(False); fig.canvas.draw_idle(); return
            found = None
            for idx,b in enumerate(bars):
                if b.contains(event)[0]:
                    found = idx; break
            if found is None:
                restore_prev(); annot.set_visible(False); fig.canvas.draw_idle(); return
            if highlighted['i'] != found:
                restore_prev()
                bars[found].set_facecolor(brighten(original_facecolors[found]))
                bars[found].set_linewidth(1.0)
                bars[found].set_edgecolor((1,1,1,0.85))
                highlighted['i'] = found
            update_annot(found, event)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('motion_notify_event', on_move)

    plt.tight_layout()
    if save_path:
        base, ext = os.path.splitext(save_path)
        out_png = save_path if ext.lower() in ('.png','.jpg','.jpeg') else base + '.png'
        plt.savefig(out_png, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches='tight')
        print(f"Saved raster: {out_png}")
        if save_svg:
            out_svg = base + '.svg'
            plt.savefig(out_svg, format='svg', facecolor=fig.get_facecolor(), bbox_inches='tight')
            print(f"Saved SVG: {out_svg}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_neon_placeholder():
    print("Neon style placeholder removed in simplification.")


if __name__ == '__main__':
    plot_glass_flower(2024, save_path='glass_flower_2024.png', use_cache=True, save_svg=True, show=True)
