import os
import datetime
from typing import Dict, Any, Tuple, Optional

import numpy as np
import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Polygon  # kept for potential future neon style reuse
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Wedge, Polygon  # kept for potential future neon style reuse
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
import matplotlib.patheffects as pe

#############################################
# Utilities
#############################################

def _magnus_es(t_c: np.ndarray) -> np.ndarray:
    return 6.1094 * np.exp((17.625 * t_c) / (243.04 + t_c))


def compute_rh_from_t_td(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    es = _magnus_es(t_c)
    e = _magnus_es(td_c)
    return np.clip(100.0 * (e / (es + 1e-9)), 0.0, 100.0)


def _to_np(arr) -> np.ndarray:
    if arr is None:
        return np.array([], dtype=float)
    out = []
    for v in arr:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append(np.nan)
        elif isinstance(v, (int, float, np.integer, np.floating)):
            out.append(float(v))
        elif isinstance(v, str):
            s = v.strip().lower()
            if s in ("", "nan", "null", "none"):
                out.append(np.nan)
            else:
                try:
                    out.append(float(v))
                except ValueError:
                    out.append(np.nan)
        else:
            try:
                out.append(float(v))
            except Exception:
                out.append(np.nan)
    return np.array(out, dtype=float)


#############################################
# Data Fetchers
#############################################

def fetch_daily_basic(year: int = 2024, use_cache: bool = True, cache_dir: str = '.cache', csv: bool = True) -> Dict[str, Any]:
    """
    Fetch daily temperature_2m_mean, rain_sum, windspeed_10m_max for the whole year.
    Provided URL (no humidity here):
    https://meteo.agrodigits.com/v1/archive?format=json&longitude=113.92554&latitude=22.5364&hourly=&daily=temperature_2m_mean,rain_sum,windspeed_10m_max&timezone=Asia/Shanghai&start_date=2024-01-01&end_date=2024-12-31&windspeed_unit=ms&api_key=
    Returns dict with arrays (365 days). If leap year and 366 days returned, drop Feb-29 to keep 365 petals.
    """
    start_date = datetime.date(year, 1, 1)
    end_date = datetime.date(year, 12, 31)
    cache_file = os.path.join(cache_dir, f"daily_basic_{year}.npz")
    csv_file = os.path.join(cache_dir, f"daily_basic_{year}.csv")
    if use_cache:
        npz_ok = False
        if os.path.exists(cache_file):
            try:
                data = np.load(cache_file, allow_pickle=True)
                dates = [datetime.date.fromisoformat(s) for s in data['dates']]
                if len(dates) == 365:  # validity check
                    npz_ok = True
                    if csv and not os.path.exists(csv_file):
                        try:
                            os.makedirs(cache_dir, exist_ok=True)
                            with open(csv_file, 'w', encoding='utf-8') as f:
                                f.write('date,temp_mean,rain_sum,wind_max,source\n')
                                for i, dte in enumerate(dates):
                                    f.write(f"{dte.isoformat()},{data['temp_mean'][i]:.4f},{data['rain_sum'][i]:.4f},{data['wind_max'][i]:.4f},cache\n")
                            print(f"[cache] reconstructed csv -> {csv_file}")
                        except Exception as ce:
                            print(f"[cache] csv reconstruct failed: {ce}")
                    return {
                        'temp_mean': data['temp_mean'],
                        'rain_sum': data['rain_sum'],
                        'wind_max': data['wind_max'],
                        'dates': dates,
                        'meta': {'source': 'cache', 'message': f'cache hit {len(dates)} days'}
                    }
                else:
                    print(f"[cache] npz invalid length={len(dates)} -> ignore")
            except Exception as e:
                print(f"[cache] load failed: {e}; will attempt csv...")
        # Try CSV fallback to rebuild npz if npz invalid/missing
        if not npz_ok and csv and os.path.exists(csv_file):
            try:
                rows = []
                with open(csv_file, 'r', encoding='utf-8') as f:
                    header = f.readline()
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 5:
                            dte = datetime.date.fromisoformat(parts[0])
                            rows.append((dte, float(parts[1]), float(parts[2]), float(parts[3])))
                rows.sort(key=lambda x: x[0])
                if len(rows) == 365:
                    dates = [r[0] for r in rows]
                    t = np.array([r[1] for r in rows])
                    rsum = np.array([r[2] for r in rows])
                    w = np.array([r[3] for r in rows])
                    # rebuild npz for faster future load
                    try:
                        np.savez_compressed(cache_file,
                                            temp_mean=t,
                                            rain_sum=rsum,
                                            wind_max=w,
                                            dates=np.array([d.isoformat() for d in dates], dtype=object))
                        print(f"[cache] rebuilt npz from csv -> {cache_file}")
                    except Exception as re:
                        print(f"[cache] rebuild npz failed: {re}")
                    return {
                        'temp_mean': t,
                        'rain_sum': rsum,
                        'wind_max': w,
                        'dates': dates,
                        'meta': {'source': 'cache(csv)', 'message': 'csv rebuild'}
                    }
                else:
                    print(f"[cache] csv invalid length={len(rows)} -> will fetch network")
            except Exception as ce:
                print(f"[cache] csv load failed: {ce}; will fetch network")

    url = (
        "https://meteo.agrodigits.com/v1/archive?format=json"
        "&longitude=113.92554&latitude=22.5364"
        "&hourly="
        "&daily=temperature_2m_mean,rain_sum,windspeed_10m_max"
        "&timezone=Asia/Shanghai"
        f"&start_date={start_date.isoformat()}&end_date={end_date.isoformat()}"
        "&windspeed_unit=ms&api_key="
    )
    resp = None
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
        # Handle leap year extra day (Feb 29) -> drop to keep 365 petals
        if len(dates) == 366:
            filtered_idx = [i for i, dt in enumerate(dates) if not (dt.month == 2 and dt.day == 29)]
            t, r, w = t[filtered_idx], r[filtered_idx], w[filtered_idx]
            dates = [dates[i] for i in filtered_idx]
        # Mask missing days
        mask = ~np.isnan(t) & ~np.isnan(r) & ~np.isnan(w)
        t, r, w = t[mask], r[mask], w[mask]
        dates = [dates[i] for i, m in enumerate(mask) if m]
        meta = {'source': 'api', 'message': f'ok {len(dates)} days'}
        out = {'temp_mean': t, 'rain_sum': r, 'wind_max': w, 'dates': dates, 'meta': meta}
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
                    try:
                        with open(csv_file, 'w', encoding='utf-8') as f:
                            f.write('date,temp_mean,rain_sum,wind_max,source\n')
                            for i, dte in enumerate(dates):
                                f.write(f"{dte.isoformat()},{t[i]:.4f},{r[i]:.4f},{w[i]:.4f},api\n")
                        print(f"[cache] csv saved -> {csv_file}")
                    except Exception as ce:
                        print(f"[cache] csv save failed: {ce}")
            except Exception as se:
                print(f"[cache] save failed: {se}")
        return out
    except Exception as e:
        print(f"[fetch_daily_basic] fallback due to {type(e).__name__}: {e}")
        # Synthetic seasonal curve
        N = 365
        days = np.arange(N)
        t = 22 + 9 * np.sin(2 * np.pi * (days - 200) / N)
        # Rain: wetter summer
        r = np.where((days > 130) & (days < 260),
                     np.random.gamma(0.9, 2.2, N),
                     np.random.gamma(0.25, 1.2, N))
        r *= 0.5
        w = np.clip(1.5 + 2.0 * np.sin(2 * np.pi * (days - 40) / N) + np.random.randn(N) * 0.8, 0, None)
        dates = [start_date + datetime.timedelta(days=int(i)) for i in range(N)]
        out = {
            'temp_mean': t,
            'rain_sum': r,
            'wind_max': w,
            'dates': dates,
            'meta': {'source': 'synthetic', 'message': 'synthetic seasonal'}
        }
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
                        for i, dte in enumerate(dates):
                            f.write(f"{dte.isoformat()},{t[i]:.4f},{r[i]:.4f},{w[i]:.4f},synthetic\n")
                print(f"[cache] synthetic saved -> {cache_file} (+csv)")
            except Exception as se:
                print(f"[cache] synthetic save failed: {se}")
        return out


#############################################
# Glassmorphism Climate Flower
#############################################

def _color_from_rain(rain: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map rain to color + alpha.
    Dry -> warm translucent amber (yellow/orange), Wet -> deep saturated blue.
    Return (colors Nx3, alphas N).
    """
    r = rain.copy()
    r[r < 0] = 0
    # Normalize rain (robust: p90) so a few huge rain days don't dominate
    p90 = np.nanpercentile(r, 90) if r.size else 1.0
    scale = p90 if p90 > 1e-6 else (r.max() + 1e-6)
    rn = np.clip(r / (scale + 1e-9), 0, 1)
    # Color blend: amber -> deep blue
    # amber ~ (1.0, 0.85, 0.4); deep blue ~ (0.05, 0.22, 0.65)
    c_dry = np.array([1.0, 0.85, 0.40])
    c_wet = np.array([0.05, 0.22, 0.65])
    cols = (1 - rn)[:, None] * c_dry + rn[:, None] * c_wet
    # Slight nonlinear darkening for wet
    cols = cols ** (1 + rn[:, None] * 0.6)
    # Alpha: base 0.25 + 0.55*rain_factor (glass layering)
    alphas = 0.25 + 0.55 * rn
    return cols, alphas


def _wind_streak_intensity(wind: np.ndarray) -> np.ndarray:
    if wind.size == 0:
        return np.array([])
    w = wind.copy()
    w[w < 0] = 0
    p80 = np.nanpercentile(w, 80) if w.size else 1.0
    scale = p80 if p80 > 1e-6 else (w.max() + 1e-6)
    wn = np.clip(w / (scale + 1e-9), 0, 1)
    # Smooth easing
    return wn ** 1.3


def plot_glass_flower(year: int = 2024,
                      save_path: Optional[str] = None,
                      dpi: int = 250,
                      use_cache: bool = True,
                      save_svg: bool = False,
                      cache_dir: str = '.cache',
                      show: bool = True,
                      enable_hover: bool = True) -> None:
    """
    Glassmorphism style 'Climate Flower'.
    - 365 petals (remove Feb 29 if leap year)
    - Length: temperature_2m_mean (normalized)
    - Color & alpha: rain_sum (amber -> deep blue, alpha increases with rain)
    - Wind texture: subtle streak/blur intensity by windspeed_10m_max
    - Jan 1 at top, clockwise
    - Adds faint radial lines at each month's first day

    Parameters
    ----------
    year : int
        Year of data.
    save_path : str | None
        If provided, save raster PNG (and optionally SVG if save_svg=True).
    dpi : int
        DPI for raster export.
    use_cache : bool
        Use on-disk cache (.npz/.csv) before network.
    save_svg : bool
        Also export SVG (vector).
    cache_dir : str
        Cache directory.
    show : bool
        If True, show the figure (GUI). If False, close it after creation.
    enable_hover : bool
        Enable interactive hover + pin (needs mplcursors & GUI backend).
    """
    data = fetch_daily_basic(year, use_cache=use_cache, cache_dir=cache_dir, csv=True)
    temp = data['temp_mean']
    rain = data['rain_sum']
    wind = data['wind_max']
    dates = data['dates']
    N = len(temp)
    assert N == 365, f"Expected 365 days, got {N}"

    # Normalize temperature for radius
    tmin, tmax = np.nanmin(temp), np.nanmax(temp)
    tnorm = (temp - tmin) / (tmax - tmin + 1e-6)
    r_base, r_max = 0.05, 1.0

    colors, alphas = _color_from_rain(rain)
    wind_intensity = _wind_streak_intensity(wind)

    theta = np.linspace(0, 2 * np.pi, N + 1)

    fig, ax = plt.subplots(figsize=(11, 11), subplot_kw={'polar': True})

    # Background: deep blue-violet radial gradient using imshow in polar coords
    ax.set_facecolor((0.04, 0.05, 0.10))
    fig.patch.set_facecolor((0.04, 0.05, 0.10))
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)  # clockwise
    ax.set_yticklabels([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Create radial glow (a few translucent concentric wedges / circles)
    glow_levels = 6
    for g in range(glow_levels):
        radius = r_max * (1.0 - g * 0.14)
        alpha_glow = 0.08 * (1 - g / glow_levels)
        circ = plt.Circle((0, 0), radius, transform=ax.transData, color=(0.15, 0.20, 0.45), alpha=alpha_glow, zorder=0)
        ax.add_artist(circ)

    # Reverted: use bar-based petals (rectangular annular sectors)
    dtheta = 2 * np.pi / N
    heights = tnorm * (r_max - r_base)
    glow_heights = heights * 1.04
    glow_rgba = [(c[0], c[1], c[2], a*0.20) for c, a in zip(colors, alphas)]
    ax.bar(theta[:-1], glow_heights, width=dtheta, bottom=r_base,
           color=glow_rgba, linewidth=0, zorder=1, align='edge')
    main_rgba = [(c[0], c[1], c[2], a*0.85) for c, a in zip(colors, alphas)]
    bars = ax.bar(theta[:-1], heights, width=dtheta, bottom=r_base,
                  color=main_rgba, linewidth=0, zorder=2, align='edge')
    for b, col, alpha in zip(bars, colors, alphas):
        b.set_path_effects([
            pe.withStroke(linewidth=0.6, foreground=(col[0], col[1], col[2], alpha * 0.55))
        ])
    # Wind streak overlay (进一步减弱视觉强度 -> “再淡一点”)
    if wind_intensity.size:
        # 自适应阈值：取中位数与 0.08 较大者，避免全部显示或全部消失
        dynamic_thr = max(0.08, float(np.nanmedian(wind_intensity)) * 0.85)
        mask = wind_intensity > dynamic_thr
        if np.any(mask):
            wi = wind_intensity[mask]
            # 再次减弱：降低 alpha 量级 (≈ 0.06 ~ 0.28)
            streak_alpha = 0.06 + 0.22 * (wi ** 0.8)
            base_cols = colors[mask]
            # 极轻微提亮：只混入 8% 白，并且不再额外偏蓝
            mix = 0.08 + 0.92 * base_cols  # white mix 8%
            streak_rgba = [(c[0], c[1], c[2], a) for c, a in zip(mix, streak_alpha)]
            # 更窄 & 贴合原高度（略短 0.97 保留细边缘）
            wind_bars = ax.bar(theta[:-1][mask], heights[mask] * 0.97, width=dtheta * 0.42, bottom=r_base,
                               color=streak_rgba, linewidth=0, zorder=3, align='edge')
            # 取消明显描边，仅留极淡描边（或直接注释掉下面两行可彻底无描边）
            for wb in wind_bars:
                wb.set_path_effects([
                    pe.withStroke(linewidth=0.25, foreground=(1, 1, 1, 0.08))
                ])

    # Day ticks (optional monthly markers)
    month_starts = []
    for i, d in enumerate(dates):
        if d.day == 1:
            month_starts.append(i)
    # Faint radial guide lines for each month's first day (very subtle)
    # User request: "每个月的1号加上一条很淡的线"
    if month_starts:
        guide_col = (1, 1, 1, 0.12)  # nearly white, very low alpha
        for idx in month_starts:
            ang = theta[idx]
            # Draw over bars slightly so it's visible but subtle
            ax.plot([ang, ang], [r_base * 0.9, r_max], color=guide_col,
                    linewidth=0.7, zorder=2.6, solid_capstyle='round')
    ax.set_xticks([theta[i] for i in month_starts])
    ax.set_xticklabels([dates[i].strftime('%m') for i in month_starts], color=(0.9, 0.92, 0.96), fontsize=10)

    title = f"Glass Climate Flower {year}\nLength=Temp  Color/Alpha=Rain  Streak=Wind"
    ax.set_title(title, fontsize=18, pad=18, color=(0.92, 0.95, 1))
    plt.tight_layout()
    if save_path:
        base, ext = os.path.splitext(save_path)
        out_png = save_path if ext.lower() in ('.png', '.jpg', '.jpeg') else base + '.png'
        plt.savefig(out_png, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches='tight')
        print(f"Saved raster: {out_png}")
        if save_svg:
            out_svg = base + '.svg'
            plt.savefig(out_svg, format='svg', facecolor=fig.get_facecolor(), bbox_inches='tight')
            print(f"Saved SVG: {out_svg}")
    # (HTML export removed in revert)
    # Hover (interactive) requires GUI + mplcursors
    if show and enable_hover:
        try:
            import mplcursors  # type: ignore
            temp_arr = temp; rain_arr = rain; wind_arr = wind; date_list = dates
            original_facecolors = [b.get_facecolor() for b in bars]

            # Hover cursor
            hover_cursor = mplcursors.cursor(bars, hover=True)

            # Pin cursor (click to freeze) - hover=False enables click selection
            pin_cursor = mplcursors.cursor(bars, hover=False)

            HIGHLIGHT_EDGE = (1, 1, 1, 0.9)

            # Three-column overlay: right-aligned label, colon, left-aligned value.
            def _annotation_entries(i: int):
                d = date_list[i]
                return [
                    ("Date", d.strftime('%Y-%m-%d')),
                    ("Temp", f"{temp_arr[i]:.2f} °C"),
                    ("Rain", f"{rain_arr[i]:.2f} mm"),
                    ("Max Wind", f"{wind_arr[i]:.2f} m/s"),
                ]

            def style_annotation(ann, pinned=False):
                box = ann.get_bbox_patch()
                # Light theme bubble
                box.set(facecolor='#f5f7fa', edgecolor='#5b78c8', alpha=0.95 if not pinned else 0.98, linewidth=1.1)
                # Set text color & enforce DejaVu Sans Mono via explicit FontProperties to guarantee fixed width
                if ann.get_children():
                    try:
                        txt = ann.get_children()[0]
                        txt.set_color('#1c2333')
                        mono_path = font_manager.findfont('DejaVu Sans Mono', fallback_to_default=True)
                        try:
                            txt.set_fontproperties(FontProperties(fname=mono_path))
                        except Exception:
                            # Fallback generic monospace
                            txt.set_fontfamily('monospace')
                    except Exception:
                        pass
                ann.set_fontsize(9)

            def brighten_color(rgba):
                r,g,b,a = rgba
                factor = 1.25
                return (min(r*factor,1.0), min(g*factor,1.0), min(b*factor,1.0), min(a*1.2,1.0))

            highlighted_index = {'i': None}

            @hover_cursor.connect('add')
            def _on_hover_add(sel):
                i = sel.index
                if i is None or i < 0 or i >= len(temp_arr):
                    return
                # Restore previous highlight
                if highlighted_index['i'] is not None and highlighted_index['i'] != i:
                    j = highlighted_index['i']
                    bars[j].set_facecolor(original_facecolors[j])
                    bars[j].set_linewidth(0)
                # Apply new highlight
                bars[i].set_facecolor(brighten_color(original_facecolors[i]))
                bars[i].set_linewidth(1.0)
                bars[i].set_edgecolor(HIGHLIGHT_EDGE)
                highlighted_index['i'] = i
                d = date_list[i]
                entries = _annotation_entries(i)
                placeholder = "\n".join([f"{lab}: {val}" for lab,val in entries])
                sel.annotation.set(text=placeholder)
                style_annotation(sel.annotation, pinned=False)
                # build overlay artists
                try:
                    # remove old if any
                    if hasattr(sel.annotation, '_overlay_artists'):
                        for a in (sel.annotation._overlay_artists.get('labels', []) +
                                  sel.annotation._overlay_artists.get('colons', []) +
                                  sel.annotation._overlay_artists.get('values', [])):
                            try: a.remove()
                            except Exception: pass
                    sel.annotation._overlay_entries = entries
                    sel.annotation._overlay_artists = {'labels': [], 'colons': [], 'values': []}
                except Exception:
                    pass

            @hover_cursor.connect('remove')
            def _on_hover_remove(sel):
                # Restore highlight when cursor leaves (unless pinned exists with same bar which is okay)
                if highlighted_index['i'] is not None:
                    idx = highlighted_index['i']
                    bars[idx].set_facecolor(original_facecolors[idx])
                    bars[idx].set_linewidth(0)
                    highlighted_index['i'] = None

            # Pin logic: clicking on a bar creates persistent annotation
            # Manage pinned annotations with mapping bar index -> {'sel': selection, 'close': text artist}
            pinned_annotations = {}
            @pin_cursor.connect('add')
            def _on_pin(sel):
                i = sel.index
                if i is None or i < 0 or i >= len(temp_arr):
                    return
                # Toggle: if already pinned -> remove
                if i in pinned_annotations:
                    ann_info = pinned_annotations.pop(i)
                    try:
                        ann_info['sel'].annotation.remove()
                        if ann_info.get('close') is not None:
                            ann_info['close'].remove()
                    except Exception:
                        pass
                    fig.canvas.draw_idle()
                    return
                # Create new pinned annotation
                d = date_list[i]
                entries = _annotation_entries(i)
                placeholder = "\n".join([f"{lab}: {val}" for lab,val in entries])
                sel.annotation.set(text=placeholder)
                style_annotation(sel.annotation, pinned=True)
                try:
                    if hasattr(sel.annotation, '_overlay_artists'):
                        for a in (sel.annotation._overlay_artists.get('labels', []) +
                                  sel.annotation._overlay_artists.get('colons', []) +
                                  sel.annotation._overlay_artists.get('values', [])):
                            try: a.remove()
                            except Exception: pass
                    sel.annotation._overlay_entries = entries
                    sel.annotation._overlay_artists = {'labels': [], 'colons': [], 'values': []}
                except Exception:
                    pass
                # Add close '×' placeholder (position later adjusted on draw)
                close_text = ax.text(0, 0, '×', fontsize=10, color='#2d3b55', ha='center', va='center', zorder=5,
                                     fontweight='bold', picker=True)
                pinned_annotations[i] = {'sel': sel, 'close': close_text}
                fig.canvas.draw_idle()

            def _update_close_positions(event=None):
                # Position close buttons and overlay
                renderer = fig.canvas.get_renderer()
                def ensure_overlay(ann):
                    if not hasattr(ann, '_overlay_entries'): return
                    if not ann._overlay_artists['labels']:
                        # create artists
                        for lab,val in ann._overlay_entries:
                            lab_t = ax.text(0,0,lab, fontsize=9, ha='right', va='top', color='#1c2333', zorder=6)
                            colon_t = ax.text(0,0,":", fontsize=9, ha='center', va='top', color='#1c2333', zorder=6)
                            val_t = ax.text(0,0,val, fontsize=9, ha='left', va='top', color='#1c2333', zorder=6)
                            ann._overlay_artists['labels'].append(lab_t)
                            ann._overlay_artists['colons'].append(colon_t)
                            ann._overlay_artists['values'].append(val_t)
                    # compute widths
                    max_label_w = 0
                    for lab_artist in ann._overlay_artists['labels']:
                        w,h,d = renderer.get_text_width_height_descent(lab_artist.get_text(), lab_artist._fontproperties, ismath=False)
                        max_label_w = max(max_label_w, w)
                    # layout parameters
                    left_pad = 8; top_pad = 6; colon_gap = 4; value_gap = 6
                    box = ann.get_bbox_patch(); bbox = ann.get_window_extent(renderer=renderer)
                    left = bbox.x0 + left_pad
                    top = bbox.y1 - top_pad
                    if ann._overlay_artists['labels']:
                        _, h0,_ = renderer.get_text_width_height_descent(ann._overlay_artists['labels'][0].get_text(), ann._overlay_artists['labels'][0]._fontproperties, ismath=False)
                        line_h = h0 + 2
                    else:
                        line_h = 12
                    label_right = left + max_label_w
                    colon_x = label_right + colon_gap
                    value_x = colon_x + value_gap
                    inv_fig = fig.transFigure.inverted()
                    for idx,(lab_t, colon_t, val_t) in enumerate(zip(ann._overlay_artists['labels'], ann._overlay_artists['colons'], ann._overlay_artists['values'])):
                        y_px = top - idx * line_h
                        lab_t.set_transform(fig.transFigure); colon_t.set_transform(fig.transFigure); val_t.set_transform(fig.transFigure)
                        lab_t.set_position(inv_fig.transform((label_right, y_px)))
                        colon_t.set_position(inv_fig.transform((colon_x, y_px)))
                        val_t.set_position(inv_fig.transform((value_x, y_px)))
                # pinned annotations
                for i, info in pinned_annotations.items():
                    ann = info['sel'].annotation; close_artist = info['close']
                    if ann is None or close_artist is None: continue
                    try:
                        ensure_overlay(ann)
                        bbox = ann.get_window_extent(renderer=renderer)
                        x = bbox.x1 - 6; y = bbox.y1 - 6
                        inv = fig.transFigure.inverted(); fx, fy = inv.transform((x, y))
                        close_artist.set_transform(fig.transFigure)
                        close_artist.set_position((fx, fy))
                    except Exception: pass
                # hover (not pinned) handled through highlighted_index if present

            fig.canvas.mpl_connect('draw_event', _update_close_positions)

            def _on_click(event):
                # Detect click inside any '×'
                if event.inaxes is None:  # we used figure coords for close buttons
                    # Iterate figure-level close buttons
                    to_remove = []
                    for i, info in pinned_annotations.items():
                        close_artist = info['close']
                        if close_artist is None:
                            continue
                        bbox = close_artist.get_window_extent(renderer=fig.canvas.get_renderer())
                        if bbox.contains(event.x, event.y):
                            # Remove annotation and button
                            try:
                                info['sel'].annotation.remove()
                                close_artist.remove()
                            except Exception:
                                pass
                            to_remove.append(i)
                    if to_remove:
                        for i in to_remove:
                            pinned_annotations.pop(i, None)
                        fig.canvas.draw_idle()
            fig.canvas.mpl_connect('button_press_event', _on_click)
        except ImportError:
            print("[hover] mplcursors 未安装，执行: pip install mplcursors 以启用悬停/固定提示")
        except Exception as e:
            print(f"[hover] 初始化失败: {e}")
    if show:
        plt.show()
    else:
        plt.close(fig)


#############################################
# Legacy Neon Style (short wrapper to reuse if needed)
#############################################

def plot_neon_placeholder():
    print("Neon style function was replaced in this shortened reconstruction. Implement again if needed.")


if __name__ == '__main__':
    # Example usage: generate glass style flower and save PNG + SVG
    out_name = f"glass_flower_2024.png"  # base name; function will also create SVG if save_svg=True
    plot_glass_flower(2024, save_path=out_name, use_cache=True, save_svg=True, show=True)
