# Climate Flower Visualization (Detailed Documentation)

## 1. Concept Overview
The **Climate Flower** is a polar (radial) annual climate summary. Each day becomes a bar ("petal"). Visual encodings:
- **Length**: Daily mean temperature (normalized per chosen year).
- **Color (amber→blue) & Alpha**: Daily rainfall; robust scaling by the 90th percentile (p90) so a few stormy days do not crush contrast.
- **Narrow Inner Streak (lighter slim bar)**: Higher wind speed days (filtered by an adaptive threshold to prevent clutter).

Goals: immediate seasonal shape perception (temperature curve), quick spotting of wet periods (monsoon cluster of blue petals), and subtle wind emphasis without overpowering color coding.

## 2. File Structure
```
climateFlower.py       # Main script/module
.cache/                # Cached API responses (npz + csv)
glass_flower_2024.png  # Example output image
glass_flower_2024.svg  # Example vector export
README.md              # Summary README
README_FULL.md         # This detailed document
```

## 3. Data Pipeline
### 3.1 Fetch Function
`fetch_daily_basic(year, use_cache=True, cache_dir='.cache', csv=True)`
- URL: Open-Meteo archival API requesting: `temperature_2m_mean`, `rain_sum`, `windspeed_10m_max`.
- Leap year handling: removes Feb 29 to keep 365 petals.
- Masks entries with any NaN among required fields.
- Caching: Compressed `.npz` + optional CSV mirror for human inspection.
- Fallback: If network fails, synthesizes a plausible seasonal set:
  - Temperature: sinusoid.
  - Rain: gamma distributions (wetter mid-year window).
  - Wind: sinusoid + noise, clipped at 0.

### 3.2 Normalizations
- Temperature: min–max across valid days → radial length.
- Rain: divide by robust scale = p90 (or max fallback) to `[0,1]` → color blend & alpha.
- Wind: p80 scaling with exponent 1.3 (soft easing) → intensity for candidate streak days; filter by dynamic threshold: `max(0.08, median*0.85)`.

## 4. Visual Encoding Details
| Aspect | Method |
|--------|--------|
| Radial bars | `ax.bar` with base radius `r_base` + height from normalized temperature |
| Glow layering | Two bar layers: faint outer glow (alpha reduced) + main bar |
| Color ramp | Linear blend: dry color `(1.0, 0.85, 0.40)` → wet color `(0.05, 0.22, 0.65)`; slight nonlinear darkening for wetter days (`cols ** (1 + rn*0.6)`) |
| Alpha | `0.25 + 0.55 * rain_factor` |
| Wind streaks | Narrower bars (≈42% width of main) slightly brighter / desaturated; alpha ≈0.06–0.28 |
| Month guides | Thin radial lines first day of each month, very low alpha |
| Labels | Month abbreviations JAN..DEC (clockwise, North=Jan 1) |
| Background | Concentric translucent circles (manual radial glow) |

## 5. Interactivity
- Requires `mplcursors` (hover only in interactive backends like TkAgg, Qt, etc.).
- Hover: Lightly brightens bar + tooltip with: Date / Temp / Rain / Max Wind.
- No pinned annotations (simplified from earlier experimental versions).

## 6. Public API
```python
from climateFlower import plot_glass_flower
plot_glass_flower(year=2024,
                  save_path='glass_flower_2024.png',
                  dpi=250,
                  use_cache=True,
                  save_svg=False,
                  cache_dir='.cache',
                  show=True,
                  enable_hover=True)
```
### Parameters
- `year` (int): Target year.
- `save_path` (str|None): If set, saves PNG (and SVG if `save_svg=True`).
- `dpi` (int): Export DPI for raster.
- `use_cache` (bool): Use & update local cache.
- `save_svg` (bool): Write vector copy.
- `cache_dir` (str): Folder for cache.
- `show` (bool): Display figure or close (for batch export).
- `enable_hover` (bool): Turn on hover tooltips.

### Returns
`None` (side effects: file output + interactive window). To integrate into another script, you could adapt to return `(fig, ax)` if needed.

## 7. Customization Recipes
### 7.1 Change Color Ramp
Replace `c_dry` / `c_wet` in `_color_from_rain` or inject a new function.

### 7.2 Remove Wind Streaks
Comment out wind streak `if wind_intensity.size:` block.

### 7.3 Different Start Angle
Use `ax.set_theta_zero_location('N')` and `ax.set_theta_direction(-1)` — change to `'E'` or reverse direction for alternative orientations.

### 7.4 Thicker Month Lines / Quarter Highlights
Add a condition: if month in (1,4,7,10) use larger linewidth and maybe label color accent.

### 7.5 Return Figure
At end of `plot_glass_flower`, instead of closing, `return fig, ax` (adjust external call expectations).

## 8. Performance Notes
- 365 bars + optional wind bars is cheap (<50 ms usual on modern machine for rendering). SVG export heavier due to path nodes.
- Caching prevents network latency on subsequent runs.

## 9. Troubleshooting
| Symptom | Cause | Fix |
|---------|-------|-----|
| No hover tooltip | Non-interactive backend | Run in standard Python (not headless), ensure `pip install mplcursors` |
| All bars same color | Rain all zeros or p90 tiny | Check raw rain values; fallback scaling uses `max()` if p90≈0 |
| 366 petals | Leap day not removed | Ensure year has Feb 29; function discards it automatically; verify dates list length=365 |
| Slow first run | Network fetch + cache build | Subsequent runs use `.cache/` |
| Wrong locale month names | Using `strftime('%m')` older version | Updated to `calendar.month_abbr` + `.upper()` |

## 10. Extending to Multiple Years
Loop across years and call `plot_glass_flower(year, save_path=f'glass_flower_{year}.png', show=False)` collecting results; mosaic in a separate figure or HTML gallery.

## 11. Possible Enhancements
- Add CLI arguments via `argparse` (year, no-wind, dark/light theme toggle).
- Provide JSON export of normalized metrics.
- Animate daily growth through the year (matplotlib FuncAnimation or export frames → GIF).
- Add colorblind-friendly alternate palette.

## 12. Dependencies
Core: `numpy`, `requests`, `matplotlib`, `mplcursors` (optional for hover). No dedicated `requirements.txt` yet; create one if distributing.

## 13. License
Add a `LICENSE` file (e.g. MIT) if sharing publicly.

## 14. Acknowledgements
- Data courtesy of Open-Meteo (public free API).
- Design concept inspired by concentric radial climatology diagrams.

---
*This document reflects the simplified hover version (overlay/glow debug code removed). For richer interaction (pinned annotations, semantic highlight) you can reintroduce earlier experimental logic.*
