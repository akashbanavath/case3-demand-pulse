# Case 3: Food Delivery Demand Pulse 🔥

**Live demo:** *[HF Spaces URL — deploy pending]*  
**Repo:** [github.com/akashbanavath/case3-demand-pulse](https://github.com/akashbanavath/case3-demand-pulse)  
**Demo video:** *[Recording pending]*

## What this is
A one-day data investigation into when food delivery demand really spikes across 7 Indian cities — and 3 concrete policy recommendations to cut wasted rider-incentive spend by ₹2.5–4L/month, written for a non-technical Ops Head.

## How to run locally
```bash
git clone https://github.com/akashbanavath/case3-demand-pulse.git
cd case3-demand-pulse
pip install -r requirements.txt
python -m streamlit run app.py
```
Open http://localhost:8501

## Stack
| Component | Choice | Why |
|-----------|--------|-----|
| Python 3.11 | Core | Data science standard |
| Pandas + NumPy | Analysis | Fast, mature |
| Plotly | Visualization | Interactive — Ops Head can filter & explore |
| Streamlit | Dashboard | Rapid deploy, free HF Spaces hosting |
| Holt-Winters (statsmodels) | Forecast | Interpretable, handles weekly seasonality, honest |

## Dashboard Tabs
1. **📊 Demand Patterns** — Hourly/daily volume, heatmap, cuisine mix, surge vs demand mismatch
2. **🏙️ City Deep-Dive** — City-level hourly patterns, weekday/weekend split, treemap
3. **🔮 Forecast** — 7-day forecast per city with 95% confidence intervals
4. **💡 Recommendations** — 3 data-backed policies with quantified impact + A/B test design
5. **📋 Exec Summary** — One-page brief for the Ops Head with action table

## Key Findings
1. **~55% of surge is wasted** on non-peak hours (outside 12–14h / 19–22h windows)
2. **Peak demand is city-specific** — Delhi peaks lunch, Mumbai peaks dinner
3. **Weekends are 10–15% busier** but use the same flat surge policy
4. **Tier-2 cities spike sharply** in narrow 2-hour windows — need proactive positioning

## 3 Recommendations (Quantified)
| # | Action | Expected Impact | Effort |
|---|--------|----------------|--------|
| 1 | City-specific surge windows | ₹2.5–4L/month saved | Low — 1 week |
| 2 | Weekend incentive tiers | 8–12% faster weekends | Medium — 2 weeks |
| 3 | Tier-2 rider pre-positioning | 5–10% faster delivery | Medium — 3 weeks |

## What's NOT done
- Prophet/ARIMA comparison (Holt-Winters is sufficient for this horizon)
- Weather/holiday augmentation (needs external API)
- Real-time streaming pipeline

## In production, I would also add
- Real-time Kafka ingestion with live demand monitoring
- Automated daily forecast refresh with accuracy drift alerting
- A/B test framework with auto-kill guardrails
- Weather API integration for forecast improvement
- Role-based access control on the dashboard

## License
MIT
