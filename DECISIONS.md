# Decisions Log — Case 3: Food Delivery Demand Pulse

## Assumptions I made
1. **Surge hours are uniformly applied across cities** — the brief says "over-paying surge during hours that aren't actually peak", implying a blanket rule.
2. **The Ops Head is non-technical** — brief says "written for the Ops Head, not for engineers". All recommendations use plain language with ₹ impact numbers.
3. **7-day daily forecast is more useful than hourly** — hourly across 7 cities = 1,176 points of noise. Daily is actionable.
4. **Rider incentive cost ≈ 15% of order value during surge** — assumed for revenue calculations since actual cost structure isn't given.
5. **"Peak" = lunch (12–14h) + dinner (19–22h)** — validated by the heatmap showing clear bimodal demand.

## Trade-offs
| Choice | Alternative | Why I picked this |
|---|---|---|
| Holt-Winters (ETS) | Prophet / ARIMA | Simpler, fewer dependencies, handles weekly seasonality well for 90-day data. Prophet is overkill for 7-day horizon. |
| Streamlit | Jupyter / Dash | Ops Head needs a click-and-explore tool, not a notebook. Deploys free on HF Spaces. |
| Plotly (interactive) | Matplotlib (static) | Interactive charts let the Ops Head filter by city/cuisine without requesting a new analysis. |
| Per-city analysis | One aggregate model | The brief says "peak demand is more nuanced" — aggregating IS the current problem. |
| 95% confidence intervals | Point forecast only | Honest uncertainty communication. A forecast without CI is a guess pretending to be a prediction. |
| Data-driven recommendation numbers | Hardcoded estimates | Recommendations compute actual surge waste % from the data in real-time, not made-up numbers. |

## What I de-scoped and why
- **Weather/holiday augmentation** — Needs external API, adds complexity, marginal gain for a 1-day build.
- **Prophet comparison** — Holt-Winters is sufficient to demonstrate the approach; Prophet adds ~30 min of tuning.
- **Notebook version** — Dashboard IS the deliverable. Notebook would duplicate the same analysis in a less usable format.

## What I'd do differently with another day
- Add Prophet + ARIMA ensemble and compare forecast accuracy
- Build rider supply-demand simulation to quantify surge savings precisely
- Integrate public holiday calendar (Republic Day, Holi, etc.) as forecast features
- Design full A/B test framework with power analysis for Recommendation 1
- Add Streamlit authentication for role-based access
