# Regivance web dashboard

React/Vite presentation layer for the Regivance research system. It reads curated, versioned development results and does not retrain models in the browser.

The interaction layer uses Framer Motion for route, ambient, metric and card animation; Recharts for research visualizations; and Lucide for accessible icons. Reduced-motion preferences disable nonessential movement. A top-level recovery boundary prevents a single presentation component from blanking the entire application.

The finalized `test` branch uses the **Warm Tones** dark direction: soft-black `#1C1C1C` surfaces, warm-beige `#F5E8D8` typography, golden interaction accents, coral statistical forecasts, muted-rose regime and neural signals, amber stress warnings, and deeper-coral failures. Soft ambient gradients and editorial display typography create a premium, calm research aesthetic without blue. The white-and-blue light theme remains independently optimized for contrast, while all wording and content stay identical between themes.

The default appearance is white with blue accents. The navigation theme switch restores a saved light/dark preference. Results include metric switching, family filters, model search and individual model inspection. The feature page includes an animated regime explainer. The dedicated **Risk analysis** route visualizes stress degradation, VaR calibration, neural seed sensitivity, rapid-rise underprediction and alternative-portfolio robustness. The **Data lab** route validates a user-selected CSV locally, reproduces the portfolio/target transformations and exports a processed file without uploading source data. Page entrances, hover effects and chart animations respect reduced-motion preferences.

## Data lab contract

The CSV must contain a unique header, one chronological date column and at least one numeric return column. Dates may use ISO format or `YYYYMMDD`. Select whether returns are percentages (`1 = 1%`) or decimals (`0.01 = 1%`). Files are limited to 10 MB and 100,000 rows for browser safety.

The Data lab computes equal-weighted portfolio returns, 21-observation trailing root-mean-square volatility and the five-observation forward root-mean-square target used by the research project. It intentionally does not train or run forecasting models.

## API explorer

Start the read-only Python API from the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.app:app --reload
```

Then start the dashboard normally and open `/api`. The explorer reads versioned metadata, reviewed model scores and a paginated forecast history. Set `VITE_API_BASE_URL` when the API is hosted somewhere other than `http://127.0.0.1:8000`. The API remains optional; all static dashboard pages continue to work while it is offline.

## Structure

```text
website/
├── src/
│   ├── components/
│   │   ├── charts/       Recharts visualizations
│   │   ├── layout/       Shared navigation, footer and page shell
│   │   └── ui/           Small reusable presentation components
│   ├── data/             Curated research results for the static demo
│   ├── pages/            Route-level screens
│   ├── styles/           Global design system and responsive layout
│   ├── test/             Test setup
│   ├── App.jsx           Routes
│   └── main.jsx          Browser entry point
├── index.html
├── package.json
└── vite.config.js
```

## Run

```powershell
cd website
npm.cmd install
npm.cmd run dev
```

Validation:

```powershell
npm.cmd test
npm.cmd run build
```

## Architecture boundary

The Python project remains the research engine. The web app is a fast presentation layer built from reviewed summary artifacts. `scripts/export_dashboard_data.py` validates the research model set, sample size and V2 evidence-table schemas, then creates `src/data/research.generated.json` for the dashboard. A future read-only API can serve larger histories without running long model-training jobs during a page request.

Recommended scalable additions:

- `src/services/` for typed API clients when a backend exists.
- `src/hooks/` for shared asynchronous state.
- `src/features/` for larger self-contained product modules.
- `src/lib/` for formatters, validators and chart helpers.
- A Python export command that transforms verified report CSVs into a versioned web JSON bundle.
- Object storage/CDN for large forecast series; never commit raw source data or secrets.

The current demo intentionally needs no account, database or live trading feed.
