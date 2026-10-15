# Regivance web dashboard

React/Vite presentation layer for the Regivance research system. It reads curated, versioned development results and does not retrain models in the browser.

The default appearance is white with blue accents. The navigation theme switch restores a saved light/dark preference. Results include metric switching, family filters, model search and individual model inspection. The feature page includes an animated regime explainer. Page entrances, hover effects and chart animations respect reduced-motion preferences.

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

The Python project remains the research engine. The web app is a fast presentation layer built from reviewed summary artifacts. `scripts/export_dashboard_data.py` validates the research model set and sample size, then creates `src/data/research.generated.json` for the dashboard. A future read-only API can serve larger histories without running long model-training jobs during a page request.

Recommended scalable additions:

- `src/services/` for typed API clients when a backend exists.
- `src/hooks/` for shared asynchronous state.
- `src/features/` for larger self-contained product modules.
- `src/lib/` for formatters, validators and chart helpers.
- A Python export command that transforms verified report CSVs into a versioned web JSON bundle.
- Object storage/CDN for large forecast series; never commit raw source data or secrets.

The current demo intentionally needs no account, database or live trading feed.
