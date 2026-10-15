# Regivance web dashboard

React/Vite presentation layer for the Regivance research system. It reads curated, versioned development results and does not retrain models in the browser.

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

The Python project remains the research engine. The web app is a fast presentation layer built from reviewed summary artifacts. Future integration should use a generated JSON export or read-only API rather than running long model-training jobs during a page request.

Recommended scalable additions:

- `src/services/` for typed API clients when a backend exists.
- `src/hooks/` for shared asynchronous state.
- `src/features/` for larger self-contained product modules.
- `src/lib/` for formatters, validators and chart helpers.
- A Python export command that transforms verified report CSVs into a versioned web JSON bundle.
- Object storage/CDN for large forecast series; never commit raw source data or secrets.

The current demo intentionally needs no account, database or live trading feed.
