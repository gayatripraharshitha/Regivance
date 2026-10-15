import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle2, Download, ShieldCheck } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { PageIntro } from "../components/ui/PageIntro";
import { alternativePortfolio, rapidRise, seedSensitivity, stressPeriods, varCoverage } from "../data/research";

const GJR = "GJR-GARCH-Student-t";
const LSTM = "Risk-feature + regime LSTM";
const shortModel = (value) => value === GJR ? "GJR-GARCH-t" : "Regime LSTM";
const label = (value) => value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
const pct = (value) => `${(value * 100).toFixed(1)}%`;

function ResearchTooltip({ active, payload, label: chartLabel }) {
  if (!active || !payload?.length) return null;
  return <div className="research-tooltip"><strong>{label(String(chartLabel))}</strong>{payload.map((item) => <span key={item.dataKey} style={{ color: item.color }}>{item.name}: {Number(item.value).toFixed(3)}</span>)}</div>;
}

function DownloadLink({ file, children }) {
  return <a className="button download-link" href={`https://github.com/gayatripraharshitha/Regivance/raw/v2/reports/tables/${file}`}><Download size={15} />{children}</a>;
}

export function RiskAnalysis() {
  const [stressMetric, setStressMetric] = useState("qlike");
  const [confidence, setConfidence] = useState(0.95);
  const [riseModel, setRiseModel] = useState(GJR);
  const stressChart = useMemo(() => ["normal", "high_volatility", "2020_shock"].map((period) => {
    const rows = stressPeriods.filter((row) => row.period === period);
    return { period: label(period), gjr: rows.find((row) => row.model === GJR)?.[stressMetric], lstm: rows.find((row) => row.model === LSTM)?.[stressMetric] };
  }), [stressMetric]);
  const varRows = varCoverage.filter((row) => row.sample === "overlapping_all" && row.confidence === confidence);
  const riseRows = rapidRise.filter((row) => row.model === riseModel);
  const seedMean = seedSensitivity.reduce((sum, row) => sum + row.qlike, 0) / seedSensitivity.length;

  return <section className="section page risk-page">
    <PageIntro eyebrow="V2 risk analysis" title={<>Where forecasts <em>break</em>.</>}>Accuracy averages can conceal the dates that matter most. This page tests shock behavior, risk calibration, neural stability and portfolio robustness using verified report artifacts.</PageIntro>

    <div className="risk-summary" aria-label="Risk analysis summary">
      <article><ShieldCheck /><span>Most dependable</span><strong>GJR-GARCH-t</strong><small>Best overall and stress-period QLIKE</small></article>
      <article><AlertTriangle /><span>Main failure</span><strong>88.7%</strong><small>LSTM underprediction in the 2020 shock</small></article>
      <article><CheckCircle2 /><span>95% VaR calibration</span><strong>4.96%</strong><small>GJR observed breach rate; target 5%</small></article>
    </div>

    <section className="risk-block" aria-labelledby="stress-title">
      <div className="risk-heading"><div><span className="eyebrow">01 · Stress behavior</span><h2 id="stress-title">Errors expand as markets become unstable.</h2><p>The same dates are used for both models. Switch the metric to inspect the deterioration from normal observations to the 2020 shock.</p></div><div className="metric-switch" aria-label="Stress metric">{["qlike", "mae", "rmse"].map((metric) => <button key={metric} className={stressMetric === metric ? "selected" : ""} onClick={() => setStressMetric(metric)}>{metric.toUpperCase()}</button>)}</div></div>
      <div className="risk-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={stressChart}><CartesianGrid stroke="var(--line)" vertical={false} /><XAxis dataKey="period" stroke="var(--muted)" tickLine={false} axisLine={false} /><YAxis stroke="var(--muted)" tickLine={false} axisLine={false} /><Tooltip content={<ResearchTooltip />} /><Legend /><Bar name="GJR-GARCH-t" dataKey="gjr" fill="var(--green)" radius={[7,7,0,0]} /><Bar name="Regime LSTM" dataKey="lstm" fill="var(--amber)" radius={[7,7,0,0]} /></BarChart></ResponsiveContainer></div>
      <div className="evidence-note warning"><AlertTriangle /><p><strong>Observed weakness:</strong> the regime LSTM’s 2020-shock QLIKE rises to 8.12 versus 1.06 for GJR-GARCH-t. Complexity did not provide crisis protection.</p></div>
      <DownloadLink file="v2_stress_period_metrics.csv">Download stress data</DownloadLink>
    </section>

    <section className="risk-block" aria-labelledby="var-title">
      <div className="risk-heading"><div><span className="eyebrow">02 · VaR calibration</span><h2 id="var-title">Does predicted risk match realized loss frequency?</h2><p>A well-calibrated model should place the observed breach bar near the expected line. This is a demonstration using five-observation forecasts and overlapping returns, not a production risk engine.</p></div><div className="metric-switch" aria-label="VaR confidence">{[0.95,0.99].map((level) => <button key={level} className={confidence === level ? "selected" : ""} onClick={() => setConfidence(level)}>{Math.round(level*100)}% VaR</button>)}</div></div>
      <div className="risk-chart compact"><ResponsiveContainer width="100%" height="100%"><BarChart data={varRows.map((row) => ({ ...row, modelLabel: shortModel(row.model), observed: row.observed_breach_rate * 100 }))}><CartesianGrid stroke="var(--line)" vertical={false} /><XAxis dataKey="modelLabel" stroke="var(--muted)" tickLine={false} axisLine={false} /><YAxis unit="%" stroke="var(--muted)" tickLine={false} axisLine={false} /><Tooltip formatter={(value) => `${Number(value).toFixed(2)}%`} /><ReferenceLine y={(1-confidence)*100} stroke="var(--amber)" strokeDasharray="5 5" label={{ value: `Expected ${((1-confidence)*100).toFixed(0)}%`, fill: "var(--muted)" }} /><Bar name="Observed breaches" dataKey="observed" fill="var(--green)" radius={[7,7,0,0]} /></BarChart></ResponsiveContainer></div>
      <div className="calibration-grid">{varRows.map((row) => <article key={row.model}><span>{shortModel(row.model)}</span><strong>{pct(row.observed_breach_rate)}</strong><small>{row.breaches} breaches from {row.n.toLocaleString()} origins</small><i className={row.exact_binomial_p_value >= .05 ? "pass" : "fail"}>{row.exact_binomial_p_value >= .05 ? "No significant mismatch" : "Calibration mismatch"}</i></article>)}</div>
      <DownloadLink file="v2_var_coverage.csv">Download VaR data</DownloadLink>
    </section>

    <section className="risk-block" aria-labelledby="seed-title">
      <div className="risk-heading"><div><span className="eyebrow">03 · Neural stability</span><h2 id="seed-title">Initialization changes the LSTM result.</h2><p>Five seeds were tested over the same eight stress-inclusive folds and 168 origins. Lower QLIKE is better.</p></div><div className="mini-stat"><span>Mean QLIKE</span><strong>{seedMean.toFixed(3)}</strong></div></div>
      <div className="risk-chart compact"><ResponsiveContainer width="100%" height="100%"><LineChart data={seedSensitivity}><CartesianGrid stroke="var(--line)" vertical={false} /><XAxis dataKey="seed_base" stroke="var(--muted)" tickLine={false} axisLine={false} label={{ value:"Random seed", position:"insideBottom", offset:-4, fill:"var(--muted)" }} /><YAxis domain={[2,3.8]} stroke="var(--muted)" tickLine={false} axisLine={false} /><Tooltip content={<ResearchTooltip />} /><ReferenceLine y={seedMean} stroke="var(--amber)" strokeDasharray="5 5" /><Line name="QLIKE" type="monotone" dataKey="qlike" stroke="var(--green)" strokeWidth={3} dot={{ r:5, fill:"var(--panel)", strokeWidth:3 }} /></LineChart></ResponsiveContainer></div>
      <p className="range-caption">Range: <strong>{Math.min(...seedSensitivity.map((row) => row.qlike)).toFixed(3)}</strong>–<strong>{Math.max(...seedSensitivity.map((row) => row.qlike)).toFixed(3)}</strong>. One neural run is not enough to claim stability.</p>
      <DownloadLink file="v2_lstm_seed_sensitivity.csv">Download seed data</DownloadLink>
    </section>

    <section className="risk-block" aria-labelledby="rise-title">
      <div className="risk-heading"><div><span className="eyebrow">04 · Shock sensitivity</span><h2 id="rise-title">Both models lag rapid volatility increases.</h2><p>Choose a model to see how its underprediction rate changes as realized volatility rises relative to its trailing level.</p></div><select aria-label="Shock sensitivity model" value={riseModel} onChange={(event) => setRiseModel(event.target.value)}><option value={GJR}>GJR-GARCH-t</option><option value={LSTM}>Regime LSTM</option></select></div>
      <div className="rise-grid">{riseRows.map((row, index) => <motion.article key={row.rise_bucket} initial={{ opacity:0, y:10 }} animate={{ opacity:1, y:0 }} transition={{ delay:index*.07 }}><span>{label(row.rise_bucket)}</span><strong>{pct(row.underprediction_rate)}</strong><small>underpredicted · QLIKE {row.qlike.toFixed(2)}</small><div><i style={{ width:pct(row.underprediction_rate) }} /></div></motion.article>)}</div>
      <DownloadLink file="v2_rapid_rise_underprediction.csv">Download shock data</DownloadLink>
    </section>

    <section className="risk-block" aria-labelledby="portfolio-title">
      <div className="risk-heading"><div><span className="eyebrow">05 · Portfolio robustness</span><h2 id="portfolio-title">The statistical ranking survives another weighting rule.</h2><p>The alternative portfolio weights industries inversely to their lagged 63-observation volatility. It is a robustness check, not a transaction-cost-aware strategy.</p></div></div>
      <div className="portfolio-cards">{alternativePortfolio.map((row, index) => <article className={index === 2 ? "best" : ""} key={row.model}><span>{row.model}</span><strong>{row.qlike.toFixed(3)}</strong><small>QLIKE · MAE {row.mae.toFixed(4)}</small>{index === 2 && <b>Strongest</b>}</article>)}</div>
      <DownloadLink file="v2_alternative_portfolio_metrics.csv">Download portfolio data</DownloadLink>
    </section>

    <section className="research-verdict"><span className="eyebrow">Evidence-based conclusion</span><h2>Prefer GJR-GARCH-t today. Keep the LSTM as research.</h2><p>Regime information is interpretable, but it has not produced a reliable improvement over the strongest statistical benchmark. The largest open problem is underprediction during abrupt volatility increases. These are historical development results—not untouched prospective evidence and not investment advice.</p></section>
  </section>;
}
