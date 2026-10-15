import { useMemo, useState } from "react";
import { PageIntro } from "../components/ui/PageIntro";
import { MetricChart } from "../components/charts/MetricChart";
import { modelResults } from "../data/research";

export function ModelResults() {
  const [metric, setMetric] = useState("qlike");
  const [family, setFamily] = useState("All");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState(modelResults[0]);
  const sorted = useMemo(() => modelResults.filter(row => (family === "All" || row.family === family) && row.model.toLowerCase().includes(query.toLowerCase())).sort((a,b) => a[metric]-b[metric]), [metric, family, query]);
  return <section className="section page"><PageIntro eyebrow="Development results" title="Every model. Same dates. Clear ranking.">Switch metrics to inspect the full comparison. Lower is better; these results describe an inspected development period, not an untouched future holdout.</PageIntro>
    <div className="results-controls">
      <div className="metric-switch" role="group" aria-label="Select metric">{["qlike","mae","rmse"].map(item => <button key={item} aria-pressed={metric===item} className={metric===item?"selected":""} onClick={()=>setMetric(item)}>{item.toUpperCase()}</button>)}</div>
      <label>Model family<select value={family} onChange={event => setFamily(event.target.value)}>{["All", "Statistical", "Baseline", "Regime", "Neural"].map(name => <option key={name}>{name}</option>)}</select></label>
      <label>Find a model<input type="search" placeholder="Search models…" value={query} onChange={event => setQuery(event.target.value)} /></label>
    </div>
    <p className="result-count" role="status">{sorted.length} of {modelResults.length} models shown · ranked by {metric.toUpperCase()}</p>
    {sorted.length ? <>
      <div className="results-layout"><div className="chart-card"><MetricChart data={sorted} metric={metric}/></div><aside className="winner-card"><span>BEST SHOWN {metric.toUpperCase()}</span><strong>{sorted[0][metric].toFixed(6)}</strong><h3>{sorted[0].model}</h3><p>{sorted[0].family} family</p><hr/><small>THE TAKEAWAY</small><p>Regime information is useful context, but did not consistently outperform the strongest statistical benchmark.</p></aside></div>
      <div className="table-wrap"><table><thead><tr><th>Rank</th><th>Model · select to inspect</th><th>Family</th><th>MAE</th><th>RMSE</th><th>QLIKE</th></tr></thead><tbody>{sorted.map((row,i)=><tr key={row.model} className={selected.model===row.model?"selected-row":""}><td>{String(i+1).padStart(2,"0")}</td><td><button className="model-select" aria-pressed={selected.model===row.model} onClick={()=>setSelected(row)}>{row.model}</button></td><td><span className={`family ${row.family.toLowerCase()}`}>{row.family}</span></td><td>{row.mae.toFixed(6)}</td><td>{row.rmse.toFixed(6)}</td><td>{row.qlike.toFixed(6)}</td></tr>)}</tbody></table></div>
    </> : <div className="empty-state"><h3>No matching models</h3><button className="button" onClick={()=>{setFamily("All");setQuery("");}}>Reset filters</button></div>}
    <section className="model-detail" aria-label="Selected model details"><span className="eyebrow">Selected model</span><h3>{selected.model}</h3><p>{selected.family} · 2,520 development origins</p><div>{["mae","rmse","qlike"].map(key=><span key={key}><small>{key.toUpperCase()}</small><strong>{selected[key].toFixed(6)}</strong></span>)}</div></section>
  </section>;
}
