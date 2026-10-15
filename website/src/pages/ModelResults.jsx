import { useMemo, useState } from "react";
import { PageIntro } from "../components/ui/PageIntro";
import { MetricChart } from "../components/charts/MetricChart";
import { modelResults } from "../data/research";

export function ModelResults() {
  const [metric, setMetric] = useState("qlike");
  const sorted = useMemo(() => [...modelResults].sort((a,b) => a[metric]-b[metric]), [metric]);
  return <section className="section page"><PageIntro eyebrow="Development results" title="Every model. Same dates. Clear ranking.">Switch metrics to inspect the full comparison. Lower is better; these results describe an inspected development period, not an untouched future holdout.</PageIntro>
    <div className="metric-switch" role="group" aria-label="Select metric">{["qlike","mae","rmse"].map(item => <button key={item} className={metric===item?"selected":""} onClick={()=>setMetric(item)}>{item.toUpperCase()}</button>)}</div>
    <div className="results-layout"><div className="chart-card"><MetricChart data={sorted} metric={metric}/></div><aside className="winner-card"><span>BEST {metric.toUpperCase()}</span><strong>{sorted[0][metric].toFixed(6)}</strong><h3>{sorted[0].model}</h3><p>{sorted[0].family} family</p><hr/><small>THE TAKEAWAY</small><p>Regime information is useful context, but did not consistently outperform the strongest statistical benchmark.</p></aside></div>
    <div className="table-wrap"><table><thead><tr><th>Rank</th><th>Model</th><th>Family</th><th>MAE</th><th>RMSE</th><th>QLIKE</th></tr></thead><tbody>{sorted.map((row,i)=><tr key={row.model}><td>{String(i+1).padStart(2,"0")}</td><td>{row.model}</td><td><span className={`family ${row.family.toLowerCase()}`}>{row.family}</span></td><td>{row.mae.toFixed(6)}</td><td>{row.rmse.toFixed(6)}</td><td>{row.qlike.toFixed(6)}</td></tr>)}</tbody></table></div>
  </section>;
}
