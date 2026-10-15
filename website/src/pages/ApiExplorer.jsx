import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Braces, CircleOff, RefreshCw, Server } from "lucide-react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { PageIntro } from "../components/ui/PageIntro";
import { researchApi } from "../services/researchApi";

export function ApiExplorer() {
  const [metadata, setMetadata] = useState(null);
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("");
  const [history, setHistory] = useState(null);
  const [status, setStatus] = useState("connecting");
  const [error, setError] = useState("");
  const [range, setRange] = useState({ start:"", end:"" });

  const connect = async () => {
    setStatus("connecting"); setError("");
    try {
      const [meta, scores] = await Promise.all([researchApi.metadata(), researchApi.models()]);
      setMetadata(meta); setModels(scores.items); setModel((current) => current || meta.models[0]); setStatus("online");
    } catch (problem) { setStatus("offline"); setError(problem.message); }
  };
  useEffect(() => { const controller = new AbortController(); Promise.all([researchApi.metadata(controller.signal), researchApi.models(controller.signal)]).then(([meta,scores]) => { setMetadata(meta); setModels(scores.items); setModel(meta.models[0]); setStatus("online"); }).catch((problem) => { if (problem.name !== "AbortError") { setStatus("offline"); setError(problem.message); } }); return () => controller.abort(); }, []);
  const load = async () => {
    setError("");
    try { setHistory(await researchApi.forecasts({ model, ...range })); }
    catch (problem) { setError(problem.message); }
  };
  const score = useMemo(() => models.find((item) => item.model === model), [models, model]);
  const command = `curl "${researchApi.baseUrl}/api/v1/forecasts?model=${encodeURIComponent(model || "GJR-GARCH-Student-t")}&limit=120"`;

  return <section className="section page api-page">
    <PageIntro eyebrow="Read-only service" title={<>Explore the <em>forecast API</em>.</>}>Query reviewed historical development forecasts without loading training code in the browser. The service is versioned, paginated and deliberately read-only.</PageIntro>
    <div className={`api-status ${status}`}><i /> <strong>{status === "online" ? "API connected" : status === "connecting" ? "Connecting to local API" : "API is offline"}</strong><span>{researchApi.baseUrl}</span><button onClick={connect} aria-label="Reconnect to API"><RefreshCw size={16} /></button></div>
    {status === "offline" ? <section className="api-offline"><CircleOff /><div><h2>Start the Python service</h2><p>The dashboard stays usable without it. To activate this explorer, open another PowerShell window at the project root and run:</p><code>.\.venv\Scripts\python.exe -m uvicorn api.app:app --reload</code><small>{error}</small></div></section> : <>
      {metadata && <div className="api-metadata"><article><span>Scope</span><strong>{metadata.scope.replaceAll("_"," ")}</strong></article><article><span>Dates</span><strong>{metadata.startDate} → {metadata.endDate}</strong></article><article><span>Observations</span><strong>{metadata.observations.toLocaleString()}</strong></article><article><span>Models</span><strong>{metadata.models.length}</strong></article></div>}
      <section className="api-console"><div className="api-controls"><label>Model<select value={model} onChange={(event) => { setModel(event.target.value); setHistory(null); }}>{metadata?.models.map((name) => <option key={name}>{name}</option>)}</select></label><label>Start date<input type="date" value={range.start} onChange={(event) => setRange({...range,start:event.target.value})} /></label><label>End date<input type="date" value={range.end} onChange={(event) => setRange({...range,end:event.target.value})} /></label><button className="button primary" onClick={load} disabled={!model}>Load forecast history</button></div>
        {score && <div className="api-score"><span>Reviewed pooled score</span><strong>QLIKE {score.qlike.toFixed(4)}</strong><small>MAE {score.mae.toFixed(4)} · RMSE {score.rmse.toFixed(4)}</small></div>}
      </section>
      {error && <div className="lab-message error" role="alert">{error}</div>}
      {history && <motion.section className="api-output" initial={{opacity:0,y:14}} animate={{opacity:1,y:0}}><div><span className="eyebrow">Returned records</span><h2>{history.model}</h2><p>{history.count} displayed from {history.total.toLocaleString()} matching observations.</p></div><div className="api-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={history.items}><CartesianGrid stroke="var(--line)" vertical={false}/><XAxis dataKey="date" minTickGap={45} stroke="var(--muted)" tickLine={false} axisLine={false}/><YAxis stroke="var(--muted)" tickLine={false} axisLine={false}/><Tooltip formatter={(value) => Number(value).toFixed(6)}/><Legend/><Line dataKey="actual" name="Realized target" stroke="var(--amber)" dot={false} strokeWidth={2}/><Line dataKey="forecast" name="Forecast" stroke="var(--green)" dot={false} strokeWidth={2}/></LineChart></ResponsiveContainer></div></motion.section>}
      <section className="api-developer"><Braces/><div><span className="eyebrow">Developer access</span><h2>Use the same evidence in another client.</h2><pre><code>{command}</code></pre><p>Interactive OpenAPI documentation is available at <a href={`${researchApi.baseUrl}/docs`} target="_blank" rel="noreferrer">{researchApi.baseUrl}/docs</a> while the server runs.</p></div></section>
    </>}
    <p className="fine-print"><Server size={13}/> Historical development artifacts only. No live market data, prospective claims, order execution or investment advice.</p>
  </section>;
}
