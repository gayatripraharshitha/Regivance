import { useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { CheckCircle2, Download, FileUp, LockKeyhole, RotateCcw, TriangleAlert } from "lucide-react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { PageIntro } from "../components/ui/PageIntro";
import { parseCsv, processReturns, toProcessedCsv } from "../lib/dataLab";

const SAMPLE = `date,industry_a,industry_b,industry_c\n2026-01-02,0.20,0.10,-0.10\n2026-01-05,-0.40,-0.20,-0.30\n2026-01-06,0.30,0.50,0.20\n2026-01-07,0.15,0.12,0.18\n2026-01-08,-0.10,0.05,0.02\n2026-01-09,0.40,0.32,0.28\n2026-01-12,0.08,0.09,0.06\n2026-01-13,-0.22,-0.18,-0.25\n2026-01-14,0.11,0.14,0.09\n2026-01-15,0.25,0.21,0.20\n2026-01-16,-0.07,-0.10,-0.06\n2026-01-20,0.19,0.22,0.17\n2026-01-21,0.05,0.02,0.07\n2026-01-22,-0.18,-0.15,-0.12\n2026-01-23,0.31,0.28,0.35\n2026-01-26,0.09,0.12,0.11\n2026-01-27,-0.11,-0.09,-0.13\n2026-01-28,0.27,0.30,0.24\n2026-01-29,0.06,0.08,0.04\n2026-01-30,-0.20,-0.16,-0.19\n2026-02-02,0.38,0.35,0.42\n2026-02-03,0.12,0.10,0.14\n2026-02-04,-0.08,-0.05,-0.11\n2026-02-05,0.17,0.20,0.15\n2026-02-06,0.04,0.07,0.03\n2026-02-09,-0.13,-0.15,-0.10\n2026-02-10,0.29,0.25,0.31\n2026-02-11,0.10,0.12,0.08\n2026-02-12,-0.06,-0.03,-0.07\n2026-02-13,0.21,0.18,0.23`;

function saveText(name, text) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a"); link.href = url; link.download = name; link.click(); URL.revokeObjectURL(url);
}

export function DataLab() {
  const inputRef = useRef(null);
  const [source, setSource] = useState(null);
  const [name, setName] = useState("");
  const [dateColumn, setDateColumn] = useState("");
  const [selected, setSelected] = useState([]);
  const [unit, setUnit] = useState("percent");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const numericCandidates = useMemo(() => source?.headers.filter((header) => header !== dateColumn) ?? [], [source, dateColumn]);

  const loadText = (text, fileName) => {
    try {
      const parsed = parseCsv(text);
      setSource(parsed); setName(fileName); setDateColumn(parsed.headers[0]); setSelected(parsed.headers.slice(1)); setResult(null); setError("");
    } catch (problem) { setError(problem.message); setSource(null); setResult(null); }
  };
  const loadFile = async (file) => {
    if (!file) return;
    if (file.size > 10 * 1024 * 1024) { setError("Choose a CSV smaller than 10 MB."); return; }
    if (!file.name.toLowerCase().endsWith(".csv")) { setError("Only CSV files are accepted."); return; }
    loadText(await file.text(), file.name);
  };
  const run = () => {
    try { setResult(processReturns({ records: source.records, dateColumn, returnColumns: selected, unit })); setError(""); }
    catch (problem) { setError(problem.message); setResult(null); }
  };
  const reset = () => { setSource(null); setResult(null); setError(""); setName(""); if (inputRef.current) inputRef.current.value = ""; };
  const toggle = (column) => setSelected((current) => current.includes(column) ? current.filter((value) => value !== column) : [...current, column]);
  const chartData = result?.filter((row) => row.trailingVolatility !== null && row.forwardTarget !== null).slice(-500) ?? [];
  const complete = result?.filter((row) => row.trailingVolatility !== null && row.forwardTarget !== null).length ?? 0;

  return <section className="section page data-lab">
    <PageIntro eyebrow="Browser-local experiment" title={<>Run your <em>own data</em>.</>}>Upload a chronological CSV of daily returns, inspect its structure and reproduce the project’s portfolio and volatility transformations without sending the file to a server.</PageIntro>
    <div className="privacy-strip"><LockKeyhole /><div><strong>Private by design</strong><span>Your file is processed only inside this browser tab. It is not uploaded, stored or used to retrain a model.</span></div></div>

    {!source ? <div className="upload-zone" onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); loadFile(event.dataTransfer.files[0]); }}>
      <FileUp size={38} /><h2>Drop a CSV here</h2><p>Maximum 10 MB and 100,000 rows. The first column should usually be the date.</p>
      <button className="button primary" onClick={() => inputRef.current?.click()}>Choose CSV</button>
      <button className="text-button" onClick={() => loadText(SAMPLE, "regivance_sample.csv")}>Or load the sample</button>
      <input ref={inputRef} type="file" accept=".csv,text/csv" hidden onChange={(event) => loadFile(event.target.files[0])} />
    </div> : <div className="lab-workspace">
      <section className="lab-config"><div className="file-badge"><CheckCircle2 /><div><strong>{name}</strong><span>{source.records.length.toLocaleString()} rows · {source.headers.length} columns</span></div><button onClick={reset} aria-label="Clear uploaded data"><RotateCcw /></button></div>
        <label>Date column<select value={dateColumn} onChange={(event) => { setDateColumn(event.target.value); setSelected((current) => current.filter((item) => item !== event.target.value)); setResult(null); }}>{source.headers.map((header) => <option key={header}>{header}</option>)}</select></label>
        <fieldset><legend>Return columns</legend><p>Select one asset or several columns for an equal-weighted portfolio.</p><div className="column-picker">{numericCandidates.map((column) => <label key={column}><input type="checkbox" checked={selected.includes(column)} onChange={() => { toggle(column); setResult(null); }} />{column}</label>)}</div></fieldset>
        <fieldset><legend>Input unit</legend><div className="unit-picker"><button className={unit === "percent" ? "selected" : ""} onClick={() => { setUnit("percent"); setResult(null); }}>Percent (1 = 1%)</button><button className={unit === "decimal" ? "selected" : ""} onClick={() => { setUnit("decimal"); setResult(null); }}>Decimal (0.01 = 1%)</button></div></fieldset>
        <button className="button primary run-button" onClick={run}>Validate and calculate</button>
      </section>
      <section className="lab-preview"><span className="eyebrow">Source preview</span><div className="table-wrap"><table><thead><tr>{source.headers.map((header) => <th key={header}>{header}</th>)}</tr></thead><tbody>{source.records.slice(0,5).map((record, index) => <tr key={index}>{source.headers.map((header) => <td key={header}>{record[header]}</td>)}</tr>)}</tbody></table></div><small>Showing the first 5 rows. Calculations use the complete file.</small></section>
    </div>}

    <AnimatePresence mode="wait">{error && <motion.div className="lab-message error" role="alert" initial={{ opacity:0,y:8 }} animate={{ opacity:1,y:0 }} exit={{ opacity:0 }}><TriangleAlert />{error}</motion.div>}</AnimatePresence>
    {result && <motion.section className="lab-results" initial={{ opacity:0,y:18 }} animate={{ opacity:1,y:0 }}>
      <div className="lab-result-head"><div><span className="eyebrow">Validated result</span><h2>Your transformed volatility series</h2><p>The first 20 trailing values and final 5 targets are intentionally unavailable because their complete windows do not exist.</p></div><button className="button" onClick={() => saveText("regivance_processed.csv", toProcessedCsv(result))}><Download size={15} />Download processed CSV</button></div>
      <div className="lab-stats"><article><span>Valid observations</span><strong>{result.length.toLocaleString()}</strong></article><article><span>Complete comparisons</span><strong>{complete.toLocaleString()}</strong></article><article><span>Portfolio columns</span><strong>{selected.length}</strong></article></div>
      <div className="lab-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={chartData}><CartesianGrid stroke="var(--line)" vertical={false} /><XAxis dataKey="date" minTickGap={50} stroke="var(--muted)" tickLine={false} axisLine={false} /><YAxis stroke="var(--muted)" tickLine={false} axisLine={false} tickFormatter={(value) => `${(value*100).toFixed(1)}%`} /><Tooltip formatter={(value) => `${(Number(value)*100).toFixed(3)}%`} /><Legend /><Line type="monotone" dataKey="trailingVolatility" name="21-observation trailing volatility" stroke="var(--green)" dot={false} strokeWidth={2} /><Line type="monotone" dataKey="forwardTarget" name="5-observation forward target" stroke="var(--amber)" dot={false} strokeWidth={2} /></LineChart></ResponsiveContainer></div>
      <div className="lab-explanation"><article><b>1</b><div><strong>Portfolio return</strong><p>Average the selected return columns on each date.</p></div></article><article><b>2</b><div><strong>Trailing volatility</strong><p>Root mean square of the latest 21 portfolio returns, including today.</p></div></article><article><b>3</b><div><strong>Forward target</strong><p>Root mean square of the next five portfolio returns. This is a research label, not a forecast produced by the browser.</p></div></article></div>
    </motion.section>}
    <p className="fine-print">This educational tool performs deterministic transformations only. It does not run GARCH, regime clustering or LSTM inference, and its output is not investment advice.</p>
  </section>;
}
