import { ArrowRight, CheckCircle2, Clock3, Layers3 } from "lucide-react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { PageIntro } from "../components/ui/PageIntro";

const stats = [["11", "forecast models"], ["120", "walk-forward folds"], ["2,520", "origins per model"], ["121", "passing tests"]];

export function Overview() {
  return <>
    <section className="hero section">
      <div>
        <PageIntro eyebrow="Regime-aware volatility research" title={<>Markets change shape.<br/><em>Forecasts should notice.</em></>}>
          Regivance tests whether market-regime information improves five-observation volatility forecasts beyond serious statistical benchmarks.
        </PageIntro>
        <div className="hero-actions"><Link className="button primary" to="/results">Explore results <ArrowRight size={17}/></Link><Link className="button" to="/methodology">View methodology</Link></div>
      </div>
      <motion.div className="signal-panel" initial={{ opacity: 0, scale: .96 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: .2 }}>
        <div className="signal-head"><span>DEVELOPMENT SIGNAL</span><span className="live">● VERIFIED</span></div>
        <div className="signal-score"><small>Best pooled QLIKE</small><strong>0.419191</strong><span>GJR-GARCH · Student-t</span></div>
        <svg className="signal-line" viewBox="0 0 500 150" role="img" aria-label="Decorative volatility signal"><path d="M0 112 C45 108 45 92 87 96 S145 125 178 80 S220 18 250 70 S295 124 330 62 S375 30 410 82 S460 101 500 43"/><path className="glow" d="M0 112 C45 108 45 92 87 96 S145 125 178 80 S220 18 250 70 S295 124 330 62 S375 30 410 82 S460 101 500 43"/></svg>
        <div className="regime-scale"><span>LOWER</span><span>MIDDLE</span><span>HIGHER</span></div>
      </motion.div>
    </section>
    <section className="stat-strip">{stats.map(([value,label], i) => <motion.div key={label} initial={{ opacity:0,y:14 }} whileInView={{opacity:1,y:0}} viewport={{once:true}} transition={{delay:i*.08}}><strong>{value}</strong><span>{label}</span></motion.div>)}</section>
    <section className="section narrative-grid">
      <div><span className="eyebrow">The question</span><h2>Does regime context improve the forecast?</h2><p>Many systems stop after building an LSTM. Regivance makes it compete on identical dates against rolling volatility, EWMA and three GARCH specifications.</p></div>
      <div className="finding-card"><Layers3/><small>DEVELOPMENT FINDING</small><h3>Complexity did not automatically win.</h3><p>Regimes helped some neural comparisons, but the asymmetric Student-t GARCH model remained the strongest pooled benchmark.</p><Link to="/results">See the evidence <ArrowRight size={15}/></Link></div>
    </section>
    <section className="section readiness">
      <div><span className="eyebrow">Project readiness</span><h2>Ready to demonstrate.<br/>Still honest about the boundary.</h2></div>
      <div className="readiness-list"><p><CheckCircle2/> Implementation, development study and report complete</p><p><CheckCircle2/> Reproducible workflow and 121 local tests passing</p><p><Clock3/> Prospective final evaluation awaits future observations</p></div>
    </section>
  </>;
}
