import { motion } from "framer-motion";
import { PageIntro } from "../components/ui/PageIntro";
import { regimeFeatures } from "../data/research";
import { RegimeExplorer } from "../components/ui/RegimeExplorer";

export function DataFeatures() {
  return <section className="section page"><PageIntro eyebrow="Data & features" title="From thirty industries to one risk signal.">The pipeline preserves source chronology, builds a daily equal-industry portfolio and produces features using information available through each forecast origin.</PageIntro>
    <div className="data-contract"><div><small>INPUT</small><strong>30 industry returns</strong><span>Value-weighted source table · decimal units</span></div><i>→</i><div><small>PORTFOLIO</small><strong>1/30 per industry</strong><span>Daily rebalancing</span></div><i>→</i><div><small>TARGET</small><strong>Next 5 observations</strong><span>Root sum of squared returns</span></div></div>
    <div className="feature-grid">{regimeFeatures.map((feature, i) => <motion.article key={feature.title} initial={{opacity:0,y:18}} whileInView={{opacity:1,y:0}} viewport={{once:true}} transition={{delay:i*.1}}><span>0{i+1}</span><h3>{feature.title}</h3><code>{feature.formula}</code><p>{feature.body}</p></motion.article>)}</div>
    <RegimeExplorer />
    <div className="callout"><strong>Timing rule</strong><p>Rolling features use only observations through date t. The five future returns used for the target are labels—not inputs.</p></div>
  </section>;
}
