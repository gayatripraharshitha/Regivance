import { PageIntro } from "../components/ui/PageIntro";
import { workflow } from "../data/research";

export function Methodology() {
  return <section className="section page"><PageIntro eyebrow="Methodology" title="A chronological experiment, not a random split.">The evaluation imitates repeated forecasting through time and keeps target information outside the model’s permitted history.</PageIntro>
    <div className="timeline">{workflow.map(([n,title,text])=><article key={n}><span>{n}</span><div><h3>{title}</h3><p>{text}</p></div></article>)}</div>
    <div className="fold-diagram"><h3>One walk-forward fold</h3><div className="blocks"><span className="train">1,260 train</span><span className="gap">5 gap</span><span className="validation">252 validation</span><span className="gap">5 gap</span><span className="test">21 test</span></div><p>120 folds advance by 21 observations, producing 2,520 common test origins per model.</p></div>
  </section>;
}
