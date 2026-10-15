import { Code2, ExternalLink, ShieldCheck } from "lucide-react";
import { PageIntro } from "../components/ui/PageIntro";

export function About() {
  return <section className="section page"><PageIntro eyebrow="About the project" title="Research engineering with an honest conclusion.">Regivance is a college, internship and portfolio project built to connect data validation, econometrics, machine learning and reproducible evaluation.</PageIntro>
    <div className="about-grid"><article><ShieldCheck/><h3>What is complete</h3><p>The research engine, 11-model development comparison, robustness analysis, tests, report and this presentation layer.</p></article><article><Code2/><h3>What comes next</h3><p>The frozen prospective protocol needs 252 future forecast origins plus five observations to form the last target.</p></article></div>
    <a className="button primary github-link" href="https://github.com/gayatripraharshitha/Regivance" target="_blank" rel="noreferrer">Open GitHub repository <ExternalLink size={16}/></a>
    <p className="fine-print">Research and educational use only. No forecast shown here is investment advice.</p>
  </section>;
}
