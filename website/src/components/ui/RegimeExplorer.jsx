import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

const regimes = [
  { name: "Lower", text: "The cluster with the lowest portfolio-volatility center in this training window. It describes relatively quieter observations, without guaranteeing low risk tomorrow." },
  { name: "Middle", text: "The cluster whose portfolio-volatility center falls between the other two. Correlation and dispersion also influence how observations are assigned." },
  { name: "Higher", text: "The cluster with the highest portfolio-volatility center in the training window. Larger recent movements do not mean every industry moves in the same direction." },
];

export function RegimeExplorer() {
  const [active, setActive] = useState(0);
  return <section className="regime-explorer"><span className="eyebrow">Explore the regimes</span><h2>Three labels. A changing market context.</h2>
    <p>Choose a label to understand its meaning. This explains the ranking system; it is not a live market classification.</p>
    <div className="regime-tabs" role="group" aria-label="Explore regime labels">{regimes.map((regime,index)=><button key={regime.name} aria-pressed={active===index} onClick={()=>setActive(index)}>{active===index && <motion.span layoutId="regime-highlight" className="regime-highlight" transition={{type:"spring",stiffness:350,damping:30}}/>}<span>{regime.name}</span></button>)}</div>
    <AnimatePresence mode="wait"><motion.div key={active} initial={{opacity:0,y:8}} animate={{opacity:1,y:0}} exit={{opacity:0,y:-8}} transition={{duration:.18}} className="regime-description"><h3>{regimes[active].name} relative volatility</h3><p>{regimes[active].text}</p></motion.div></AnimatePresence>
    <small>Ranks are recomputed from training-only cluster centers in each fold.</small>
  </section>;
}
