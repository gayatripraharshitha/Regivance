import { useEffect, useState } from "react";
import { BarChart3, Braces, ChevronRight, Code2, Database, FlaskConical, Info, Moon, ShieldAlert, Sun, UploadCloud } from "lucide-react";
import { AnimatePresence, motion, useScroll, useSpring } from "framer-motion";
import { NavLink, useLocation } from "react-router-dom";
import { useTheme } from "../ui/ThemeProvider";

const links = [
  ["/", "Overview", BarChart3],
  ["/data", "Data & features", Database],
  ["/results", "Results", FlaskConical],
  ["/risk", "Risk analysis", ShieldAlert],
  ["/lab", "Data lab", UploadCloud],
  ["/api", "API", Braces],
  ["/methodology", "Methodology", Info],
  ["/about", "About", Code2],
];

export function AppShell({ children }) {
  const [open, setOpen] = useState(false);
  const { theme, toggleTheme } = useTheme();
  const { pathname } = useLocation();
  const { scrollYProgress } = useScroll();
  const progress = useSpring(scrollYProgress, { stiffness: 150, damping: 30 });
  useEffect(() => {
    const close = (event) => { if (event.key === "Escape") setOpen(false); };
    document.addEventListener("keydown", close);
    document.body.style.overflow = open ? "hidden" : "";
    return () => { document.removeEventListener("keydown", close); document.body.style.overflow = ""; };
  }, [open]);
  return (
    <div className="app-shell">
      <div className="ambient-canvas" aria-hidden="true"><motion.i className="ambient-orb orb-one" animate={{ x:[0,45,-15,0], y:[0,-25,35,0], scale:[1,1.12,.94,1] }} transition={{ duration:18, repeat:Infinity, ease:"easeInOut" }} /><motion.i className="ambient-orb orb-two" animate={{ x:[0,-35,25,0], y:[0,30,-18,0] }} transition={{ duration:22, repeat:Infinity, ease:"easeInOut" }} /><span className="ambient-grid" /></div>
      <motion.div className="scroll-progress" style={{ scaleX: progress }} />
      <header className="topbar">
        <NavLink to="/" className="brand" onClick={() => setOpen(false)} aria-label="Regivance home">
          <span className="brand-mark">R</span><span>REGIVANCE</span>
        </NavLink>
        <span className="current-view">{links.find(([to]) => to === pathname)?.[1] ?? "Research"}</span>
        <motion.button whileTap={{ scale: .93 }} className="theme-toggle" onClick={toggleTheme} aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}>
          {theme === "light" ? <Moon size={17} /> : <Sun size={17} />}<span>{theme === "light" ? "Dark" : "Light"}</span>
        </motion.button>
        <span className="status-pill"><i /> Research build</span>
        <button className={`menu-button ${open ? "open" : ""}`} onClick={() => setOpen((value) => !value)} aria-label="Toggle navigation" aria-expanded={open} aria-controls="site-navigation"><span/><span/><span/></button>
      </header>
      <AnimatePresence>
        {open && <><motion.button className="nav-backdrop" aria-label="Close navigation" onClick={() => setOpen(false)} initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}} transition={{duration:.28}}/><motion.aside id="site-navigation" className="side-nav" role="dialog" aria-modal="true" aria-label="Site navigation" initial={{x:"105%"}} animate={{x:0}} exit={{x:"105%"}} transition={{type:"spring",stiffness:260,damping:29}}>
          <div className="side-nav-head"><div><span className="eyebrow">Navigate research</span><strong>REGIVANCE</strong></div><span>{links.length} views</span></div>
          <nav>{links.map(([to,label,Icon],index) => <motion.div key={to} initial={{opacity:0,x:28}} animate={{opacity:1,x:0}} transition={{delay:.08+index*.045}}><NavLink to={to} onClick={() => setOpen(false)}><span className="nav-icon"><Icon size={18}/></span><span><small>{String(index+1).padStart(2,"0")}</small>{label}</span><ChevronRight size={16}/></NavLink></motion.div>)}</nav>
          <div className="side-nav-foot"><i/><div><strong>Research build</strong><span>Historical development evidence</span></div></div>
        </motion.aside></>}
      </AnimatePresence>
      <main><motion.div key={pathname} initial={{ opacity: 0, y: 16, filter:"blur(3px)" }} animate={{ opacity: 1, y: 0, filter:"blur(0px)" }} transition={{ duration: .42, ease:[.22,1,.36,1] }}>{children}</motion.div></main>
      <footer><span>Regivance · quantitative research system</span><span>Research only — not investment advice</span></footer>
    </div>
  );
}
