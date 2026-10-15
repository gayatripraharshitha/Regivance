import { useState } from "react";
import { BarChart3, Code2, Database, FlaskConical, Info, Menu, Moon, ShieldAlert, Sun, X } from "lucide-react";
import { AnimatePresence, motion, useScroll, useSpring } from "framer-motion";
import { NavLink, useLocation } from "react-router-dom";
import { useTheme } from "../ui/ThemeProvider";

const links = [
  ["/", "Overview", BarChart3],
  ["/data", "Data & features", Database],
  ["/results", "Results", FlaskConical],
  ["/risk", "Risk analysis", ShieldAlert],
  ["/methodology", "Methodology", Info],
  ["/about", "About", Code2],
];

export function AppShell({ children }) {
  const [open, setOpen] = useState(false);
  const { theme, toggleTheme } = useTheme();
  const { pathname } = useLocation();
  const { scrollYProgress } = useScroll();
  const progress = useSpring(scrollYProgress, { stiffness: 150, damping: 30 });
  return (
    <div className="app-shell">
      <motion.div className="scroll-progress" style={{ scaleX: progress }} />
      <header className="topbar">
        <NavLink to="/" className="brand" onClick={() => setOpen(false)} aria-label="Regivance home">
          <span className="brand-mark">R</span><span>REGIVANCE</span>
        </NavLink>
        <button className="menu-button" onClick={() => setOpen((value) => !value)} aria-label="Toggle navigation" aria-expanded={open}>
          {open ? <X /> : <Menu />}
        </button>
        <nav className="desktop-nav">{links.map(([to, label]) => <NavItem key={to} to={to} label={label} />)}</nav>
        <motion.button whileTap={{ scale: .93 }} className="theme-toggle" onClick={toggleTheme} aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}>
          {theme === "light" ? <Moon size={17} /> : <Sun size={17} />}<span>{theme === "light" ? "Dark" : "Light"}</span>
        </motion.button>
        <span className="status-pill"><i /> Research build</span>
      </header>
      <AnimatePresence>
        {open && (
          <motion.nav className="mobile-nav" initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
            {links.map(([to, label, Icon]) => <NavLink key={to} to={to} onClick={() => setOpen(false)}><Icon size={17} />{label}</NavLink>)}
          </motion.nav>
        )}
      </AnimatePresence>
      <main><motion.div key={pathname} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .3 }}>{children}</motion.div></main>
      <footer><span>Regivance · quantitative research system</span><span>Research only — not investment advice</span></footer>
    </div>
  );
}

function NavItem({ to, label }) {
  return <NavLink to={to} className={({ isActive }) => isActive ? "active" : ""}>{label}</NavLink>;
}
