import { useState } from "react";
import { BarChart3, Code2, Database, FlaskConical, Info, Menu, X } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import { NavLink } from "react-router-dom";

const links = [
  ["/", "Overview", BarChart3],
  ["/data", "Data & features", Database],
  ["/results", "Results", FlaskConical],
  ["/methodology", "Methodology", Info],
  ["/about", "About", Code2],
];

export function AppShell({ children }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="app-shell">
      <header className="topbar">
        <NavLink to="/" className="brand" onClick={() => setOpen(false)} aria-label="Regivance home">
          <span className="brand-mark">R</span><span>REGIVANCE</span>
        </NavLink>
        <button className="menu-button" onClick={() => setOpen((value) => !value)} aria-label="Toggle navigation">
          {open ? <X /> : <Menu />}
        </button>
        <nav className="desktop-nav">{links.map(([to, label]) => <NavItem key={to} to={to} label={label} />)}</nav>
        <span className="status-pill"><i /> Research build</span>
      </header>
      <AnimatePresence>
        {open && (
          <motion.nav className="mobile-nav" initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
            {links.map(([to, label, Icon]) => <NavLink key={to} to={to} onClick={() => setOpen(false)}><Icon size={17} />{label}</NavLink>)}
          </motion.nav>
        )}
      </AnimatePresence>
      <main>{children}</main>
      <footer><span>Regivance · quantitative research system</span><span>Research only — not investment advice</span></footer>
    </div>
  );
}

function NavItem({ to, label }) {
  return <NavLink to={to} className={({ isActive }) => isActive ? "active" : ""}>{label}</NavLink>;
}
