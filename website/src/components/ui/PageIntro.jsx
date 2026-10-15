import { motion } from "framer-motion";

export function PageIntro({ eyebrow, title, children }) {
  return (
    <motion.header className="page-intro" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .55 }}>
      <span className="eyebrow">{eyebrow}</span>
      <h1>{title}</h1>
      <p>{children}</p>
    </motion.header>
  );
}
