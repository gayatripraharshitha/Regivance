import { motion } from "framer-motion";

export function AliveCard({ children, className = "" }) {
  return <motion.div className={className} whileHover={{ y:-5, scale:1.008 }} transition={{ type:"spring", stiffness:230, damping:20 }}>{children}</motion.div>;
}
