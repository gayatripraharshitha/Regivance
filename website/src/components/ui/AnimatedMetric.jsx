import { motion } from "framer-motion";

export function AnimatedMetric({ value }) {
  return <motion.strong aria-label={String(value)} initial={{ opacity:0, scale:.88 }} whileInView={{ opacity:1, scale:1 }} viewport={{ once:true, amount:.8 }} transition={{ type:"spring", stiffness:170, damping:18 }}>{value}</motion.strong>;
}
