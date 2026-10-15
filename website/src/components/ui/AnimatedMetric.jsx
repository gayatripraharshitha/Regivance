import CountUp from "react-countup";
import { useInView, useReducedMotion } from "framer-motion";
import { useRef } from "react";

export function AnimatedMetric({ value }) {
  const ref = useRef(null);
  const visible = useInView(ref, { once:true, margin:"-40px" });
  const reduced = useReducedMotion();
  const numeric = Number(String(value).replaceAll(",", ""));
  if (!Number.isFinite(numeric)) return <strong ref={ref}>{value}</strong>;
  return <strong ref={ref} aria-label={String(value)}>{visible ? <CountUp aria-hidden="true" end={numeric} duration={reduced ? 0 : 1.25} separator="," preserveValue /> : "0"}</strong>;
}
