import Tilt from "react-parallax-tilt";
import { useReducedMotion } from "framer-motion";

export function AliveCard({ children, className = "" }) {
  const reduced = useReducedMotion();
  return <Tilt className={className} tiltEnable={!reduced} tiltMaxAngleX={4} tiltMaxAngleY={4} perspective={1100} scale={1.012} transitionSpeed={900} glareEnable={!reduced} glareMaxOpacity={0.08} glareColor="#74d9ff" glarePosition="all">{children}</Tilt>;
}
