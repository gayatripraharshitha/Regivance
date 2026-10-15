import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useReducedMotion } from "framer-motion";

export function MetricChart({ data, metric }) {
  const reducedMotion = useReducedMotion();
  return (
    <div className="chart-wrap" aria-label={`${metric.toUpperCase()} model comparison`}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 8, right: 20, bottom: 8, left: 25 }}>
          <CartesianGrid stroke="var(--line)" horizontal={false} />
          <XAxis type="number" stroke="var(--muted)" tickLine={false} axisLine={false} tickFormatter={(v) => v.toFixed(metric === "qlike" ? 1 : 3)} />
          <YAxis dataKey="model" type="category" width={145} stroke="var(--muted)" tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
          <Tooltip cursor={{ fill: "var(--line)" }} contentStyle={{ color: "var(--ink)", background: "var(--panel)", border: "1px solid var(--line)", borderRadius: 12 }} formatter={(value) => Number(value).toFixed(6)} />
          <Bar dataKey={metric} fill="var(--green)" radius={[0, 7, 7, 0]} isAnimationActive={!reducedMotion} animationDuration={700} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
