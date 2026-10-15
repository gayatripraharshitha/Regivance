import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import { ThemeProvider } from "./components/ui/ThemeProvider";

const Overview = lazy(() => import("./pages/Overview").then((module) => ({ default: module.Overview })));
const DataFeatures = lazy(() => import("./pages/DataFeatures").then((module) => ({ default: module.DataFeatures })));
const ModelResults = lazy(() => import("./pages/ModelResults").then((module) => ({ default: module.ModelResults })));
const Methodology = lazy(() => import("./pages/Methodology").then((module) => ({ default: module.Methodology })));
const About = lazy(() => import("./pages/About").then((module) => ({ default: module.About })));

export default function App() {
  return (
    <ThemeProvider><AppShell>
      <Suspense fallback={<div className="route-loading">Loading research view…</div>}>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/data" element={<DataFeatures />} />
          <Route path="/results" element={<ModelResults />} />
          <Route path="/methodology" element={<Methodology />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </AppShell></ThemeProvider>
  );
}
