import { createContext, useContext, useEffect, useState } from "react";
import { MotionConfig } from "framer-motion";

const ThemeContext = createContext(null);

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem("regivance-theme") === "dark" ? "dark" : "light"; }
    catch { return "light"; }
  });
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    document.querySelector('meta[name="theme-color"]')?.setAttribute("content", theme === "light" ? "#f5f8ff" : "#05070a");
    try { localStorage.setItem("regivance-theme", theme); } catch { /* Storage can be disabled. */ }
  }, [theme]);
  return <ThemeContext.Provider value={{ theme, toggleTheme: () => setTheme(value => value === "light" ? "dark" : "light") }}>
    <MotionConfig reducedMotion="user">{children}</MotionConfig>
  </ThemeContext.Provider>;
}

export const useTheme = () => useContext(ThemeContext);
