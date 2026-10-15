import { expect, test } from "vitest";
import { readFileSync } from "node:fs";

test("Warm Tones dark theme keeps distinct semantic colors", () => {
  const styles = readFileSync("src/styles/interactions.css", "utf8");
  const colors = ["#d8a05c", "#e07a5f", "#c98491", "#dda15e", "#d85c4a"];
  colors.forEach((color) => expect(styles).toContain(color));
  expect(new Set(colors).size).toBe(5);
});

test("light and dark themes share the same typography", () => {
  const baseStyles = readFileSync("src/styles/index.css", "utf8");
  const themeStyles = readFileSync("src/styles/interactions.css", "utf8");
  expect(baseStyles).not.toContain("Cormorant Garamond");
  expect(themeStyles).not.toMatch(/:root:not\(\[data-theme="light"\]\)\s+(h1|h2|\.eyebrow)/);
});

test("light navigation has distinct drawer and active-item contrast", () => {
  const styles = readFileSync("src/styles/interactions.css", "utf8");
  expect(styles).toContain(':root[data-theme="light"] .side-nav {');
  expect(styles).toContain(':root[data-theme="light"] .side-nav nav a.active {');
  expect(styles).toContain(':root[data-theme="light"] .side-nav nav a.active .nav-icon {');
});
