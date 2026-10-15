import { expect, test } from "vitest";
import { readFileSync } from "node:fs";

test("Cyberpunk dark theme keeps distinct semantic colors", () => {
  const styles = readFileSync("src/styles/interactions.css", "utf8");
  const colors = ["#fcee09", "#00f0ff", "#ff2a6d", "#ff9f1c", "#ff3b30"];
  colors.forEach((color) => expect(styles).toContain(color));
  expect(new Set(colors).size).toBe(5);
});
