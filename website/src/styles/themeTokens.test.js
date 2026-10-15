import { expect, test } from "vitest";
import { readFileSync } from "node:fs";

test("Warm Tones dark theme keeps distinct semantic colors", () => {
  const styles = readFileSync("src/styles/interactions.css", "utf8");
  const colors = ["#d8a05c", "#e07a5f", "#c98491", "#dda15e", "#d85c4a"];
  colors.forEach((color) => expect(styles).toContain(color));
  expect(new Set(colors).size).toBe(5);
});
