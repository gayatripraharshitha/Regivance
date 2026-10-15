import { expect, test } from "vitest";
import { readFileSync } from "node:fs";

test("Midnight Quant dark theme keeps distinct semantic colors", () => {
  const styles = readFileSync("src/styles/interactions.css", "utf8");
  const colors = ["#4d8dff", "#4ed8e8", "#9b87f5", "#f2b84b", "#f26b6b"];
  colors.forEach((color) => expect(styles).toContain(color));
  expect(new Set(colors).size).toBe(5);
});
