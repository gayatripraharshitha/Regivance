import { expect, test } from "vitest";
import { readFileSync } from "node:fs";

test("Obsidian Ember dark theme keeps distinct semantic colors", () => {
  const styles = readFileSync("src/styles/interactions.css", "utf8");
  const colors = ["#d6a85f", "#c77845", "#a66f7d", "#dfa85a", "#d46157"];
  colors.forEach((color) => expect(styles).toContain(color));
  expect(new Set(colors).size).toBe(5);
});
