import { describe, expect, test } from "vitest";
import { parseCsv, processReturns, toProcessedCsv } from "./dataLab";

describe("data lab transformations", () => {
  test("parses quoted CSV values", () => {
    const parsed = parseCsv('date,"asset,a",b\n20260102,"1.5",-0.5');
    expect(parsed.headers).toEqual(["date", "asset,a", "b"]);
    expect(parsed.records[0]["asset,a"]).toBe("1.5");
  });

  test("matches equal-weighted RMS definitions", () => {
    const parsed = parseCsv("date,a,b\n20260101,1,3\n20260102,2,4\n20260103,3,5\n20260104,4,6");
    const rows = processReturns({ records:parsed.records, dateColumn:"date", returnColumns:["a","b"], unit:"percent", window:2, horizon:2 });
    expect(rows[0].portfolioReturn).toBeCloseTo(.02);
    expect(rows[0].trailingVolatility).toBeNull();
    expect(rows[1].trailingVolatility).toBeCloseTo(Math.sqrt((.02 ** 2 + .03 ** 2) / 2));
    expect(rows[0].forwardTarget).toBeCloseTo(Math.sqrt((.03 ** 2 + .04 ** 2) / 2));
    expect(rows[3].forwardTarget).toBeNull();
  });

  test("rejects duplicate and unsorted dates", () => {
    const duplicate = [{ date:"2026-01-02", r:"1" }, { date:"2026-01-02", r:"2" }];
    expect(() => processReturns({ records:duplicate, dateColumn:"date", returnColumns:["r"] })).toThrow(/Duplicate/);
    const reversed = [{ date:"2026-01-03", r:"1" }, { date:"2026-01-02", r:"2" }];
    expect(() => processReturns({ records:reversed, dateColumn:"date", returnColumns:["r"] })).toThrow(/oldest to newest/);
  });

  test("exports stable processed columns", () => {
    expect(toProcessedCsv([{ date:"2026-01-02", portfolioReturn:.01, trailingVolatility:null, forwardTarget:.02 }])).toContain("2026-01-02,0.01,,0.02");
  });
});
