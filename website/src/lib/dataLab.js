const MAX_ROWS = 100_000;

export function parseCsv(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  const source = String(text).replace(/^\uFEFF/, "");
  for (let index = 0; index < source.length; index += 1) {
    const char = source[index];
    if (char === '"') {
      if (quoted && source[index + 1] === '"') { field += '"'; index += 1; }
      else quoted = !quoted;
    } else if (char === "," && !quoted) { row.push(field.trim()); field = ""; }
    else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && source[index + 1] === "\n") index += 1;
      row.push(field.trim());
      if (row.some(Boolean)) rows.push(row);
      row = []; field = "";
    } else field += char;
  }
  row.push(field.trim());
  if (row.some(Boolean)) rows.push(row);
  if (quoted) throw new Error("The CSV contains an unclosed quoted field.");
  if (rows.length < 2) throw new Error("The CSV needs a header and at least one data row.");
  const headers = rows[0];
  if (headers.some((value) => !value) || new Set(headers).size !== headers.length) throw new Error("Column names must be present and unique.");
  if (rows.length - 1 > MAX_ROWS) throw new Error(`The browser demo accepts at most ${MAX_ROWS.toLocaleString()} rows.`);
  const records = rows.slice(1).map((values, index) => {
    if (values.length !== headers.length) throw new Error(`Row ${index + 2} has ${values.length} values; expected ${headers.length}.`);
    return Object.fromEntries(headers.map((header, position) => [header, values[position]]));
  });
  return { headers, records };
}

function parseDate(value) {
  const compact = /^\d{8}$/.test(value) ? `${value.slice(0,4)}-${value.slice(4,6)}-${value.slice(6,8)}` : value;
  const timestamp = Date.parse(compact);
  return Number.isFinite(timestamp) ? timestamp : NaN;
}

export function processReturns({ records, dateColumn, returnColumns, unit = "percent", window = 21, horizon = 5 }) {
  if (!dateColumn || !returnColumns.length) throw new Error("Choose one date column and at least one return column.");
  if (returnColumns.includes(dateColumn)) throw new Error("The date column cannot also be a return column.");
  const scale = unit === "percent" ? 0.01 : 1;
  const seen = new Set();
  const clean = records.map((record, rowIndex) => {
    const timestamp = parseDate(record[dateColumn]);
    if (!Number.isFinite(timestamp)) throw new Error(`Row ${rowIndex + 2} has an invalid date.`);
    const date = new Date(timestamp).toISOString().slice(0, 10);
    if (seen.has(date)) throw new Error(`Duplicate date: ${date}.`);
    seen.add(date);
    const values = returnColumns.map((column) => Number(record[column]) * scale);
    if (values.some((value) => !Number.isFinite(value))) throw new Error(`Row ${rowIndex + 2} has a missing or non-numeric return.`);
    if (values.some((value) => Math.abs(value) > 1)) throw new Error(`Row ${rowIndex + 2} contains a return outside -100% to 100%. Check the selected unit.`);
    return { date, timestamp, portfolioReturn: values.reduce((sum, value) => sum + value, 0) / values.length };
  });
  for (let index = 1; index < clean.length; index += 1) if (clean[index].timestamp <= clean[index - 1].timestamp) throw new Error("Dates must be sorted from oldest to newest.");
  return clean.map((row, index) => {
    const trailing = clean.slice(Math.max(0, index - window + 1), index + 1);
    const future = clean.slice(index + 1, index + 1 + horizon);
    const rms = (items) => Math.sqrt(items.reduce((sum, item) => sum + item.portfolioReturn ** 2, 0) / items.length);
    return {
      date: row.date,
      portfolioReturn: row.portfolioReturn,
      trailingVolatility: trailing.length === window ? rms(trailing) : null,
      forwardTarget: future.length === horizon ? rms(future) : null,
    };
  });
}

export function toProcessedCsv(rows) {
  const header = "date,portfolio_return,portfolio_volatility_21d,forward_volatility_5d";
  const values = rows.map((row) => [row.date, row.portfolioReturn, row.trailingVolatility ?? "", row.forwardTarget ?? ""].join(","));
  return [header, ...values].join("\n");
}
