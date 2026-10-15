import { cleanup, fireEvent, render, screen, waitForElementToBeRemoved, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import App from "./App";
import { beforeEach, expect, test, vi } from "vitest";

vi.mock("./components/charts/MetricChart", () => ({ MetricChart: () => <div>Comparison chart</div> }));

beforeEach(() => { cleanup(); localStorage.clear(); });

test("renders the research question and verified test count", async () => {
  render(<MemoryRouter><App /></MemoryRouter>);
  expect(await screen.findByText(/Markets change shape/i)).toBeInTheDocument();
  expect(screen.getByLabelText("125")).toBeInTheDocument();
});

test("animated menu opens the complete side navigation and closes on selection", async () => {
  render(<MemoryRouter><App /></MemoryRouter>);
  fireEvent.click(screen.getByRole("button", { name:"Toggle navigation" }));
  const navigation = screen.getByRole("dialog", { name:"Site navigation" });
  expect(within(navigation).getByText("8 views")).toBeInTheDocument();
  fireEvent.click(within(navigation).getByRole("link", { name:/Risk analysis/ }));
  await waitForElementToBeRemoved(() => screen.queryByRole("dialog", { name:"Site navigation" }));
});

test("theme defaults to light, switches and persists across remounts", async () => {
  const view = render(<MemoryRouter><App /></MemoryRouter>);
  expect(document.documentElement.dataset.theme).toBe("light");
  fireEvent.click(screen.getByRole("button", { name: "Switch to dark theme" }));
  expect(document.documentElement.dataset.theme).toBe("dark");
  expect(localStorage.getItem("regivance-theme")).toBe("dark");
  view.unmount();
  render(<MemoryRouter><App /></MemoryRouter>);
  expect(screen.getByRole("button", { name: "Switch to light theme" })).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Switch to light theme" }));
  expect(document.documentElement.dataset.theme).toBe("light");
});

test("results combine filters, recover from no matches and inspect a model", async () => {
  render(<MemoryRouter initialEntries={["/results"]}><App /></MemoryRouter>);
  fireEvent.change(await screen.findByLabelText("Model family"), { target: { value: "Neural" } });
  expect(screen.getByRole("status")).toHaveTextContent("5 of 11");
  fireEvent.change(screen.getByLabelText("Find a model"), { target: { value: "no-such-model" } });
  expect(screen.getByText("No matching models")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Reset filters" }));
  expect(screen.getByRole("status")).toHaveTextContent("11 of 11");
  fireEvent.click(screen.getByRole("button", { name: "MAE", exact: true }));
  expect(screen.getByRole("status")).toHaveTextContent("ranked by MAE");
  fireEvent.click(screen.getByRole("button", { name: "EWMA", exact: true }));
  expect(within(screen.getByRole("region", { name: "Selected model details" })).getByRole("heading")).toHaveTextContent("EWMA");
  expect(screen.getByText("Stress transparency")).toBeInTheDocument();
  expect(screen.getByText(/88.7% of 2020 shock origins/)).toBeInTheDocument();
});

test("risk analysis exposes verified stress, VaR and robustness evidence", async () => {
  render(<MemoryRouter initialEntries={["/risk"]}><App /></MemoryRouter>);
  expect(await screen.findByRole("heading", { name: /Where forecasts break/i })).toBeInTheDocument();
  expect(screen.getByText("88.7%", { exact: true })).toBeInTheDocument();
  expect(screen.getByRole("heading", { name: /Does predicted risk/i })).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "99% VaR" }));
  expect(screen.getAllByText("Calibration mismatch")).toHaveLength(2);
  expect(screen.getByText(/Prefer GJR-GARCH-t today/i)).toBeInTheDocument();
  expect(screen.getByRole("link", { name: /Download seed data/i })).toHaveAttribute("href", expect.stringContaining("v2_lstm_seed_sensitivity.csv"));
});

test("data lab loads a sample, validates it and produces downloadable results", async () => {
  render(<MemoryRouter initialEntries={["/lab"]}><App /></MemoryRouter>);
  expect(await screen.findByRole("heading", { name: /Run your own data/i })).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: /load the sample/i }));
  expect(screen.getByText(/30 rows/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Validate and calculate" }));
  expect(screen.getByRole("heading", { name: "Your transformed volatility series" })).toBeInTheDocument();
  expect(screen.getByText("5", { exact: true })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /Download processed CSV/i })).toBeInTheDocument();
});

test("API explorer connects and loads reviewed forecast history", async () => {
  const responses = [
    { schemaVersion:1, scope:"historical_development", observations:2520, startDate:"2016-08-15", endDate:"2026-08-31", models:["GJR-GARCH-Student-t"] },
    { items:[{ model:"GJR-GARCH-Student-t", mae:.007, rmse:.011, qlike:.419 }] },
    { model:"GJR-GARCH-Student-t", total:1, count:1, items:[{ date:"2020-03-16", actual:.08, forecast:.04, fold:43, regime:"higher" }] },
  ];
  vi.stubGlobal("fetch", vi.fn(() => Promise.resolve({ ok:true, json:() => Promise.resolve(responses.shift()) })));
  render(<MemoryRouter initialEntries={["/api"]}><App /></MemoryRouter>);
  expect(await screen.findByText("API connected")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name:"Load forecast history" }));
  expect(await screen.findByText(/1 displayed from 1/)).toBeInTheDocument();
  vi.unstubAllGlobals();
});
