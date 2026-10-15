import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import App from "./App";
import { expect, test } from "vitest";

test("renders the research question and verified test count", async () => {
  render(<MemoryRouter><App /></MemoryRouter>);
  expect(await screen.findByText(/Markets change shape/i)).toBeInTheDocument();
  expect(screen.getByText("121")).toBeInTheDocument();
});
