const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

async function request(path, signal) {
  const response = await fetch(`${API_BASE}${path}`, { signal, headers:{ Accept:"application/json" } });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `API request failed (${response.status}).`);
  return body;
}

export const researchApi = {
  baseUrl:API_BASE,
  metadata:(signal) => request("/api/v1/metadata", signal),
  models:(signal) => request("/api/v1/models", signal),
  forecasts:({ model, start, end, limit = 120 }, signal) => {
    const query = new URLSearchParams({ model, limit:String(limit) });
    if (start) query.set("start", start);
    if (end) query.set("end", end);
    return request(`/api/v1/forecasts?${query}`, signal);
  },
};
