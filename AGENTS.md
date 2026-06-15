## Playwright
- เวลาใช้ Playwright screenshot ให้ใช้ parameter `fullPage: true` เสมอ เพื่อให้ได้รูปเต็มจอ
- viewport size ใช้ width 1400, height 900

## Download Progress Tracking
- `POST /api/download_model` returns `{status, download_id}` immediately, spawns daemon thread
- `GET /api/download_progress?download_id=X` returns `{total, received, status, message?, error?}`
- CivitAI: uses HEAD request for Content-Length, tracks bytes during streaming
- Other (requests fallback): tracks bytes using iter_content; wget/curl tracks final file size
- Frontend polls every 500ms, updates progress bar width + friendly byte display
- On `status: "done"` → auto-close modal + toast + refresh file list
- On `status: "error"` → show error, re-enable download button
