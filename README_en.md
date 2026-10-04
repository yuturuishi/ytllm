# ytllm

**语言 / Language：** [简体中文](README.md) | [English](README_en.md)

> Open-source on-device LLM inference platform for Rockchip RK3576 / RK3588 — upload `.rkllm` models from the web, start/stop inference services with one click, and get OpenAI-compatible APIs out of the box. No command line required.

| | |
|---|---|
| Gitee | https://gitee.com/yuturuishi/ytllm |
| GitHub | https://github.com/yuturuishi/ytllm |

**License:** MIT License, free for commercial use. See `LICENSE`.

## Introduction

ytllm is organized as "management plane + inference engines": the management plane is a persistent web service handling model upload and inference-instance lifecycle; each instance runs as a standalone engine process (built on the RKLLM runtime), exposed through a built-in gateway as standard OpenAI APIs.

## Key Features

- **Upload models via web**: upload `.rkllm` / `.rknn` directly from the browser, with chunked resumable transfer for large files — no need to log into the device
- **One-click lifecycle**: start/stop/restart inference services from the web, with automatic port allocation, automatic process reclamation, and parallel multi-instance support
- **OpenAI compatible**: built-in gateway exposes standard endpoints such as `/v1/chat/completions`; works with Dify / Open WebUI / the openai SDK out of the box, with optional API key auth
- **Ready-to-use frontend**: model test console (streaming / image input / copy curl), light & dark themes, mobile responsive
- **Hardware dashboard**: live SoC model, NPU driver version, per-core NPU load, and MPP / RGA / RKNN / RKLLM versions on the home page

## Hardware & OS Requirements

### Hardware

| Item | Requirement |
|------|-------------|
| SoC | RK3576 (2 NPU cores) or RK3588 (3 NPU cores) |
| RAM | ≥ 4GB (for 0.5B~4B w4a16 quantized models; 8GB recommended for 4B) |
| Storage | ≥ 8GB free (model files are typically 0.4GB~4GB each) |
| NPU driver | RKNPU ≥ 0.9.x (verify: `cat /sys/kernel/debug/rknpu/version`) |

### Operating system

- **Inference (on the board)**: any aarch64 Linux; tested on Debian 12 and Ubuntu 22.04. Rockchip's official runtime libraries are **bundled with the repository** (`lib/`, RKLLM v1.3.0 / RKNN v2.3.2, Linux aarch64). Install either way:
  - Copy to the system default path: `sudo cp lib/*.so /usr/local/lib/ && sudo ldconfig`
  - Or point the `RKLLM_LIB` / `RKNN_LIB` environment variables at the files inside `lib/` — no copying needed
- **Management plane (cross-platform)**: the web UI itself is a pure-Python service that runs on x86 / ARM, Linux / Windows / macOS. It can manage inference instances on a remote board (remote OpenAI-compatible services need no NPU at all)

## Quick Start (on the board)

```bash
# 1. Clone (runtime libraries are already inside lib/)
git clone https://gitee.com/yuturuishi/ytllm.git
cd ytllm

# 2. Install runtime libraries (skip if the board already has the same versions)
sudo cp lib/*.so /usr/local/lib/ && sudo ldconfig

# 3. Install dependencies (a venv is recommended)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Start (add --debug=false for production)
python server.py --port=9697 --debug=false

# 5. Open in a browser
# http://<board-ip>:9697
```

That's it. Configuration data is stored in `config.json` automatically (full structure see `config.example.json`), and the `models/` and `logs/` directories are generated on first run.

### Run as a systemd service (recommended)

```ini
# /etc/systemd/system/ytllm.service
[Unit]
Description=ytllm inference platform
After=network.target

[Service]
WorkingDirectory=/opt/ytllm
ExecStart=/opt/ytllm/venv/bin/python server.py --port=9697 --debug=false
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ytllm
```

> Adjust paths to your actual install location. Model loading takes time; an instance shows "running" in the web UI once its model is ready.

## First Run

1. Go to the "Inference Services" page → upload an `.rkllm` model (chunked resumable upload)
2. "Add Service" → Local Engine → pick the model, platform (rk3576/rk3588) and NPU core count
3. Click ▶ on the row, wait for the status to become "running"
4. Verify in the "Model Chat" page, or copy the external API endpoint (see below)

## Where to Get Models

`.rkllm` is Rockchip's quantized model format for the RKLLM stack. Two ways to obtain one:

**Option 1: download pre-converted models (recommended)**

- Official repository: https://github.com/airockchip/rknn-llm (SDK, docs and examples)
- Pre-converted models: the **rkllm_model_zoo** download link (fetch code `rkllm`) in that repository's README, organized in per-platform folders (RK3576 / RK3588)
- Covered models include: Qwen2 / Qwen2.5 / Qwen3 / Qwen3.5, Qwen2-VL / Qwen3-VL (multimodal), DeepSeek-R1-Distill, MiniCPM, InternVL, Gemma, etc.
- **Pick the right file**: filenames carry the platform suffix (e.g. `..._rk3576.rkllm`) — don't mix platforms; multimodal models also need the matching `_vision_*.rknn` file

**Option 2: convert your own**

Use the official rkllm-toolkit on a PC to convert HuggingFace models to `.rkllm` (w8a8 / w4a16 quantization supported); see the rknn-llm repository docs.

After downloading, upload via "Inference Services → Upload Model" — no specific naming required.

## External API (OpenAI compatible)

Running inference services are exposed through the built-in gateway with standard OpenAI endpoints:

```
POST http://<board-ip>:9697/v1/chat/completions
```

```bash
curl http://<board-ip>:9697/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<service-name>","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

- Service selection: put the service name or model ID in the `model` field; use the `X-Service-Id` header only when disambiguating same-named services
- Both streaming (SSE) and non-streaming are fully supported; `/v1/models` and `/v1/completions` work the same way
- **Auth (optional)**: by default no API key is required (LAN use). To restrict access, set an API key in the web UI under "Settings → Security"; requests must then carry `Authorization: Bearer <key>` (or `X-Api-Key`)

## Pages

| Page | Description |
|------|-------------|
| Dashboard `/` | System/CPU/RAM/disk status, Rockchip hardware metrics (SoC model, NPU driver version, per-core load, MPP/RGA/RKNN/RKLLM versions), service overview |
| Inference Services `/models/` | Start/stop/restart/logs/edit/delete services; upload/delete models; local engines and remote services |
| Model Chat `/chat/` | Streaming/non-streaming testing, image input (multimodal models), copy curl |

## Tech Stack & Project Layout

| Layer | Technology |
|------|------|
| Management plane | Python 3.9+ / Tornado 6.4+ / JSON file storage |
| Inference engine | Flask + ctypes calling `librkllmrt.so` (RKLLM) / `librknnrt.so` (RKNN) directly |
| Frontend | Bootstrap 5 + jQuery, responsive (desktop/mobile), light & dark themes |

```
ytllm/
├── server.py                # Management plane (routing/config storage/orchestration/OpenAI gateway)
├── config.example.json      # Sample structure of config.json
├── engine/rkllm_engine.py   # Inference engine worker (spawned per instance)
├── core/
│   ├── engine_manager.py    # Engine process lifecycle (port allocation/start-stop/readiness)
│   ├── model_store.py       # Model repository (chunked upload/scan/delete)
│   └── templates/core/      # Page templates
├── models/                  # Model repository (uploaded .rkllm / .rknn)
├── uploads/                 # Upload staging directory (chunks assembled, then moved into models/)
├── lib/                     # Rockchip runtime libraries (RKLLM 1.3.0 / RKNN 2.3.2, aarch64)
├── logs/                    # Instance logs
├── static/                  # Frontend assets
└── requirements.txt
```

## How It Works

Each local service = one standalone engine process (`engine/rkllm_engine.py`): the management plane allocates a port and spawns the process; if model loading fails, the process exits immediately and the log tail is surfaced to the UI; once loading succeeds, the management plane polls `/health` to confirm readiness. Stopping sends `SIGTERM` for graceful release first, then `SIGKILL` on timeout; deleting a service stops the process first, leaving no orphans.

## FAQ

- **"librkllmrt.so not found" on startup**: run `sudo cp lib/*.so /usr/local/lib/ && sudo ldconfig` (v1.3.0 runtime libraries are bundled in the repo), or set `RKLLM_LIB=/your/path/librkllmrt.so`.
- **Model fails to load with a version mismatch**: the `librkllmrt.so` version must match the model — the bundled v1.3.0 matches models converted with rkllm_model_zoo v1.3.0; for older models, take the matching runtime from the official rknn-llm release.
- **The web UI reports missing runtime components**: there is an environment self-check at the top-right of the page; missing items are listed explicitly (driver/runtime/engine script).
- **How many NPU cores to assign?**: RK3576 has 2 cores, RK3588 has 3; a single model can limit itself via `--rknn-cores`, and sensible allocation improves total throughput when running multiple instances.
- **Can it run on a PC?**: the management plane can (remote-service mode), but local inference requires RK3576/RK3588 + NPU.

## Feedback

Issues and suggestions are welcome on Gitee: https://gitee.com/yuturuishi/ytllm/issues

## License

This project is released under the [MIT License](LICENSE), free for learning, research and commercial use.

## Third-Party Dependencies & Licenses

ytllm's **own code** is released under the MIT License, free for learning, research and commercial use.

Main third-party dependencies and their licenses:

| Dependency | Purpose | License |
|---|---|---|
| Tornado | Management-plane web framework | Apache-2.0 |
| Flask | Engine web framework | BSD-3-Clause |
| OpenAI Python SDK | OpenAI-compatible client | Apache-2.0 |
| requests | Gateway HTTP forwarding | Apache-2.0 |
| psutil | System metrics | BSD-3-Clause |
| NumPy / PyYAML | Engine numerics & config | BSD-3-Clause / MIT |
| opencv-python-headless | Multimodal image processing (optional, multimodal models only) | Apache-2.0 |
| Pillow | Image processing | MIT-CMU |
| Bootstrap 5 / jQuery / Bootstrap Icons / animate.css | Frontend UI (bundled) | MIT |
| Rockchip librkllmrt.so / librknnrt.so | NPU inference runtime (bundled in `lib/`) | Rockchip BSD-style license (redistribution permitted, see `lib/README.txt`) |

- **Data & privacy**: ytllm is a locally deployed tool. Apart from reaching the inference services you configure yourself, it does not send any user data or telemetry to external servers.
- **Models**: `.rkllm` / `.rknn` model files are provided through Rockchip's official channels or converted by the user; their licensing is the model provider's responsibility.
- **Bundled runtime libraries**: taken from Rockchip's official airockchip/rknn-llm v1.3.0 release; copyright belongs to Rockchip Electronics Co., Ltd., redistributed under its license terms.
- **Disclaimer**: users are responsible for the legality of the models and deployment environments they use. ytllm is provided "as is", without warranty of any kind.

## Version

- Current version: **v1.0** (also shown at the top-left of the web UI; single source of truth is `APP_VERSION` in `server.py` — change it there when releasing)
- License: MIT (see [LICENSE](LICENSE))
