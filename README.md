# ytllm

**语言 / Language：** [简体中文](README.md) | [English](README_en.md)

> 开源端侧大模型推理平台 —— 适配瑞芯微（Rockchip）RK3576 / RK3588，网页上传 `.rkllm` 模型、一键启停推理服务、直接得到 OpenAI 兼容接口，无需命令行操作。

| | |
|---|---|
| Gitee | https://gitee.com/yuturuishi/ytllm |
| GitHub | https://github.com/yuturuishi/ytllm |

**开源协议：** MIT License，可自由商用。详见 `LICENSE`。

## 简介

ytllm 以「管理面 + 推理引擎」的方式组织：管理面是常驻网页服务，负责模型上传与推理实例的生命周期管理；每个实例由管理面拉起独立引擎进程（基于 RKLLM 运行时），并经内置网关对外暴露标准 OpenAI 接口。

## 核心特性

- **网页传模型**：浏览器直接上传 `.rkllm` / `.rknn`，大文件自动分片续传，不用登录设备
- **一键启停**：网页启动/停止/重启推理服务，自动分配端口、自动回收进程，多实例并行
- **OpenAI 兼容**：内置网关暴露 `/v1/chat/completions` 等标准接口，Dify / Open WebUI / openai SDK 直接接入，可选密钥鉴权
- **开箱即用的前端**：模型对话台（流式/图片输入/curl 复制）、深浅双主题、手机自适应
- **硬件可视化**：首页实时展示 SoC 型号、NPU 驱动版本、各核心负载、MPP / RGA / RKNN / RKLLM 版本

## 适用硬件与系统

### 硬件

| 项目 | 要求 |
|------|------|
| SoC | RK3576（NPU 2 核）或 RK3588（NPU 3 核） |
| 内存 | ≥ 4GB（跑 0.5B~4B 的 w4a16 量化模型；4B 模型建议 8GB） |
| 存储 | ≥ 8GB 可用空间（模型文件通常 0.4GB~4GB/个） |
| NPU 驱动 | RKNPU ≥ 0.9.x（验证：`cat /sys/kernel/debug/rknpu/version`） |

### 系统

- **推理（板端）**：任意 aarch64 Linux，已实测 Debian 12 / Ubuntu 22.04。需要的 Rockchip 官方 runtime 库**已随仓库附带**（`lib/` 目录，RKLLM v1.3.0 / RKNN v2.3.2，Linux aarch64），安装方式二选一：
  - 复制到系统默认路径：`sudo cp lib/*.so /usr/local/lib/ && sudo ldconfig`
  - 或设环境变量 `RKLLM_LIB` / `RKNN_LIB` 指向仓库内 `lib/` 下的文件，无需复制
- **管理面（跨平台）**：管理界面本身是纯 Python 服务，x86 / ARM、Linux / Windows / macOS 均可运行，可远程管理板子上的推理实例（远程接入任意 OpenAI 兼容服务时无需 NPU）

## 快速部署（板端）

```bash
# 1. 拉取代码（runtime 库已在 lib/ 目录内）
git clone https://gitee.com/yuturuishi/ytllm.git
cd ytllm

# 2. 安装 runtime 库（板上已有同版本库可跳过）
sudo cp lib/*.so /usr/local/lib/ && sudo ldconfig

# 3. 安装依赖（建议 venv 隔离）
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. 启动（生产建议加 --debug=false）
python server.py --port=9697 --debug=false

# 5. 浏览器访问
# http://<板子IP>:9697
```

就这么多。配置数据自动落在 `config.json`（完整结构见 `config.example.json`），`models/`、`logs/` 目录自动生成。

### systemd 常驻（推荐）

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

> 路径按实际安装位置修改。模型加载需要时间，服务启动后网页上实例状态变为「运行中」即模型已就绪。

## 首次使用

1. 「推理服务」页 → 上传 `.rkllm` 模型（大文件自动分片续传）
2. 「添加服务」→ 本地引擎 → 选择模型、平台（rk3576/rk3588）、NPU 核心数
3. 点该行 ▶ 启动，等待状态变「运行中」
4. 「模型对话」页选服务发一条消息验证；或直接复制对外接口（见下）

## 模型从哪里来

`.rkllm` 是瑞芯微 RKLLM 格式的量化模型，两种获取方式：

**方式一：直接下载预转换模型（推荐）**

- 官方仓库：https://github.com/airockchip/rknn-llm （SDK、文档、示例都在这里）
- 预转换模型：仓库 README 中的 **rkllm_model_zoo** 网盘链接（提取码 `rkllm`），按平台分目录（RK3576 / RK3588）下载即可
- 已覆盖：Qwen2 / Qwen2.5 / Qwen3 / Qwen3.5、Qwen2-VL / Qwen3-VL（多模态）、DeepSeek-R1-Distill、MiniCPM、InternVL、Gemma 等
- **注意选对文件**：文件名里带平台名（如 `..._rk3576.rkllm`），不要下错平台；多模态模型还需同目录下的 `_vision_*.rknn` 视觉模型

**方式二：自己转换**

在 PC 上用官方 rkllm-toolkit 把 HuggingFace 模型转成 `.rkllm`（量化格式支持 w8a8 / w4a16 等），转换方法见 rknn-llm 仓库文档。

下载后回到网页「推理服务 → 上传模型」即可，文件名/目录无需特定命名。

## 对外调用（OpenAI 兼容）

启动的推理服务通过内置网关对外暴露标准 OpenAI 接口：

```
POST http://<板子IP>:9697/v1/chat/completions
```

```bash
curl http://<板子IP>:9697/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<服务名>","messages":[{"role":"user","content":"你好"}],"stream":true}'
```

- 服务定位：`model` 字段填服务名或模型 ID 即可；同名服务需精确指定时才用请求头 `X-Service-Id`
- 完整支持流式（SSE）与非流式；`/v1/models`、`/v1/completions` 同样可用
- **鉴权（可选）**：默认不设密钥、内网直接调用；如需管控，在网页「设置 → 安全」设置 API 密钥，之后请求需带 `Authorization: Bearer <密钥>`（或 `X-Api-Key`）

## 页面说明

| 页面 | 说明 |
|------|------|
| 控制中心 `/` | 系统/CPU/内存/磁盘状态、Rockchip 硬件指标（SoC 型号、NPU 驱动版本、各核心实时负载、MPP/RGA/RKNN/RKLLM 版本）、服务概览 |
| 推理服务 `/models/` | 服务启停/重启/日志/编辑/删除；模型仓库上传/删除；本地引擎与远程接入 |
| 模型对话 `/chat/` | 流式/非流式调试，支持图片输入（多模态模型），一键复制 curl |

## 技术栈与项目结构

| 层级 | 技术 |
|------|------|
| 管理面 | Python 3.9+ / Tornado 6.4+ / JSON 文件存储 |
| 推理引擎 | Flask + ctypes 直调 `librkllmrt.so`（RKLLM）/ `librknnrt.so`（RKNN） |
| 前端 | Bootstrap 5 + jQuery，响应式（桌面/手机），深浅双主题 |

```
ytllm/
├── server.py                # 管理面主服务（路由/配置存储/实例编排/OpenAI 网关）
├── config.example.json      # config.json 结构样例
├── engine/rkllm_engine.py   # 推理引擎 worker（管理面按实例拉起）
├── core/
│   ├── engine_manager.py    # 引擎进程生命周期（端口分配/启停/就绪判定）
│   ├── model_store.py       # 模型仓库（分片上传/扫描/删除）
│   └── templates/core/      # 页面模板
├── models/                  # 模型仓库（上传的 .rkllm / .rknn）
├── uploads/                 # 上传临时目录（分片汇总后移入 models/）
├── lib/                     # Rockchip runtime 库（RKLLM 1.3.0 / RKNN 2.3.2，aarch64）
├── logs/                    # 实例运行日志
├── static/                  # 前端静态资源
└── requirements.txt
```

## 运行原理

每个本地服务 = 一个独立引擎进程（`engine/rkllm_engine.py`）：管理面自动分配端口并拉起进程，模型加载失败立即退出并把日志尾部原因返回前端；加载成功后管理面轮询 `/health` 确认就绪。停止先 `SIGTERM` 优雅释放、超时 `SIGKILL`；删除服务先停进程，不留孤儿进程。

## 常见问题

- **启动报找不到 `librkllmrt.so`**：执行 `sudo cp lib/*.so /usr/local/lib/ && sudo ldconfig`（仓库已附带 v1.3.0 运行时库），或设 `RKLLM_LIB=/你的路径/librkllmrt.so`。
- **模型加载失败、日志出现版本不匹配**：`librkllmrt.so` 与模型版本必须匹配——仓库附带 v1.3.0，对应 rkllm_model_zoo v1.3.0 转出的模型；老版本模型请从 rknn-llm 官方发布包取对应版本运行时。
- **网页提示运行环境缺失**：页面右上角有环境自检，缺失项会明确指出（驱动/runtime/引擎脚本）。
- **NPU 核心数怎么选**：RK3576 有 2 核、RK3588 有 3 核；单模型可用 `--rknn-cores` 指定占用核数，多实例并行时合理分配可提升总吞吐。
- **PC 上能跑吗**：管理面可以（远程接入模式），但本地推理必须 RK3576/RK3588 + NPU 环境。

## 问题反馈

欢迎通过 Gitee Issue 提交问题与建议：https://gitee.com/yuturuishi/ytllm/issues

## 开源协议

本项目基于 [MIT License](LICENSE) 开源，可免费用于学习、研究与商业用途。

## 第三方依赖与许可说明

ytllm 的**自有代码**以 MIT License 发布，可自由用于学习、研究与商业用途。

本项目涉及的主要第三方依赖及其许可：

| 依赖 | 用途 | 许可 |
|---|---|---|
| Tornado | 管理面 Web 框架 | Apache-2.0 |
| Flask | 推理引擎 Web 框架 | BSD-3-Clause |
| OpenAI Python SDK | OpenAI 兼容客户端 | Apache-2.0 |
| requests | 网关 HTTP 转发 | Apache-2.0 |
| psutil | 系统状态采集 | BSD-3-Clause |
| NumPy / PyYAML | 引擎数值处理与配置 | BSD-3-Clause / MIT |
| opencv-python-headless | 多模态图片处理（可选，仅多模态模型需要） | Apache-2.0 |
| Pillow | 图片处理 | MIT-CMU |
| Bootstrap 5 / jQuery / Bootstrap Icons / animate.css | 前端 UI（随仓库分发） | MIT |
| Rockchip librkllmrt.so / librknnrt.so | NPU 推理运行时（`lib/` 目录随仓库分发） | Rockchip BSD 风格许可（允许再分发，声明见 `lib/README.txt`） |

- **数据与隐私**：ytllm 为本地部署工具，除访问使用者自行配置的推理服务外，不向任何外部服务器回传用户数据或运行信息。
- **模型**：`.rkllm` / `.rknn` 模型文件由瑞芯微官方渠道提供或使用者自行转换，其许可由模型提供方负责。
- **随仓库分发的 runtime 库**：来自瑞芯微官方 airockchip/rknn-llm v1.3.0 发布包，版权归 Rockchip Electronics Co., Ltd. 所有，按其许可条款再分发。
- **免责**：使用者应自行确保所用模型与部署环境的合法性。ytllm 按「现状」提供，不作任何担保。

## 版本

- 当前版本：**v1.0**（页面左上角同步显示；唯一定义处为 `server.py` 的 `APP_VERSION`，升级时改这一处即可）
- License：MIT（见 [LICENSE](LICENSE)）
