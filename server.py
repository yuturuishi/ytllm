#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ytllm 管理面主服务
==================

ytllm 是面向瑞芯微 RK3576 / RK3588 的端侧大模型推理平台。本文件是平台的唯一入口：

  - 网页管理界面（控制中心 / 推理服务 / 模型对话，Tornado 模板渲染）
  - 模型仓库（.rkllm / .rknn 分片上传、扫描、删除）
  - 推理实例编排（通过 core.engine_manager 为每个本地服务拉起独立的
    engine/rkllm_engine.py 进程，一个模型一个进程一个端口）
  - OpenAI 兼容网关（/v1/chat/completions、/v1/completions、/v1/models，
    可选 Bearer 密钥鉴权，支持流式 SSE 转发）

运行::

    python server.py --port=9697 --debug=false

数据全部落在本地：config.json 存服务、界面配置与网关密钥，
models/ 存模型，logs/ 存实例日志。

开源仓库：https://gitee.com/yuturuishi/ytllm
"""
import tornado.ioloop
import tornado.web
import tornado.template
import tornado.websocket
import tornado.httpserver
import tornado.concurrent
import tornado.gen
import tornado.options
import tornado.httpclient
import tornado.httputil
import tornado.escape
import os
import json
import queue
import uuid
import time
import datetime
import threading
import base64
import re
import requests
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

from core.model_store import ModelStore, ModelStoreError, human_size
from core.engine_manager import EngineManager, EngineError

tornado.options.define('port', default=9697, type=int)
tornado.options.define('debug', default=True, type=bool)

BASE_DIR = Path(__file__).resolve().parent

APP_VERSION = 'v1.0'  # 全站唯一版本号来源：页面左上角、/api 版本检测、README 均以此为准
STATIC_DIR = BASE_DIR / 'static'
TEMPLATE_DIR = BASE_DIR / 'core' / 'templates'
UPLOADS_DIR = BASE_DIR / 'uploads'
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'
DATABASE_FILE = BASE_DIR / 'ytllm.sqlite3'

executor = ThreadPoolExecutor(max_workers=10)

class ServiceStore:
    """推理服务注册表：数据存 config.json 的 services 字段，不再使用 SQLite。"""
    _instance = None
    _lock = threading.Lock()

    _DEFAULTS = {
        'name': '', 'base_url': '', 'api_key': 'sk-ytllm', 'model_id': '',
        'is_active': 1, 'last_checked_at': None, 'last_status': 'unknown',
        'description': '', 'engine_type': '', 'max_tokens': 4096,
        'service_type': 'remote', 'model_path': '', 'vision_model_path': '',
        'platform': 'rk3576', 'rknn_cores': 2, 'port': None, 'pid': None,
        'status': 'stopped', 'started_at': None, 'last_error': '',
        'params': {}, 'created_at': '', 'updated_at': '',
    }

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self):
        self.io_lock = threading.Lock()
        self._import_sqlite()

    def _import_sqlite(self):
        """老版本 ytllm.sqlite3 一次性导入 config.json，成功后删除旧库文件。"""
        if not DATABASE_FILE.exists():
            return
        try:
            import sqlite3
            conn = sqlite3.connect(str(DATABASE_FILE))
            conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute("SELECT * FROM openai_service")]
            conn.close()
            for row in rows:
                self._services().append(self._fill(row))
            self._save()
            DATABASE_FILE.unlink()
            print(f"已从 ytllm.sqlite3 导入 {len(rows)} 个服务到 config.json，旧库文件已删除")
        except Exception as e:
            print(f"ytllm.sqlite3 导入失败（原文件保留）：{e}")

    def _services(self):
        return config_manager.config.setdefault('services', [])

    def _save(self):
        config_manager._save_config()

    def _fill(self, data):
        svc = dict(data)
        for key, val in self._DEFAULTS.items():
            svc.setdefault(key, val)
        if isinstance(svc.get('params'), str):
            try:
                svc['params'] = json.loads(svc['params'] or '{}')
            except ValueError:
                svc['params'] = {}
        return svc

    def list(self, active_only=False, local_only=False):
        with self.io_lock:
            rows = self._services()
            if active_only:
                rows = [r for r in rows if r.get('is_active')]
            if local_only:
                rows = [r for r in rows if (r.get('service_type') or 'remote') == 'local']
            return [dict(r) for r in sorted(rows, key=lambda r: r.get('created_at') or '', reverse=True)]

    def get(self, pk):
        with self.io_lock:
            for r in self._services():
                if str(r.get('id')) == str(pk):
                    return dict(r)
        return None

    def get_by_ref(self, ref):
        """按 id / name / model_id 任一匹配定位服务（网关入口用）。"""
        if not ref:
            return None
        ref = str(ref).strip()
        for key in ('id', 'name', 'model_id'):
            for r in self._services():
                if str(r.get(key) or '') == ref:
                    return dict(r)
        return None

    def allocated_ports(self):
        return [r['port'] for r in self._services() if r.get('port') is not None]

    def create(self, **fields):
        with self.io_lock:
            row = self._fill(fields)
            self._services().append(row)
            self._save()
            return dict(row)

    def update(self, pk, **fields):
        with self.io_lock:
            for r in self._services():
                if str(r.get('id')) == str(pk):
                    for k, v in fields.items():
                        if k in self._DEFAULTS:
                            r[k] = v
                    return dict(r)
        return None

    def delete(self, pk):
        with self.io_lock:
            services = self._services()
            for i, r in enumerate(services):
                if str(r.get('id')) == str(pk):
                    row = dict(r)
                    del services[i]
                    self._save()
                    return row
        return None

    def counts(self):
        rows = self._services()
        return {
            'total': len(rows),
            'local': sum(1 for r in rows if (r.get('service_type') or 'remote') == 'local'),
            'running': sum(1 for r in rows if r.get('status') == 'running'),
        }

model_store = ModelStore(MODELS_DIR)
engine_manager = EngineManager(BASE_DIR, LOGS_DIR)

# 引擎默认推理参数（新建本地服务时写入 params 字段）
DEFAULT_ENGINE_PARAMS = {
    'max_context': 4096,
    'max_tokens': 512,
    'top_k': 1,
    'top_p': 0.9,
    'temperature': 0.8,
    'repeat_penalty': 1.1,
}


def _probe_lib_version(candidates, marker):
    """在候选动态库里找版本标记串，返回 (路径, 版本串)。"""
    for path in candidates:
        p = Path(path)
        if not p.is_file():
            continue
        try:
            data = p.read_bytes()
        except OSError:
            continue
        idx = data.find(marker)
        if idx >= 0:
            chunk = data[idx:idx + 160].split(b'\x00')[0]
            return str(p), chunk.decode('utf-8', 'replace').strip()
        return str(p), ''
    return '', ''


_runtime_cache = None


def read_engine_runtime(use_cache=True):
    """引擎运行环境自检：平台 / 架构 / NPU / RKLLM 运行时。

    非 Rockchip 设备（例如在 PC 上做开发调试）会明确标出缺失项，
    让用户一眼看出"为什么启动不了"，而不是对着一个模糊的失败发呆。
    """
    global _runtime_cache
    if use_cache and _runtime_cache is not None:
        return _runtime_cache

    import platform as _platform

    info = {
        'os': _platform.system(),
        'arch': _platform.machine(),
        'python': _platform.python_version(),
        'is_rockchip': False,
        'soc': '',
        'npu_driver': '',
        'rkllm_lib': '',
        'rkllm_version': '',
        'rknn_version': '',
        'libraries_ready': False,
        'engine_script_ready': engine_manager.script.is_file(),
    }

    try:
        with open('/proc/device-tree/compatible', 'rb') as f:
            info['soc'] = f.read().decode('utf-8', 'replace').replace('\x00', ', ').strip(', ')
        info['is_rockchip'] = 'rockchip' in info['soc'].lower()
    except OSError:
        pass

    try:
        with open('/sys/kernel/debug/rknpu/version') as f:
            info['npu_driver'] = f.read().strip()
    except OSError:
        pass

    lib, version = _probe_lib_version(
        ('/usr/local/lib/librkllmrt.so', '/usr/lib/librkllmrt.so'), b'RKLLM SDK')
    info['rkllm_lib'], info['rkllm_version'] = lib, version

    _, version = _probe_lib_version(
        ('/usr/local/lib/librknnrt.so', '/usr/lib/librknnrt.so'), b'librknnrt version')
    info['rknn_version'] = version

    info['libraries_ready'] = bool(info['rkllm_lib']) and info['engine_script_ready']
    _runtime_cache = info
    return info


def _run_cmd(cmd, timeout=5):
    """在板端本地执行命令（server 本身跑在 RK 板上），拿文本，失败返回空。"""
    try:
        return subprocess.check_output(cmd, shell=True, timeout=timeout,
                                      stderr=subprocess.DEVNULL).decode('utf-8', 'replace')
    except Exception:
        return ''


_HW_VER_CACHE = {}
_HW_VER_CACHE_TS = 0.0


def _detect_rk_versions():
    """Rockchip 静态版本信息（SoC / NPU 驱动 / RKNN / RKLLM / MPP / RGA）。

    运行期不变，缓存 60s，避免首页每 5s 轮询都去 dpkg/pkg-config。
    非 Rockchip 设备（PC 调试）各字段留空，前端照常渲染「未检测到」。
    """
    global _HW_VER_CACHE, _HW_VER_CACHE_TS
    now = time.time()
    if _HW_VER_CACHE and now - _HW_VER_CACHE_TS < 60:
        return _HW_VER_CACHE

    info = {
        'soc': '', 'soc_raw': '', 'is_rockchip': False,
        'npu_driver': '', 'rknn_version': '', 'rknn_lib': '',
        'rkllm_version': '', 'rkllm_lib': '', 'mpp_version': '', 'rga_version': '',
    }

    # SoC 型号：从 device-tree compatible 里捞 rkXXXX
    try:
        with open('/proc/device-tree/compatible', 'rb') as f:
            soc_raw = f.read().decode('utf-8', 'replace').replace('\x00', ' ').strip()
        info['soc_raw'] = soc_raw
        m = re.search(r'rk\s*(\d+)', soc_raw, re.IGNORECASE)
        info['soc'] = ('RK' + m.group(1)) if m else (soc_raw.split()[0] if soc_raw else '')
    except OSError:
        pass
    info['is_rockchip'] = bool(re.search(r'rockchip|rk\d', info['soc_raw'], re.IGNORECASE))

    # NPU 驱动版本
    try:
        with open('/sys/kernel/debug/rknpu/version') as f:
            info['npu_driver'] = f.read().strip()
    except OSError:
        pass

    # RKNN 运行时（优先 /usr/local/lib，容器/Docker 用的也是这份）
    lib, ver = _probe_lib_version(
        ('/usr/local/lib/librknnrt.so', '/usr/lib/librknnrt.so'), b'librknnrt version')
    info['rknn_lib'] = lib
    m = re.search(r'version:\s*([\d.]+)', ver)
    info['rknn_version'] = m.group(1) if m else ver

    # RKLLM 运行时
    lib2, ver2 = _probe_lib_version(
        ('/usr/local/lib/librkllmrt.so', '/usr/lib/librkllmrt.so'), b'RKLLM SDK')
    info['rkllm_lib'] = lib2
    m = re.search(r'version:\s*([\d.]+)', ver2)
    info['rkllm_version'] = m.group(1) if m else ver2

    # MPP（编解码加速）：pkg-config > dpkg > so 名
    mpp = _run_cmd('pkg-config --modversion rockchip_mpp 2>/dev/null').strip()
    if not mpp:
        m = re.search(r'librockchip-mpp1\s+(\S+)', _run_cmd('dpkg -l librockchip-mpp1 2>/dev/null'))
        if m:
            mpp = re.sub(r'-[0-9]+$', '', m.group(1))
    if not mpp:
        for p in ('/usr/lib/aarch64-linux-gnu/librockchip_mpp.so.0',
                  '/usr/lib/librockchip_mpp.so.0'):
            if os.path.exists(p):
                mm = re.search(r'_mpp\.so\.([\d.]+)$', p)
                if mm:
                    mpp = mm.group(1)
                    break
    info['mpp_version'] = mpp

    # RGA（2D 加速）：dpkg > so 名
    rga = ''
    m = re.search(r'librga2\s+(\S+)', _run_cmd('dpkg -l librga2 2>/dev/null'))
    if m:
        rga = re.sub(r'-[0-9]+$', '', m.group(1))
    if not rga:
        for p in ('/usr/lib/aarch64-linux-gnu/librga.so.2', '/usr/lib/librga.so.2'):
            if os.path.islink(p):
                target = os.readlink(p)
                mm = re.search(r'librga\.so\.([\d.]+)$', target)
                if mm:
                    rga = mm.group(1)
                    break
    info['rga_version'] = rga

    _HW_VER_CACHE = info
    _HW_VER_CACHE_TS = now
    return info


def get_hardware_info():
    """实时硬件指标：版本（缓存）+ NPU 各核负载（每次实时读）。"""
    static = _detect_rk_versions()
    cores = []
    try:
        with open('/sys/kernel/debug/rknpu/load') as f:
            content = f.read()
        for cm in re.finditer(r'Core\s*(\d+)\s*:\s*(\d+)%', content):
            cores.append({'core': int(cm.group(1)), 'load': int(cm.group(2))})
    except OSError:
        pass
    result = dict(static)
    result['npu_cores'] = cores
    result['npu_core_count'] = len(cores)
    return result


class OpenAIClient:
    def __init__(self, base_url="http://localhost:9698/v1", api_key="sk-ytllm"):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def check_server_status(self):
        start_time = time.time()
        try:
            response = self.client.models.list()
            end_time = time.time()
            return {
                "status": "success",
                "message": "服务器状态正常",
                "status_code": 200,
                "response_time": f"{end_time - start_time:.2f}"
            }
        except Exception as e:
            end_time = time.time()
            raise Exception(f"服务器状态异常 ({end_time - start_time:.2f}s): {str(e)}")

    def infer(self, model_id, question, image=None, max_new_tokens=100, history=None,
              repetition_penalty=None, temperature=None, top_p=None, top_k=None):
        print(f"\n执行推理")
        print(f"模型ID: {model_id}")
        print(f"问题: {question}")
        print(f"最大生成token数: {max_new_tokens}")
        if history:
            print(f"历史消息数量: {len(history)}")

        try:
            messages = []
            if history:
                for msg in history:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    messages.append({"role": role, "content": content})

            if image:
                print(f"处理图片...")
                image_data = image
                if Path(image).exists():
                    print(f"检测到图片文件路径，正在转换为base64...")
                    with open(image, "rb") as f:
                        image_bytes = f.read()
                        image_data = base64.b64encode(image_bytes).decode('utf-8')
                    print("图片转换完成")

                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                        }
                    ]
                })
            else:
                print("未提供图片，仅使用文本提示词进行推理")
                messages.append({"role": "user", "content": question})

            extra_body = {}
            if repetition_penalty is not None:
                extra_body["repetition_penalty"] = repetition_penalty
                print(f"重复惩罚系数: {repetition_penalty}")
            if top_k is not None:
                extra_body["top_k"] = top_k
                print(f"Top-k采样: {top_k}")

            start_time = time.time()
            response = self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature if temperature is not None else 0.7,
                top_p=top_p if top_p is not None else 0.9,
                extra_body=extra_body if extra_body else None
            )

            end_time = time.time()
            answer = response.choices[0].message.content

            print(f"推理成功，耗时: {end_time - start_time:.2f}秒")
            print(f"回答: {answer}")

            return {
                "status": "success",
                "answer": answer,
                "model_id": model_id,
                "inference_time": f"{end_time - start_time:.2f}",
                "tokens_generated": response.usage.completion_tokens if response.usage else 0
            }
        except Exception as e:
            print(f"执行推理失败: {str(e)}")
            raise

    def infer_stream(self, model_id, question, image=None, max_new_tokens=100, history=None,
                     repetition_penalty=None, temperature=None, top_p=None, top_k=None):
        print(f"\n执行流式推理")
        print(f"模型ID: {model_id}")
        print(f"问题: {question}")

        try:
            messages = []
            if history:
                for msg in history:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    messages.append({"role": role, "content": content})

            if image:
                image_data = image
                if Path(image).exists():
                    with open(image, "rb") as f:
                        image_bytes = f.read()
                        image_data = base64.b64encode(image_bytes).decode('utf-8')

                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                        }
                    ]
                })
            else:
                messages.append({"role": "user", "content": question})

            extra_body = {}
            if repetition_penalty is not None:
                extra_body["repetition_penalty"] = repetition_penalty
            if top_k is not None:
                extra_body["top_k"] = top_k

            stream = self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature if temperature is not None else 0.7,
                top_p=top_p if top_p is not None else 0.9,
                extra_body=extra_body if extra_body else None,
                stream=True
            )

            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content
        except Exception as e:
            print(f"执行流式推理失败: {str(e)}")
            raise

class ConfigManager:
    def __init__(self, config_file='config.json'):
        self.config_file = BASE_DIR / config_file
        self.config = self._load_config()

    def _load_config(self):
        default_config = {
            'theme': 'light',
            'language': 'zh-CN',
            'send_shortcut': 'Enter',
            'gateway_api_key': ''
        }
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        return default_config

    def _save_config(self):
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        self._save_config()

    def get_general_config(self):
        return {
            'theme': self.get('theme', 'light'),
            'language': self.get('language', 'zh-CN'),
            'send_shortcut': self.get('send_shortcut', 'Enter'),
            'gateway_api_key': self.get('gateway_api_key', '')
        }

    def set_general_config(self, theme=None, language=None, send_shortcut=None, gateway_api_key=None):
        if theme is not None:
            self.config['theme'] = theme
        if language is not None:
            self.config['language'] = language
        if send_shortcut is not None:
            self.config['send_shortcut'] = send_shortcut
        if gateway_api_key is not None:
            self.config['gateway_api_key'] = gateway_api_key
        self._save_config()

config_manager = ConfigManager()
service_store = ServiceStore()

class SessionManager:
    def __init__(self):
        self.sessions = {}

    def get(self, session_id):
        return self.sessions.get(session_id)

    def set(self, session_id, value):
        self.sessions[session_id] = value

    def clear_expired(self):
        pass

session_manager = SessionManager()

def get_current_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def localtime(dt_str):
    return dt_str


def serialize_service(service, reveal_key=False):
    """统一的对外结构（列表/详情/启停接口共用）。"""
    api_key = service['api_key'] or ''
    if reveal_key:
        shown_key = api_key
    else:
        shown_key = api_key[:8] + '***' if len(api_key) > 8 else '***'
    return {
        'id': str(service['id']),
        'name': service['name'],
        'service_type': service['service_type'] or 'remote',
        'base_url': service['base_url'],
        'api_key': shown_key,
        'model_id': service['model_id'],
        'is_active': bool(service['is_active']),
        'last_status': service['last_status'],
        'last_checked_at': service['last_checked_at'],
        'engine_type': service['engine_type'],
        'max_tokens': service['max_tokens'],
        'description': service['description'],
        'model_path': service['model_path'] or '',
        'vision_model_path': service['vision_model_path'] or '',
        'platform': service['platform'] or 'rk3576',
        'rknn_cores': service['rknn_cores'] or 2,
        'port': service['port'],
        'pid': service['pid'],
        'status': service['status'] or 'stopped',
        'started_at': service['started_at'],
        'last_error': service['last_error'] or '',
        'params': service['params'] or {},
        'created_at': service['created_at'],
        'updated_at': service['updated_at'],
    }


def parse_body(handler):
    """兼容 JSON 与表单两种提交方式。"""
    try:
        return json.loads(handler.request.body)
    except (ValueError, TypeError):
        args = handler.request.arguments
        return {k: v[0].decode('utf-8') if isinstance(v[0], bytes) else v[0] for k, v in args.items()}


def _list_services_with_sync():
    return [sync_local_status(row) for row in service_store.list()]


def sync_local_status(service):
    """本地服务以真实进程存活为准：进程没了就是停了，PID 顺手清掉。

    status='error' 时保留错误信息不覆盖，否则用户会因为"状态被自动改成 stopped"
    而看不到上次启动失败的原因。
    """
    if (service['service_type'] or 'remote') != 'local':
        return service

    stored = service['status'] or 'stopped'
    if engine_manager.pid_alive(service['pid']):
        if stored != 'running':
            return service_store.update(service['id'], status='running') or service
        return service

    if service['pid'] is not None:
        new_status = 'error' if stored == 'error' else 'stopped'
        return service_store.update(service['id'], status=new_status, pid=None) or service
    return service


class BaseHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header('Cache-Control', 'no-cache')

    def get_current_user(self):
        return self.get_cookie("user_id")

    def get_app_version(self):
        return APP_VERSION

    def render(self, template_name, **kwargs):
        """走 Tornado 模板引擎（支持 extends / block），模板根目录为 TEMPLATE_DIR。
        找不到模板时回退为读取原始文件。"""
        kwargs.setdefault('app_version', APP_VERSION)
        try:
            self.write(self.render_string(template_name, **kwargs))
            return
        except FileNotFoundError:
            pass
        template_path = TEMPLATE_DIR / template_name
        if not template_path.exists():
            template_path = TEMPLATE_DIR / 'core' / template_name
        if template_path.exists():
            with open(template_path, 'r', encoding='utf-8') as f:
                self.write(f.read())
        else:
            self.set_status(404)
            self.write(f"Template not found: {template_name}")

    def json_response(self, data):
        self.set_header('Content-Type', 'application/json')
        self.write(json.dumps(data, ensure_ascii=False))

class HomeHandler(BaseHandler):
    async def get(self):
        loop = tornado.ioloop.IOLoop.current()
        services = await loop.run_in_executor(None, lambda: service_store.list(active_only=True))
        self.render('core/home.html', services=services)

class ModelListHandler(BaseHandler):
    async def get(self):
        loop = tornado.ioloop.IOLoop.current()
        services = await loop.run_in_executor(None, lambda: service_store.list())
        self.render('core/model_list.html', services=services)


class PlaygroundHandler(BaseHandler):
    async def get(self):
        self.render('core/playground.html')

class SystemStatsHandler(BaseHandler):
    async def get(self):
        import platform
        import subprocess
        import psutil

        loop = tornado.ioloop.IOLoop.current()
        data = await loop.run_in_executor(None, _get_system_stats)
        self.json_response(data)

def _get_system_stats():
    import platform
    import subprocess
    import psutil

    device_name = platform.node()
    try:
        if platform.system() == "Windows":
            device_name = subprocess.check_output("hostname", shell=True).decode('utf-8').strip()
    except:
        pass

    os_name = f"{platform.system()} {platform.release()}"
    if platform.system() == "Windows":
        try:
            os_detail = subprocess.check_output("wmic os get caption", shell=True).decode('gbk').strip().split('\n')
            if len(os_detail) > 1:
                os_name = os_detail[1].strip().replace('Microsoft ', '')
        except:
            pass

    disk_info = {'usage': 0, 'total': 0, 'used': 0, 'free': 0}
    try:
        disk = psutil.disk_usage('C:\\' if platform.system() == 'Windows' else '/')
        disk_info = {
            'usage': disk.percent,
            'total': round(disk.total / (1024**3), 1),
            'used': round(disk.used / (1024**3), 1),
            'free': round(disk.free / (1024**3), 1)
        }
    except:
        pass

    cpu_usage = psutil.cpu_percent(interval=None)
    cpu_model = "Unknown CPU"
    try:
        cpu_info = subprocess.check_output("wmic cpu get name", shell=True).decode('gbk').strip().split('\n')
        if len(cpu_info) > 1:
            cpu_model = cpu_info[1].strip()
    except:
        cpu_model = platform.processor()

    gpus = []
    try:
        gpu_output = subprocess.check_output(
            'wmic path win32_videocontroller get name /format:csv',
            shell=True
        ).decode('gbk').strip().split('\n')

        for line in gpu_output[1:]:
            line = line.strip()
            if line and ',' in line:
                csv_parts = line.split(',')
                if len(csv_parts) >= 2:
                    gpu_name = csv_parts[-1]
                    if not gpu_name:
                        continue
                    gpu_name_lower = gpu_name.lower()
                    if 'nvidia' in gpu_name_lower or 'geforce' in gpu_name_lower or 'rtx' in gpu_name_lower or 'gtx' in gpu_name_lower:
                        gpu_type = 'nvidia'
                    elif 'amd' in gpu_name_lower or 'radeon' in gpu_name_lower:
                        gpu_type = 'amd'
                    elif 'intel' in gpu_name_lower or 'uhd' in gpu_name_lower or 'iris' in gpu_name_lower:
                        gpu_type = 'intel'
                    else:
                        gpu_type = 'other'
                    gpus.append({'name': gpu_name, 'type': gpu_type})
    except:
        gpus = [{'name': '未检测到显卡', 'type': 'other'}]

    if not gpus:
        gpus = [{'name': '未检测到显卡', 'type': 'other'}]

    memory = psutil.virtual_memory()
    total_mem = round(memory.total / (1024**3), 1)

    return {
        'device_name': device_name,
        'os_name': os_name,
        'disk': disk_info,
        'cpu_usage': cpu_usage,
        'cpu_model': cpu_model,
        'gpus': gpus,
        'memory_usage': memory.percent,
        'total_mem': total_mem
    }

class LoginAPIHandler(BaseHandler):
    async def post(self):
        try:
            data = json.loads(self.request.body)
            username = data.get('username', '').strip()
            password = data.get('password', '').strip()
        except:
            username = self.get_body_argument('username', '').strip()
            password = self.get_body_argument('password', '').strip()

        if not username or not password:
            self.json_response({'status': 'error', 'message': '用户名和密码不能为空'})
            return

        user_info = {
            'id': 10086,
            'code': str(uuid.uuid4())[:8].upper(),
            'username': username,
            'nickname': f'{username}的昵称',
            'last_login_time': get_current_time()
        }

        self.set_cookie('user_id', str(user_info['id']))
        self.json_response({'status': 'success', 'message': '登录成功', 'user': user_info})

class OpenAIServiceListAPIHandler(BaseHandler):
    async def get(self):
        loop = tornado.ioloop.IOLoop.current()
        services = await loop.run_in_executor(None, _list_services_with_sync)
        self.json_response({
            'status': 'success',
            'services': [serialize_service(s) for s in services],
        })

class OpenAIServiceCreateAPIHandler(BaseHandler):
    async def post(self):
        data = parse_body(self)
        loop = tornado.ioloop.IOLoop.current()

        service_type = (data.get('service_type') or 'remote').strip()
        name = (data.get('name') or '').strip()
        model_id = (data.get('model_id') or '').strip()
        description = (data.get('description') or '').strip()
        engine_type = (data.get('engine_type') or '').strip()
        max_tokens = int(data.get('max_tokens') or 4096)

        if not name or not model_id:
            return self.json_response({'status': 'error', 'message': '服务名称和模型 ID 不能为空'})

        record = {
            'base_url': '', 'api_key': 'sk-ytllm', 'model_path': '', 'vision_model_path': '',
            'platform': 'rk3576', 'rknn_cores': 2, 'port': None, 'params': {},
        }

        if service_type == 'local':
            try:
                record['model_path'] = model_store.resolve((data.get('model_name') or '').strip())
                vision_name = (data.get('vision_model_name') or '').strip()
                record['vision_model_path'] = model_store.resolve(vision_name) if vision_name else ''
            except ModelStoreError as e:
                return self.json_response({'status': 'error', 'message': str(e)})

            record['platform'] = (data.get('platform') or 'rk3576').strip()
            record['rknn_cores'] = int(data.get('rknn_cores') or 2)
            record['params'] = {**DEFAULT_ENGINE_PARAMS, **(data.get('params') or {})}
            engine_type = engine_type or 'RKLLM'

            used = await loop.run_in_executor(None, lambda: service_store.allocated_ports())
            try:
                record['port'] = await loop.run_in_executor(None, lambda: engine_manager.find_free_port(used))
            except EngineError as e:
                return self.json_response({'status': 'error', 'message': str(e)})
            record['base_url'] = f"http://127.0.0.1:{record['port']}/v1"
        else:
            base_url = (data.get('base_url') or '').strip()
            if not base_url:
                return self.json_response({'status': 'error', 'message': 'API 地址不能为空'})
            record['base_url'] = base_url if base_url.endswith('/v1') else base_url.rstrip('/') + '/v1'
            record['api_key'] = (data.get('api_key') or 'sk-ytllm').strip()

        service_id = str(uuid.uuid4())
        now = get_current_time()
        await loop.run_in_executor(None, lambda: service_store.create(
            id=service_id, name=name, service_type=service_type,
            base_url=record['base_url'], api_key=record['api_key'], model_id=model_id,
            engine_type=engine_type, max_tokens=max_tokens, description=description,
            model_path=record['model_path'], vision_model_path=record['vision_model_path'],
            platform=record['platform'], rknn_cores=record['rknn_cores'], port=record['port'],
            params=record['params'], created_at=now, updated_at=now))

        self.json_response({
            'status': 'success',
            'message': '服务创建成功',
            'service': {'id': service_id, 'name': name, 'base_url': record['base_url']}
        })

class OpenAIServiceUpdateAPIHandler(BaseHandler):
    async def post(self, pk):
        try:
            data = json.loads(self.request.body)
        except:
            data = self.request.arguments
            data = {k: v[0].decode('utf-8') if isinstance(v[0], bytes) else v[0] for k, v in data.items()}

        loop = tornado.ioloop.IOLoop.current()
        service = await loop.run_in_executor(None, lambda: service_store.get(pk))
        if not service:
            self.json_response({'status': 'error', 'message': '服务不存在'})
            return

        fields = {}

        if 'name' in data:
            fields['name'] = data['name'].strip()
        if 'base_url' in data:
            base_url = data['base_url'].strip()
            fields['base_url'] = base_url if base_url.endswith('/v1') else base_url.rstrip('/') + '/v1'
        if 'api_key' in data:
            fields['api_key'] = data['api_key'].strip()
        if 'model_id' in data:
            fields['model_id'] = data['model_id'].strip()
        if 'engine_type' in data:
            fields['engine_type'] = data['engine_type'].strip()
        if 'max_tokens' in data:
            fields['max_tokens'] = int(data['max_tokens'])
        if 'model_name' in data:
            try:
                fields['model_path'] = model_store.resolve((data['model_name'] or '').strip())
            except ModelStoreError as e:
                return self.json_response({'status': 'error', 'message': str(e)})
        if 'vision_model_name' in data:
            vision_name = (data['vision_model_name'] or '').strip()
            try:
                fields['vision_model_path'] = model_store.resolve(vision_name) if vision_name else ''
            except ModelStoreError as e:
                return self.json_response({'status': 'error', 'message': str(e)})
        if 'platform' in data:
            fields['platform'] = (data['platform'] or 'rk3576').strip()
        if 'rknn_cores' in data:
            fields['rknn_cores'] = int(data['rknn_cores'])
        if 'params' in data:
            fields['params'] = data['params']
        if 'description' in data:
            fields['description'] = data['description'].strip()
        if 'is_active' in data:
            fields['is_active'] = 1 if data['is_active'] in (True, 'true', '1', 1) else 0

        if fields:
            fields['updated_at'] = get_current_time()
            await loop.run_in_executor(None, lambda: service_store.update(pk, **fields))

        self.json_response({'status': 'success', 'message': '服务更新成功'})

class OpenAIServiceDeleteAPIHandler(BaseHandler):
    async def post(self, pk):
        loop = tornado.ioloop.IOLoop.current()
        service = await loop.run_in_executor(None, lambda: service_store.get(pk))
        if not service:
            return self.json_response({'status': 'error', 'message': '服务不存在'})

        # 本地服务先把引擎进程收掉，否则删完记录会留下孤儿进程占着端口和内存
        if service['service_type'] == 'local':
            await loop.run_in_executor(None, lambda: engine_manager.stop(service))

        await loop.run_in_executor(None, lambda: service_store.delete(pk))
        self.json_response({'status': 'success', 'message': f'服务 {service["name"]} 已删除'})

class OpenAIServiceCheckAPIHandler(BaseHandler):
    async def post(self, pk):
        loop = tornado.ioloop.IOLoop.current()
        service = await loop.run_in_executor(None, lambda: service_store.get(pk))
        if not service:
            self.json_response({'status': 'error', 'message': '服务不存在'})
            return

        try:
            client = OpenAIClient(base_url=service['base_url'], api_key=service['api_key'] or 'sk-ytllm')
            await loop.run_in_executor(None, client.check_server_status)
            status = 'online'
        except:
            status = 'offline'

        now = get_current_time()
        await loop.run_in_executor(None, lambda: service_store.update(
            pk, last_status=status, last_checked_at=now, updated_at=now))

        self.json_response({'status': 'success', 'service_status': status, 'message': '服务在线' if status == 'online' else '服务离线'})

class OpenAIServiceToggleAPIHandler(BaseHandler):
    async def post(self, pk):
        loop = tornado.ioloop.IOLoop.current()
        service = await loop.run_in_executor(None, lambda: service_store.get(pk))
        if not service:
            self.json_response({'status': 'error', 'message': '服务不存在'})
            return

        new_status = 0 if service['is_active'] else 1
        await loop.run_in_executor(None, lambda: service_store.update(
            pk, is_active=new_status, updated_at=get_current_time()))

        self.json_response({'status': 'success', 'is_active': bool(new_status), 'message': f'服务已{"启用" if new_status else "禁用"}'})

class ActiveServicesAPIHandler(BaseHandler):
    async def get(self):
        loop = tornado.ioloop.IOLoop.current()
        services = await loop.run_in_executor(None, lambda: service_store.list(active_only=True))
        result = []
        for s in services:
            result.append({
                'id': str(s['id']),
                'name': s['name'],
                'base_url': s['base_url'],
                'api_key': s['api_key'],
                'model_id': s['model_id'],
                'engine_type': s['engine_type'],
                'max_tokens': s['max_tokens'],
                'last_status': s['last_status'],
            })
        self.json_response({'status': 'success', 'services': result})

class GeneralConfigAPIHandler(BaseHandler):
    async def get(self):
        self.json_response({'status': 'success', 'config': config_manager.get_general_config()})

    async def post(self):
        try:
            data = json.loads(self.request.body)
        except:
            data = self.request.arguments
            data = {k: v[0].decode('utf-8') if isinstance(v[0], bytes) else v[0] for k, v in data.items()}

        theme = data.get('theme')
        language = data.get('language')
        send_shortcut = data.get('send_shortcut')
        gateway_api_key = data.get('gateway_api_key')

        loop = tornado.ioloop.IOLoop.current()
        await loop.run_in_executor(None, lambda: config_manager.set_general_config(
            theme=theme, language=language, send_shortcut=send_shortcut, gateway_api_key=gateway_api_key))
        self.json_response({'status': 'success', 'message': '常规设置更新成功'})

class CheckVersionAPIHandler(BaseHandler):
    async def get(self):
        current_version = self.get_app_version()
        self.json_response({
            'current_version': current_version,
            'latest_version': current_version,
            'has_update': False,
            'download_url': '',
            'release_notes': '版本检测服务未启动'
        })

# ============================================================================
# 模型仓库
# ============================================================================

def _services_using_model(name):
    """哪些服务引用了这个模型文件——删之前先挡住，免得删掉正在用的模型。"""
    hits = []
    for row in service_store.list():
        for path in (row['model_path'], row['vision_model_path']):
            if path and Path(path).name == name:
                hits.append(row['name'])
                break
    return hits


class ModelFileListAPIHandler(BaseHandler):
    async def get(self):
        loop = tornado.ioloop.IOLoop.current()
        files = await loop.run_in_executor(None, model_store.list_models)
        disk = await loop.run_in_executor(None, model_store.disk_usage)
        self.json_response({'status': 'success', 'files': files, 'disk': disk})


class ModelFileUploadAPIHandler(BaseHandler):
    """分片上传：?upload_id=x&filename=y&index=i&total=n，请求体为该分片的二进制内容。"""

    def prepare(self):
        # Content-Type 一旦被识别成 form-urlencoded，Tornado 会把二进制分片当表单解析成
        # 百万级字段，直接回一个没有任何解释的 400（Max number of fields exceeded）。
        # 这里提前拦下，换成能看懂的提示。
        ctype = (self.request.headers.get('Content-Type') or '').split(';')[0].strip().lower()
        if ctype != 'application/octet-stream':
            self.set_status(415)
            self.json_response({
                'status': 'error',
                'message': f'分片上传要求 Content-Type: application/octet-stream，当前收到「{ctype or "未设置"}」',
            })
            self.finish()

    async def post(self):
        upload_id = self.get_query_argument('upload_id', '')
        filename = self.get_query_argument('filename', '')
        index = self.get_query_argument('index', '0')
        total = self.get_query_argument('total', '1')
        payload = self.request.body

        loop = tornado.ioloop.IOLoop.current()
        try:
            result = await loop.run_in_executor(
                None, lambda: model_store.write_chunk(upload_id, filename, index, total, payload))
        except ModelStoreError as e:
            return self.json_response({'status': 'error', 'message': str(e)})
        self.json_response({'status': 'success', **result})


class ModelFileAbortAPIHandler(BaseHandler):
    async def post(self):
        data = parse_body(self)
        loop = tornado.ioloop.IOLoop.current()
        try:
            await loop.run_in_executor(None, lambda: model_store.abort_upload(data.get('upload_id', '')))
        except ModelStoreError as e:
            return self.json_response({'status': 'error', 'message': str(e)})
        self.json_response({'status': 'success', 'message': '已取消上传'})


class ModelFileDeleteAPIHandler(BaseHandler):
    async def post(self):
        data = parse_body(self)
        name = (data.get('name') or '').strip()
        loop = tornado.ioloop.IOLoop.current()

        in_use = await loop.run_in_executor(None, lambda: _services_using_model(name))
        if in_use:
            return self.json_response({
                'status': 'error',
                'message': f'该模型正被服务「{"、".join(in_use)}」引用，请先删除对应服务',
            })
        try:
            await loop.run_in_executor(None, lambda: model_store.delete(name))
        except ModelStoreError as e:
            return self.json_response({'status': 'error', 'message': str(e)})
        self.json_response({'status': 'success', 'message': '模型已删除'})


# ============================================================================
# 推理实例（本地引擎进程）
# ============================================================================

def _get_service(pk):
    return service_store.get(pk)


def _get_service_by_ref(ref):
    return service_store.get_by_ref(ref)


def _update_service(pk, **fields):
    service_store.update(pk, **fields)


class ServiceStartAPIHandler(BaseHandler):
    async def post(self, pk):
        loop = tornado.ioloop.IOLoop.current()
        service = await loop.run_in_executor(None, lambda: _get_service(pk))
        if not service:
            return self.json_response({'status': 'error', 'message': '服务不存在'})
        if service['service_type'] != 'local':
            return self.json_response({'status': 'error', 'message': '远程接入类型的服务无需启动本地进程'})

        model_path = Path(service['model_path'])
        if not model_path.is_file():
            return self.json_response({'status': 'error', 'message': f'模型文件不存在：{model_path}'})

        await loop.run_in_executor(None, lambda: _update_service(pk, status='starting', last_error=''))
        try:
            pid = await loop.run_in_executor(None, lambda: engine_manager.start(service))
        except EngineError as e:
            detail = str(e)
            await loop.run_in_executor(
                None, lambda: _update_service(pk, status='error', pid=None, last_error=detail))
            return self.json_response({'status': 'error', 'message': detail})

        await loop.run_in_executor(None, lambda: _update_service(
            pk, status='running', pid=pid, started_at=get_current_time(), last_error=''))
        self.json_response({
            'status': 'success', 'message': '服务已启动',
            'pid': pid, 'port': service['port'], 'base_url': service['base_url'],
        })

    async def get(self, pk):
        """GET 也允许，方便直接点开链接启动。"""
        await self.post(pk)


class ServiceStopAPIHandler(BaseHandler):
    async def post(self, pk):
        loop = tornado.ioloop.IOLoop.current()
        service = await loop.run_in_executor(None, lambda: _get_service(pk))
        if not service:
            return self.json_response({'status': 'error', 'message': '服务不存在'})

        stopped = await loop.run_in_executor(None, lambda: engine_manager.stop(service))
        if not stopped:
            await loop.run_in_executor(None, lambda: _update_service(pk, status='running'))
            return self.json_response({'status': 'error', 'message': '进程未能停止，请检查系统权限'})

        await loop.run_in_executor(None, lambda: _update_service(pk, status='stopped', pid=None))
        self.json_response({'status': 'success', 'message': '服务已停止'})


class ServiceRestartAPIHandler(BaseHandler):
    async def post(self, pk):
        loop = tornado.ioloop.IOLoop.current()
        service = await loop.run_in_executor(None, lambda: _get_service(pk))
        if not service:
            return self.json_response({'status': 'error', 'message': '服务不存在'})
        if service['service_type'] != 'local':
            return self.json_response({'status': 'error', 'message': '远程接入类型的服务无需重启进程'})

        await loop.run_in_executor(None, lambda: engine_manager.stop(service))
        await loop.run_in_executor(None, lambda: _update_service(
            pk, status='starting', pid=None, last_error=''))
        try:
            pid = await loop.run_in_executor(None, lambda: engine_manager.start(service))
        except EngineError as e:
            detail = str(e)
            await loop.run_in_executor(
                None, lambda: _update_service(pk, status='error', pid=None, last_error=detail))
            return self.json_response({'status': 'error', 'message': detail})

        await loop.run_in_executor(None, lambda: _update_service(
            pk, status='running', pid=pid, started_at=get_current_time(), last_error=''))
        self.json_response({'status': 'success', 'message': '服务已重启', 'pid': pid})


class ServiceLogAPIHandler(BaseHandler):
    async def get(self, pk):
        try:
            lines = min(int(self.get_query_argument('lines', '200')), 2000)
        except ValueError:
            lines = 200
        loop = tornado.ioloop.IOLoop.current()
        logs = await loop.run_in_executor(None, lambda: engine_manager.read_log(pk, lines))
        self.json_response({'status': 'success', 'logs': logs})


class EngineRuntimeAPIHandler(BaseHandler):
    """引擎运行环境自检 + 总览统计。"""

    async def get(self):
        loop = tornado.ioloop.IOLoop.current()
        runtime = await loop.run_in_executor(None, read_engine_runtime)
        stats = await loop.run_in_executor(None, _engine_overview)
        self.json_response({'status': 'success', 'runtime': runtime, 'overview': stats})


def _engine_overview():
    counts = service_store.counts()
    counts['models'] = len(model_store.list_models())
    return counts


class HardwareAPIHandler(BaseHandler):
    """Rockchip 硬件加速指标：SoC / NPU 驱动与负载 / RKNN / RKLLM / MPP / RGA。"""

    async def get(self):
        loop = tornado.ioloop.IOLoop.current()
        info = await loop.run_in_executor(None, get_hardware_info)
        self.json_response({'status': 'success', 'hardware': info})


class OpenAIGatewayHandler(BaseHandler):
    """OpenAI 兼容对外网关：/v1/chat/completions、/v1/completions、/v1/models。
    通过 X-Service-Id / X-Service-Name 头、请求体 service 字段、或 OpenAI 必填的 model 字段指定目标服务。
    gateway_api_key 非空时强制校验 Authorization: Bearer <key>（或 X-Api-Key）。"""

    def _auth_fail(self):
        self.set_status(401)
        self.set_header('Content-Type', 'application/json')
        self.write(json.dumps({'error': {'message': 'Invalid API key', 'type': 'invalid_request_error', 'code': 'invalid_api_key'}}, ensure_ascii=False))
        self.finish()

    def _require_auth(self):
        gw_key = config_manager.get('gateway_api_key', '')
        if not gw_key:
            return True
        auth = self.request.headers.get('Authorization', '') or self.request.headers.get('X-Api-Key', '')
        token = ''
        if auth:
            token = auth[7:].strip() if auth.lower().startswith('bearer ') else auth.strip()
        return token == gw_key

    async def prepare(self):
        if not self._require_auth():
            self._auth_fail()

    def _resolve_service(self):
        ref = (self.request.headers.get('X-Service-Id') or self.request.headers.get('X-Service-Name') or '').strip()
        body = {}
        try:
            body = json.loads(self.request.body or b'{}') or {}
        except Exception:
            body = {}
        if not ref and isinstance(body, dict):
            ref = (body.get('service') or '').strip()
        if not ref and isinstance(body, dict):
            ref = (str(body.get('model') or '')).strip()
        return ref, body

    async def get(self, path):
        if path == 'models':
            await self._list_models()
            return
        self.set_status(405)
        self.set_header('Content-Type', 'application/json')
        self.write(json.dumps({'error': {'message': 'Method not allowed', 'type': 'invalid_request_error'}}, ensure_ascii=False))

    async def post(self, path):
        if path == 'models':
            await self._list_models()
            return
        ref, body = self._resolve_service()
        if not ref:
            self.set_status(400)
            self.set_header('Content-Type', 'application/json')
            self.write(json.dumps({'error': {'message': '未指定服务：请设置请求头 X-Service-Id（或 X-Service-Name），或在 model 字段填写服务名', 'type': 'invalid_request_error'}}, ensure_ascii=False))
            return
        loop = tornado.ioloop.IOLoop.current()
        service = await loop.run_in_executor(None, lambda: _get_service_by_ref(ref))
        if not service:
            self.set_status(404)
            self.set_header('Content-Type', 'application/json')
            self.write(json.dumps({'error': {'message': '服务不存在：%s' % ref, 'type': 'invalid_request_error'}}, ensure_ascii=False))
            return
        base = (service.get('base_url') or '').rstrip('/')
        if not base:
            self.set_status(400)
            self.set_header('Content-Type', 'application/json')
            self.write(json.dumps({'error': {'message': '目标服务未配置 base_url', 'type': 'invalid_request_error'}}, ensure_ascii=False))
            return
        target = base + '/' + path
        fwd = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        svc_key = (service.get('api_key') or '').strip()
        if svc_key:
            fwd['Authorization'] = 'Bearer ' + svc_key
        is_stream = bool(body.get('stream')) if isinstance(body, dict) else False
        loop = tornado.ioloop.IOLoop.current()
        try:
            if is_stream:
                self.set_status(200)
                self.set_header('Content-Type', 'text/event-stream')
                self.set_header('Cache-Control', 'no-cache')
                self.set_header('Connection', 'keep-alive')
                self.set_header('X-Accel-Buffering', 'no')

                def on_chunk(chunk):
                    self.write(chunk)
                    self.flush()

                http_client = tornado.httpclient.AsyncHTTPClient()
                req = tornado.httpclient.HTTPRequest(
                    url=target, method='POST',
                    body=json.dumps(body).encode('utf-8'),
                    headers=fwd, request_timeout=600,
                    streaming_callback=on_chunk)
                await http_client.fetch(req)
                if not self._finished:
                    self.finish()
            else:
                resp = await loop.run_in_executor(None, lambda: requests.post(target, json=body, headers=fwd, timeout=600))
                self.set_status(resp.status_code)
                self.set_header('Content-Type', resp.headers.get('content-type', 'application/json'))
                self.write(resp.content)
        except Exception as e:
            if not self._finished:
                self.set_status(502)
                self.set_header('Content-Type', 'application/json')
                self.write(json.dumps({'error': {'message': '网关转发失败：%s' % e, 'type': 'gateway_error'}}, ensure_ascii=False))

    async def _list_models(self):
        loop = tornado.ioloop.IOLoop.current()
        rows = await loop.run_in_executor(None, _list_services_with_sync)
        data = []
        for r in rows:
            d = dict(r)
            mid = d.get('name') or d.get('id')
            data.append({
                'id': mid,
                'object': 'model',
                'created': 0,
                'owned_by': 'ytllm',
                'root': mid,
                'service_id': d.get('id'),
                'service_type': d.get('service_type'),
                'status': d.get('status'),
                'model_id': d.get('model_id')
            })
        self.set_status(200)
        self.set_header('Content-Type', 'application/json')
        self.write(json.dumps({'object': 'list', 'data': data}, ensure_ascii=False))


# ============================================================================
# 模型对话台
# ============================================================================

def _friendly_error(exc):
    text = str(exc).strip() or exc.__class__.__name__
    if 'APIConnectionError' in exc.__class__.__name__ or 'Connection' in text:
        return f'无法连接推理服务（进程未启动或端口不通）：{text}'
    return text


def _playground_extra(params):
    extra = {}
    if params.get('top_k') is not None:
        extra['top_k'] = int(params['top_k'])
    if params.get('repeat_penalty') is not None:
        extra['repetition_penalty'] = float(params['repeat_penalty'])
    return extra or None


def _playground_kwargs(service, prompt, params, image=None):
    """构造 OpenAI 请求体；image 为 data URL 时走多模态 content 数组。

    注意：RKLLM 引擎一次仅处理一张图片（见 rkllm_engine.py 的 image_data），
    因此测试台也只接受单张，多张时由前端限制。
    """
    if image:
        content = [
            {'type': 'text', 'text': prompt},
            {'type': 'image_url', 'image_url': {'url': image}},
        ]
    else:
        content = prompt
    return dict(
        model=service['model_id'],
        messages=[{'role': 'user', 'content': content}],
        max_tokens=int(params.get('max_tokens') or service['max_tokens'] or 512),
        temperature=float(params.get('temperature', 0.8)),
        top_p=float(params.get('top_p', 0.9)),
        extra_body=_playground_extra(params),
    )


def _playground_client(service):
    return OpenAI(api_key=service['api_key'] or 'sk-ytllm', base_url=service['base_url'])


def _playground_call(service, prompt, params, image=None):
    client = _playground_client(service)
    started = time.time()
    resp = client.chat.completions.create(**_playground_kwargs(service, prompt, params, image))
    elapsed = round(time.time() - started, 2)
    content = resp.choices[0].message.content or ''
    usage = getattr(resp, 'usage', None)
    return {
        'content': content,
        'elapsed': elapsed,
        'chars': len(content),
        'completion_tokens': getattr(usage, 'completion_tokens', 0) if usage else 0,
    }


def _playground_stream(service, prompt, params, image=None):
    client = _playground_client(service)
    stream = client.chat.completions.create(
        **_playground_kwargs(service, prompt, params, image), stream=True)
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


class PlaygroundChatAPIHandler(BaseHandler):
    """测试台专用转发：只试接口，不写会话库。"""

    async def post(self):
        data = parse_body(self)
        service_id = (data.get('service_id') or '').strip()
        prompt = (data.get('prompt') or '').strip()
        params = data.get('params') or {}
        want_stream = data.get('stream', True)
        image = (data.get('image') or '').strip()

        if image and not image.startswith('data:image/'):
            return self.json_response({'status': 'error', 'message': '图片格式不合法，需为 data:image/... 形式'})

        if not prompt:
            if not image:
                return self.json_response({'status': 'error', 'message': '请输入测试内容'})
            prompt = '请描述这张图片'

        loop = tornado.ioloop.IOLoop.current()
        service = await loop.run_in_executor(None, lambda: _get_service(service_id))
        if not service:
            return self.json_response({'status': 'error', 'message': '服务不存在'})

        if image and service['service_type'] == 'local' and not service['vision_model_path']:
            return self.json_response({
                'status': 'error',
                'message': '该服务未配置视觉模型（vision_model_path 为空），无法处理图片。请在推理服务页为该模型指定视觉模型后重启服务。'
            })

        if not want_stream:
            try:
                result = await loop.run_in_executor(
                    None, lambda: _playground_call(service, prompt, params, image))
            except Exception as e:
                return self.json_response({'status': 'error', 'message': _friendly_error(e)})
            return self.json_response({'status': 'success', **result})

        self.set_header('Content-Type', 'text/event-stream')
        self.set_header('Cache-Control', 'no-cache')
        self.set_header('X-Accel-Buffering', 'no')

        started = time.time()
        self.write(f"data: {json.dumps({'type': 'start'})}\n\n")
        await self.flush()

        q = queue.Queue()

        def _run():
            try:
                for chunk in _playground_stream(service, prompt, params, image):
                    q.put(('chunk', chunk))
                q.put(('done', None))
            except Exception as e:
                q.put(('error', _friendly_error(e)))

        await loop.run_in_executor(None, _run)

        collected = ''
        while True:
            try:
                kind, payload = q.get(timeout=0.1)
            except queue.Empty:
                await tornado.gen.sleep(0.05)
                continue
            if kind == 'chunk':
                collected += payload
                self.write(f"data: {json.dumps({'type': 'chunk', 'content': payload})}\n\n")
                await self.flush()
            elif kind == 'done':
                break
            else:
                self.write(f"data: {json.dumps({'type': 'error', 'message': payload})}\n\n")
                await self.flush()
                return

        elapsed = round(time.time() - started, 2)
        self.write(f"data: {json.dumps({'type': 'done', 'content': collected, 'elapsed': elapsed, 'chars': len(collected)})}\n\n")
        await self.flush()


class StaticFileHandler(tornado.web.StaticFileHandler):
    def set_default_headers(self):
        self.set_header('Cache-Control', 'no-cache')

def make_app():
    return tornado.web.Application([
        (r"/", HomeHandler),
        (r"/models/", ModelListHandler),
        (r"/models/api/services/", OpenAIServiceListAPIHandler),
        (r"/models/api/services/create/", OpenAIServiceCreateAPIHandler),
        (r"/models/api/services/([^/]+)/update/", OpenAIServiceUpdateAPIHandler),
        (r"/models/api/services/([^/]+)/delete/", OpenAIServiceDeleteAPIHandler),
        (r"/models/api/services/([^/]+)/check/", OpenAIServiceCheckAPIHandler),
        (r"/models/api/services/([^/]+)/toggle/", OpenAIServiceToggleAPIHandler),
        (r"/models/api/files/", ModelFileListAPIHandler),
        (r"/models/api/files/upload/", ModelFileUploadAPIHandler),
        (r"/models/api/files/upload/abort/", ModelFileAbortAPIHandler),
        (r"/models/api/files/delete/", ModelFileDeleteAPIHandler),
        (r"/models/api/services/([^/]+)/start/", ServiceStartAPIHandler),
        (r"/models/api/services/([^/]+)/stop/", ServiceStopAPIHandler),
        (r"/models/api/services/([^/]+)/restart/", ServiceRestartAPIHandler),
        (r"/models/api/services/([^/]+)/logs/", ServiceLogAPIHandler),
        (r"/api/engine/runtime/", EngineRuntimeAPIHandler),
        (r"/api/hardware/", HardwareAPIHandler),
        (r"/chat/", PlaygroundHandler),
        (r"/chat/api/chat/", PlaygroundChatAPIHandler),
        (r"/api/active-services/", ActiveServicesAPIHandler),
        (r"/api/stats/", SystemStatsHandler),
        (r"/api/login/", LoginAPIHandler),
        (r"/api/config/", GeneralConfigAPIHandler),
        (r"/api/version/check/", CheckVersionAPIHandler),
        (r"/v1/(chat/completions|completions|models)", OpenAIGatewayHandler),
        (r"/static/(.*)", StaticFileHandler, {"path": str(STATIC_DIR)}),
        (r"/uploads/(.*)", StaticFileHandler, {"path": str(UPLOADS_DIR)}),
    ], debug=tornado.options.options.debug, template_path=str(TEMPLATE_DIR))

if __name__ == "__main__":
    tornado.options.parse_command_line()
    port = tornado.options.options.port
    print(f"启动 Tornado 服务，端口: {port}")

    # 上次退出可能留下「DB 说 running、进程其实早没了」的记录，启动时先跟真实进程对齐
    for _row in service_store.list(local_only=True):
        sync_local_status(_row)

    app = make_app()
    app.listen(port)
    tornado.ioloop.IOLoop.current().start()
