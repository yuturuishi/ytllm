# -*- coding: utf-8 -*-
"""推理实例的进程生命周期管理。

ytllm 采用「一个模型服务 = 一个引擎进程」的模型：

    ytllm 管理面 (Tornado, 常驻 9697)
      ├── rkllm_engine.py --model-path A.rkllm --port 9701   ← 模型 A 的推理进程
      └── rkllm_engine.py --model-path B.rkllm --port 9702   ← 模型 B 的推理进程

每个引擎进程对外暴露 OpenAI 兼容接口（/v1/chat/completions 等）。本模块只负责
进程侧的活：分配端口、拉起、等待就绪、停止、判定状态、读日志。
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil

READY_TIMEOUT = 300   # 模型加载 + 就绪的最长等待（秒），4B 模型首次加载通常 10~60s
STOP_TIMEOUT = 15     # 优雅停止等待（秒），超时强杀
POLL_INTERVAL = 0.5

# 绕过环境里的 http_proxy：健康检查打的是本机回环地址，走代理必然失败
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class EngineError(Exception):
    """引擎进程的可展示错误：message 会直接呈现在前端。"""


def _is_windows():
    return os.name == 'nt'


class EngineManager:
    def __init__(self, project_root, log_dir):
        self.root = Path(project_root)
        self.script = self.root / 'engine' / 'rkllm_engine.py'
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 基础 ----------

    def log_path(self, service_id):
        return self.log_dir / f'{service_id}.log'

    def read_log(self, service_id, lines=300):
        path = self.log_path(service_id)
        if not path.exists():
            return []
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return [ln.rstrip() for ln in f.readlines()[-lines:]]

    def clear_log(self, service_id):
        self.log_path(service_id).write_bytes(b'')

    def find_free_port(self, used=(), start=9700, end=9799):
        used = {int(p) for p in used if p}
        for port in range(start, end + 1):
            if port in used:
                continue
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('127.0.0.1', port))
                except OSError:
                    continue
            return port
        raise EngineError(f'没有可用端口（{start}-{end} 已占满），请先停止一些服务')

    def pid_alive(self, pid):
        """存活判断，并确认这个 PID 确实是我们的引擎进程（防止 PID 被系统复用）。"""
        if not pid:
            return False
        try:
            proc = psutil.Process(int(pid))
            cmdline = proc.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, ValueError):
            return False
        return any('rkllm_engine.py' in part for part in cmdline)

    def health_ok(self, port, timeout=3):
        """引擎就绪判断：/health 返回 model_ready=true。"""
        try:
            with _NO_PROXY_OPENER.open(f'http://127.0.0.1:{int(port)}/health', timeout=timeout) as resp:
                payload = json.loads(resp.read().decode('utf-8'))
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
            return False
        return payload.get('model_ready') is True

    # ---------- 启动 ----------

    def build_command(self, svc):
        params = json.loads(svc['params'] or '{}') if isinstance(svc['params'], str) else (svc['params'] or {})
        cmd = [
            sys.executable, '-u', str(self.script),
            '--model-path', svc['model_path'],
            '--model-name', svc['model_id'] or 'default',
            '--platform', svc['platform'] or 'rk3576',
            '--rknn-cores', str(svc['rknn_cores'] or 2),
            '--host', '127.0.0.1',
            '--port', str(svc['port']),
            '--log-file', str(self.log_path(svc['id'])),
        ]
        if svc['vision_model_path']:
            cmd += ['--vision-model', svc['vision_model_path']]
        for flag, key in (
            ('--max-context', 'max_context'),
            ('--max-tokens', 'max_tokens'),
            ('--top-k', 'top_k'),
            ('--top-p', 'top_p'),
            ('--temperature', 'temperature'),
            ('--repeat-penalty', 'repeat_penalty'),
        ):
            value = params.get(key)
            if value is not None:
                cmd += [flag, str(value)]
        return cmd

    def start(self, svc):
        """拉起引擎进程并等待就绪，返回 PID；失败抛 EngineError（附日志尾部原因）。"""
        if self.pid_alive(svc['pid']):
            return svc['pid']

        log_path = self.log_path(svc['id'])
        self.clear_log(svc['id'])   # 清掉上次的日志，避免旧报错误导判断

        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        log_file = open(log_path, 'ab')
        kwargs = dict(stdout=log_file, stderr=subprocess.STDOUT,
                      cwd=str(self.root), env=env)
        if _is_windows():
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        else:
            kwargs['start_new_session'] = True   # 独立进程组，方便整组回收

        cmd = self.build_command(svc)
        try:
            proc = subprocess.Popen(cmd, **kwargs)
        except OSError as e:
            raise EngineError(f'无法启动引擎进程: {e}')
        finally:
            log_file.close()

        deadline = time.time() + READY_TIMEOUT
        while time.time() < deadline:
            if proc.poll() is not None:
                raise EngineError(self.failure_hint(svc['id'], proc.returncode))
            if self.health_ok(svc['port']):
                return proc.pid
            time.sleep(POLL_INTERVAL)

        self.kill(proc.pid)
        raise EngineError(f'启动超时（{READY_TIMEOUT}s）：模型未在预期时间内就绪，详见日志')

    def failure_hint(self, service_id, returncode):
        """进程提前退出时，从日志里捞出最有信息量的几行。"""
        tail = [ln for ln in self.read_log(service_id, 60) if ln.strip()]
        detail = '\n'.join(tail[-6:]) if tail else '(日志为空)'
        return f'引擎进程已退出（退出码 {returncode}）：\n{detail}'

    # ---------- 停止 ----------

    def _signal(self, pid, sig):
        try:
            if _is_windows():
                os.kill(int(pid), sig)
            else:
                os.killpg(os.getpgid(int(pid)), sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def kill(self, pid):
        self._signal(pid, signal.SIGTERM)
        deadline = time.time() + STOP_TIMEOUT
        while time.time() < deadline:
            if not self.pid_alive(pid):
                return
            time.sleep(0.3)
        self._signal(pid, signal.SIGKILL)

    def stop(self, svc):
        """停止实例，返回 True 表示进程确已退出。"""
        pid = svc['pid']
        if not self.pid_alive(pid):
            return True
        self.kill(pid)
        return not self.pid_alive(pid)
