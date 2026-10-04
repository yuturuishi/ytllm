# -*- coding: utf-8 -*-
"""模型仓库：管理 models/ 目录下的 RKLLM / RKNN 模型文件。

大模型动辄数百 MB 到数 GB，因此上传走「分片」：前端把文件切成若干片按序 POST，
服务端逐片追加到临时文件，最后一片到达后原子改名落盘。整个过程服务端只持有单片
大小的内存，也不存在断线重传要重头再来的问题。
"""

import os
import re
import shutil
import time
from pathlib import Path

MODEL_SUFFIXES = ('.rkllm', '.rknn')
_SAFE_NAME = re.compile(r'^[\w.\-\u4e00-\u9fa5 ]+$')
_SAFE_ID = re.compile(r'^[A-Za-z0-9_-]{1,64}$')


class ModelStoreError(Exception):
    """模型仓库的输入/状态错误，message 可直接展示给用户。"""


def human_size(num):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if num < 1024 or unit == 'TB':
            return f'{num:.0f} {unit}' if unit == 'B' else f'{num:.2f} {unit}'
        num /= 1024


def safe_name(name):
    """校验模型文件名：去掉目录部分，拒绝路径穿越与非法字符。"""
    name = os.path.basename((name or '').strip())
    if not name or name.startswith('.') or not _SAFE_NAME.match(name):
        raise ModelStoreError('文件名非法（仅允许字母、数字、汉字、空格、点、下划线与连字符）')
    if Path(name).suffix.lower() not in MODEL_SUFFIXES:
        raise ModelStoreError('仅支持 .rkllm / .rknn 模型文件')
    return name


class ModelStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = self.root / '.upload'
        self.tmp_dir.mkdir(exist_ok=True)

    # ---------- 查询 ----------

    def list_models(self):
        items = []
        for p in self.root.iterdir():
            if not p.is_file() or p.suffix.lower() not in MODEL_SUFFIXES:
                continue
            st = p.stat()
            items.append({
                'name': p.name,
                'size': st.st_size,
                'size_human': human_size(st.st_size),
                'suffix': p.suffix.lower()[1:],
                'mtime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime)),
            })
        items.sort(key=lambda x: x['name'])
        return items

    def exists(self, name):
        try:
            return (self.root / safe_name(name)).is_file()
        except ModelStoreError:
            return False

    def resolve(self, name):
        """模型名 -> 绝对路径（供引擎进程使用）。"""
        safe = safe_name(name)
        path = self.root / safe
        if not path.is_file():
            raise ModelStoreError(f'模型文件不存在: {safe}')
        return str(path.resolve())

    def delete(self, name):
        path = self.root / safe_name(name)
        if not path.is_file():
            raise ModelStoreError('模型文件不存在')
        path.unlink()

    def disk_usage(self):
        models_bytes = sum(
            p.stat().st_size for p in self.root.iterdir()
            if p.is_file() and p.suffix.lower() in MODEL_SUFFIXES
        )
        usage = shutil.disk_usage(str(self.root))
        return {
            'models_bytes': models_bytes,
            'models_human': human_size(models_bytes),
            'disk_total': human_size(usage.total),
            'disk_free': human_size(usage.free),
            'disk_percent': round(usage.used / usage.total * 100, 1) if usage.total else 0,
        }

    # ---------- 上传 ----------

    def write_chunk(self, upload_id, filename, index, total, data):
        """写入一个分片。返回 {'done': bool, ...}，最后一片落盘时给出最终文件名。"""
        if not _SAFE_ID.match(upload_id or ''):
            raise ModelStoreError('上传会话标识非法')
        name = safe_name(filename)
        try:
            index, total = int(index), int(total)
        except (TypeError, ValueError):
            raise ModelStoreError('分片参数非法')
        if total < 1 or index < 0 or index >= total:
            raise ModelStoreError('分片序号越界')
        if not data:
            raise ModelStoreError('分片内容为空')

        part = self.tmp_dir / f'{upload_id}.part'
        if index == 0 and part.exists():
            part.unlink()
        if index > 0 and not part.exists():
            raise ModelStoreError('上传会话已失效，请重新上传')

        with open(part, 'ab') as f:
            f.write(data)

        received = part.stat().st_size
        if index + 1 < total:
            return {'done': False, 'received': received, 'received_human': human_size(received)}

        target = self.root / name
        shutil.move(str(part), str(target))
        return {
            'done': True,
            'name': name,
            'size': target.stat().st_size,
            'size_human': human_size(target.stat().st_size),
        }

    def abort_upload(self, upload_id):
        if not _SAFE_ID.match(upload_id or ''):
            raise ModelStoreError('上传会话标识非法')
        part = self.tmp_dir / f'{upload_id}.part'
        if part.exists():
            part.unlink()
