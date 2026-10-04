#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ytllm RKLLM 推理引擎（OpenAI 兼容 API）worker
=============================================

本文件是 ytllm 管理平台的推理后端，由 ytllm 按「一个模型服务 = 一个进程」的方式拉起，
通常无需手动运行。手动调试时可直接执行下面的命令。

安装依赖:
    pip install flask numpy opencv-python-headless pyyaml

运行:
    python engine/rkllm_engine.py [选项]

示例:
    # 使用默认配置
    python engine/rkllm_engine.py
    
    # 指定模型和端口
    python engine/rkllm_engine.py --model-path /path/to/model.rkllm --port 9696
    
    # 使用配置文件
    python engine/rkllm_engine.py --config config.yaml
    
    # 使用环境变量指定动态库
    RKLLM_LIB=/opt/rkllm/librkllmrt.so RKNN_LIB=/opt/rknn/librknnrt.so \
        python engine/rkllm_engine.py
    
    # 完整参数示例
    python engine/rkllm_engine.py \
        --model-path /data/models/qwen.rkllm \
        --vision-model /data/models/vision.rknn \
        --model-name qwen3-vl-4b \
        --rkllm-lib /usr/local/lib/librkllmrt.so \
        --rknn-lib /usr/local/lib/librknnrt.so \
        --platform rk3576 \
        --rknn-cores 2 \
        --host 0.0.0.0 \
        --port 9696 \
        --max-context 4096 \
        --max-tokens 512 \
        --top-k 1 \
        --top-p 0.9 \
        --temperature 0.8 \
        --repeat-penalty 1.1

依赖库:
    - flask>=3.0.0      # Web 框架
    - numpy>=2.0.0      # 数值计算
    - opencv-python-headless  # 图像处理 (可选,仅多模态需要)
    - pyyaml>=6.0       # 配置文件解析

硬件要求:
    - Rockchip RK3576/RK3588 开发板
    - RKLLM 运行时库: /usr/local/lib/librkllmrt.so
    - RKNN 运行时库: /usr/local/lib/librknnrt.so

API 接口:
    GET  /health                    # 健康检查
    GET  /v1/models                 # 列出可用模型
    POST /v1/chat/completions       # 聊天补全 (支持流式)

Author: ytllm
"""

import os
import sys
import ctypes
import logging
import queue
import threading
import json
import uuid
import time
import base64
import argparse
from typing import Optional, Generator

import numpy as np
import yaml

from flask import Flask, request, jsonify, Response

# ============================================================================
# 命令行参数解析
# ============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="ytllm RKLLM 推理引擎（OpenAI 兼容 API）器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --model-path /path/to/model.rkllm --port 8080
  %(prog)s --config config.yaml
  %(prog)s --help
        """
    )
    
    # 模型配置
    parser.add_argument("--model-path", 
                       default="/data/rkllm_model_zoo/1.2.3/RK3576/Qwen3-VL-4B/qwen3-vl-4b-instruct_w4a16_g128_rk3576.rkllm",
                       help="RKLLM 模型路径 (默认: /data/rkllm_model_zoo/.../qwen3-vl-4b.rkllm)")
    parser.add_argument("--vision-model", 
                       default="/data/rkllm_model_zoo/1.2.3/RK3576/Qwen3-VL-4B/qwen3-vl-4b_vision_rk3576.rknn",
                       help="Vision 模型路径 (默认: /data/rkllm_model_zoo/.../vision.rknn)")
    parser.add_argument("--model-name", 
                       default="qwen3-vl-4b",
                       help="模型名称 (默认: qwen3-vl-4b)")
    
    # 运行时库 (优先从环境变量读取)
    parser.add_argument("--rkllm-lib", 
                       default=os.environ.get("RKLLM_LIB", "/usr/local/lib/librkllmrt.so"),
                       help="RKLLM 运行时库路径 (默认: 环境变量 RKLLM_LIB 或 /usr/local/lib/librkllmrt.so)")
    parser.add_argument("--rknn-lib", 
                       default=os.environ.get("RKNN_LIB", "/usr/local/lib/librknnrt.so"),
                       help="RKNN 运行时库路径 (默认: 环境变量 RKNN_LIB 或 /usr/local/lib/librknnrt.so)")
    
    # 平台配置
    parser.add_argument("--platform", 
                       default="rk3576",
                       choices=["rk3576", "rk3588"],
                       help="目标平台 (默认: rk3576)")
    parser.add_argument("--rknn-cores", 
                       type=int,
                       default=2,
                       choices=[1, 2, 3],
                       help="RKNN 使用的 NPU 核心数 (默认: 2)")
    
    # 服务配置
    parser.add_argument("--host", 
                       default="0.0.0.0",
                       help="服务器监听地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", 
                       type=int,
                       default=9696,
                       help="服务器监听端口 (默认: 9696)")
    
    # 推理参数
    parser.add_argument("--max-context", 
                       type=int,
                       default=4096,
                       help="最大上下文长度 (默认: 4096)")
    parser.add_argument("--max-tokens", 
                       type=int,
                       default=512,
                       help="最大生成 token 数 (默认: 512)")
    parser.add_argument("--top-k", 
                       type=int,
                       default=1,
                       help="Top-K 采样参数 (默认: 1)")
    parser.add_argument("--top-p", 
                       type=float,
                       default=0.9,
                       help="Top-P 采样参数 (默认: 0.9)")
    parser.add_argument("--temperature", 
                       type=float,
                       default=0.8,
                       help="温度参数 (默认: 0.8)")
    parser.add_argument("--repeat-penalty", 
                       type=float,
                       default=1.1,
                       help="重复惩罚参数 (默认: 1.1)")
    
    # 多模态配置
    parser.add_argument("--img-start", 
                       default="<|vision_start|>",
                       help="图片开始标记 (默认: <|vision_start|>)")
    parser.add_argument("--img-end", 
                       default="<|vision_end|>",
                       help="图片结束标记 (默认: <|vision_end|>)")
    parser.add_argument("--img-content", 
                       default="<|image_pad|>",
                       help="图片内容标记 (默认: <|image_pad|>，与 rknn-llm 官方示例一致)")
    
    # 配置文件
    parser.add_argument("--config", 
                       default=None,
                       help="YAML 配置文件路径 (优先级高于命令行参数)")
    
    # 日志
    parser.add_argument("--log-file",
                       default=None,
                       help="日志文件路径 (默认: <项目>/logs/engine.log)")

    return parser.parse_args()

# 解析参数
args = parse_args()

# 如果指定了配置文件，加载并覆盖参数
if args.config and os.path.exists(args.config):
    with open(args.config, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}
        for key, value in config_data.items():
            key_lower = key.lower().replace("_", "-")
            if hasattr(args, key_lower.replace("-", "_")):
                setattr(args, key_lower.replace("-", "_"), value)

# ============================================================================
# 日志配置（控制台 + 文件双输出）
# ============================================================================

_DEFAULT_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "engine.log")
LOG_FILE = os.path.abspath(args.log_file or _DEFAULT_LOG)
LOG_DIR = os.path.dirname(LOG_FILE)

def setup_logging():
    """配置日志系统：同时输出到控制台和文件"""
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger("RKLLM")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # 日志格式
    log_format = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台 Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # 文件 Handler（自动轮转，单文件最大10MB，保留5个备份）
    try:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning("Failed to setup file logging: %s", e)

    return logger

logger = setup_logging()

# ============================================================================
# RKLLM ctypes 定义
# ============================================================================

RKLLM_Handle_t = ctypes.c_void_p
userdata = ctypes.c_void_p(None)


class LLMCallState:
    """LLM Call State Constants"""
    RKLLM_RUN_NORMAL = 0
    RKLLM_RUN_WAITING = 1
    RKLLM_RUN_FINISH = 2
    RKLLM_RUN_ERROR = 3


class RKLLMInputType:
    RKLLM_INPUT_PROMPT = 0
    RKLLM_INPUT_TOKEN = 1
    RKLLM_INPUT_EMBED = 2
    RKLLM_INPUT_MULTIMODAL = 3


class RKLLMInferMode:
    RKLLM_INFER_GENERATE = 0
    RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1
    RKLLM_INFER_GET_LOGITS = 2


class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104)
    ]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("ignore_eos_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("extend_param", RKLLMExtendParam),
    ]


class RKLLMLoraAdapter(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_path", ctypes.c_char_p),
        ("lora_adapter_name", ctypes.c_char_p),
        ("scale", ctypes.c_float)
    ]


class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_name", ctypes.c_char_p)
    ]


class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [
        ("save_prompt_cache", ctypes.c_int),
        ("prompt_cache_path", ctypes.c_char_p)
    ]


class RKLLMSamplingParam(ctypes.Structure):
    _fields_ = [
        ("top_k", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float)
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
        ("sampling_params", ctypes.POINTER(RKLLMSamplingParam)),
        ("keep_history", ctypes.c_int),
        ("max_new_tokens", ctypes.c_int32)
    ]


class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t)
    ]


class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t)
    ]


class RKLLMImageInput(ctypes.Structure):
    """1.3.0: RKLLMMultiModalInput.image 子结构"""
    _fields_ = [
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_start", ctypes.c_char_p),
        ("image_end", ctypes.c_char_p),
        ("image_content", ctypes.c_char_p),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t)
    ]


class RKLLMVideoInput(ctypes.Structure):
    """1.3.0: RKLLMMultiModalInput.video 子结构"""
    _fields_ = [
        ("video_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_frame_tokens", ctypes.c_size_t),
        ("n_frame_per_video", ctypes.c_size_t),
        ("n_video", ctypes.c_size_t),
        ("video_start", ctypes.c_char_p),
        ("video_end", ctypes.c_char_p),
        ("video_content", ctypes.c_char_p),
        ("frame_width", ctypes.c_size_t),
        ("frame_height", ctypes.c_size_t)
    ]


class RKLLMMultiModalInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image", RKLLMImageInput),
        ("video", RKLLMVideoInput)
    ]


class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModalInput)
    ]


class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", ctypes.c_int),
        ("input_data", RKLLMInputUnion)
    ]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int)
    ]


class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int)
    ]


class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [
        ("prefill_time_ms", ctypes.c_float),
        ("prefill_tokens", ctypes.c_int),
        ("generate_time_ms", ctypes.c_float),
        ("generate_tokens", ctypes.c_int),
        ("memory_usage_mb", ctypes.c_float)
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
        ("perf", RKLLMPerfStat)
    ]


callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)

# 1.3.0 的 rkllm_init 第三个参数是 RKLLMCallback*（结构体指针），不是裸的 result 函数指针。
# 传裸指针时 init 仍能返回 0，但运行时回调会被当成结构体的第一个字段去解引用 → 推理瞬间段错误。
tokenizer_callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p,
                                           ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32)
embed_callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                                       ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64)


class RKLLMCallback(ctypes.Structure):
    """1.3.0: rkllm.h 里的 RKLLMCallback 结构体"""
    _fields_ = [
        ("result_callback", callback_type),
        ("result_userdata", ctypes.c_void_p),
        ("tokenizer_callback", tokenizer_callback_type),
        ("tokenizer_userdata", ctypes.c_void_p),
        ("embed_callback", embed_callback_type),
        ("embed_userdata", ctypes.c_void_p),
    ]

# ============================================================================
# RKNN ctypes 定义
# ============================================================================

if ctypes.sizeof(ctypes.c_void_p) == 8:
    rknn_context = ctypes.c_uint64
else:
    rknn_context = ctypes.c_uint32

RKNN_SUCC = 0
RKNN_MAX_DIMS = 16
RKNN_MAX_NAME_LEN = 256


class RKNNQueryCmd:
    RKNN_QUERY_IN_OUT_NUM = 0
    RKNN_QUERY_INPUT_ATTR = 1
    RKNN_QUERY_OUTPUT_ATTR = 2
    RKNN_QUERY_SDK_VERSION = 5


class RKNNTensorType:
    RKNN_TENSOR_FLOAT32 = 0
    RKNN_TENSOR_FLOAT16 = 1
    RKNN_TENSOR_INT8 = 2
    RKNN_TENSOR_UINT8 = 3


class RKNNTensorFormat:
    RKNN_TENSOR_NCHW = 0
    RKNN_TENSOR_NHWC = 1


class RKNNCoreMask:
    RKNN_NPU_CORE_AUTO = 0
    RKNN_NPU_CORE_0 = 1
    RKNN_NPU_CORE_1 = 2
    RKNN_NPU_CORE_2 = 4
    RKNN_NPU_CORE_0_1 = 3
    RKNN_NPU_CORE_0_1_2 = 7


class RKNNInputOutputNum(ctypes.Structure):
    _fields_ = [
        ("n_input", ctypes.c_uint32),
        ("n_output", ctypes.c_uint32)
    ]


class RKNNTensorAttr(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * RKNN_MAX_DIMS),
        ("name", ctypes.c_char * RKNN_MAX_NAME_LEN),
        ("n_elems", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_int),
        ("type", ctypes.c_int),
        ("qnt_type", ctypes.c_int),
        ("fl", ctypes.c_int8),
        ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("h_stride", ctypes.c_uint32)
    ]


class RKNNInput(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("type", ctypes.c_int),
        ("fmt", ctypes.c_int)
    ]


class RKNNOutput(ctypes.Structure):
    _fields_ = [
        ("want_float", ctypes.c_uint8),
        ("is_prealloc", ctypes.c_uint8),
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32)
    ]


# ============================================================================
# 加载 RKLLM 库
# ============================================================================

rkllm_lib = None
try:
    lib_path = os.path.expanduser(args.rkllm_lib)
    if os.path.exists(lib_path):
        rkllm_lib = ctypes.CDLL(lib_path)
        logger.info("RKLLM library loaded: %s", lib_path)
    else:
        logger.warning("RKLLM library not found at %s", lib_path)
except Exception as e:
    logger.error("Error loading RKLLM library: %s", e)

# ============================================================================
# 加载 RKNN 库
# ============================================================================

rknn_lib = None
try:
    lib_path = os.path.expanduser(args.rknn_lib)
    if os.path.exists(lib_path):
        rknn_lib = ctypes.CDLL(lib_path)
        logger.info("RKNN library loaded: %s", lib_path)

        # 设置 RKNN 函数签名
        rknn_lib.rknn_init.argtypes = [ctypes.POINTER(rknn_context), ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
        rknn_lib.rknn_init.restype = ctypes.c_int
        rknn_lib.rknn_destroy.argtypes = [rknn_context]
        rknn_lib.rknn_destroy.restype = ctypes.c_int
        rknn_lib.rknn_query.argtypes = [rknn_context, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        rknn_lib.rknn_query.restype = ctypes.c_int
        rknn_lib.rknn_inputs_set.argtypes = [rknn_context, ctypes.c_uint32, ctypes.POINTER(RKNNInput)]
        rknn_lib.rknn_inputs_set.restype = ctypes.c_int
        rknn_lib.rknn_run.argtypes = [rknn_context, ctypes.c_void_p]
        rknn_lib.rknn_run.restype = ctypes.c_int
        rknn_lib.rknn_outputs_get.argtypes = [rknn_context, ctypes.c_uint32, ctypes.POINTER(RKNNOutput), ctypes.c_void_p]
        rknn_lib.rknn_outputs_get.restype = ctypes.c_int
        rknn_lib.rknn_outputs_release.argtypes = [rknn_context, ctypes.c_uint32, ctypes.POINTER(RKNNOutput)]
        rknn_lib.rknn_outputs_release.restype = ctypes.c_int
        rknn_lib.rknn_set_core_mask.argtypes = [rknn_context, ctypes.c_int]
        rknn_lib.rknn_set_core_mask.restype = ctypes.c_int
    else:
        logger.warning("RKNN library not found at %s", lib_path)
except Exception as e:
    logger.error("Error loading RKNN library: %s", e)


# ============================================================================
# RKLLM 模型封装
# ============================================================================

class RKLLMModel:
    """RKLLM 模型封装"""

    def __init__(self, model_path: str, platform: str = "rk3588", callback_func=None):
        if rkllm_lib is None:
            raise RuntimeError("RKLLM library not loaded")

        self._user_callback = callback_func
        if callback_func:
            def wrapped_callback(result, userdata, state):
                try:
                    callback_func(result, userdata, state)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
                return 0
            self.callback = callback_type(wrapped_callback)
        else:
            self.callback = callback_type(0)

        # 1.3.0: 必须把回调装进 RKLLMCallback 结构体再传指针
        self.callback_struct = RKLLMCallback()
        self.callback_struct.result_callback = self.callback
        self.callback_struct.result_userdata = None
        self.callback_struct.tokenizer_callback = tokenizer_callback_type(0)
        self.callback_struct.tokenizer_userdata = None
        self.callback_struct.embed_callback = embed_callback_type(0)
        self.callback_struct.embed_userdata = None

        # 构建参数
        rkllm_param = RKLLMParam()
        rkllm_param.model_path = bytes(os.path.expanduser(model_path), "utf-8")
        rkllm_param.max_context_len = args.max_context
        rkllm_param.max_new_tokens = args.max_tokens
        rkllm_param.skip_special_token = True
        rkllm_param.n_keep = -1
        rkllm_param.top_k = args.top_k
        rkllm_param.top_p = args.top_p
        rkllm_param.temperature = args.temperature
        rkllm_param.repeat_penalty = args.repeat_penalty
        rkllm_param.frequency_penalty = 0.0
        rkllm_param.presence_penalty = 0.0
        rkllm_param.mirostat = 0
        rkllm_param.mirostat_tau = 5.0
        rkllm_param.mirostat_eta = 0.1
        rkllm_param.is_async = False

        # 1.3.0: 图片起止标签已迁到 RKLLMMultiModalInput.image，初始化这里只设 base_domain_id
        rkllm_param.ignore_eos_token = False
        rkllm_param.extend_param.base_domain_id = 1 if args.vision_model else 0

        rkllm_param.extend_param.embed_flash = 1
        rkllm_param.extend_param.n_batch = 1
        rkllm_param.extend_param.use_cross_attn = 0
        rkllm_param.extend_param.enabled_cpus_num = 4

        if platform.lower() in ["rk3576", "rk3588"]:
            rkllm_param.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)
        else:
            rkllm_param.extend_param.enabled_cpus_mask = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)

        self.handle = RKLLM_Handle_t()

        rkllm_lib.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam),
                                         ctypes.POINTER(RKLLMCallback)]
        rkllm_lib.rkllm_init.restype = ctypes.c_int

        ret = rkllm_lib.rkllm_init(ctypes.byref(self.handle), ctypes.byref(rkllm_param),
                                   ctypes.byref(self.callback_struct))
        if ret != 0:
            raise RuntimeError(f"rkllm init failed with ret={ret}")

        rkllm_lib.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        rkllm_lib.rkllm_run.restype = ctypes.c_int

        rkllm_lib.rkllm_set_function_tools.argtypes = [RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        rkllm_lib.rkllm_set_function_tools.restype = ctypes.c_int

        rkllm_lib.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        rkllm_lib.rkllm_destroy.restype = ctypes.c_int

        self.rkllm_lib = rkllm_lib

        # 推理参数
        self.rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(self.rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        self.rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        self.rkllm_infer_params.lora_params = None
        self.rkllm_infer_params.keep_history = 0

    def set_function_tools(self, system_prompt: str, tools: str = "[]", tool_response_str: str = ""):
        """设置函数工具"""
        self.rkllm_lib.rkllm_set_function_tools(
            self.handle,
            ctypes.c_char_p(system_prompt.encode("utf-8")),
            ctypes.c_char_p(tools.encode("utf-8")),
            ctypes.c_char_p(tool_response_str.encode("utf-8"))
        )

    def run(self, role: str, enable_thinking: bool, prompt, multimodal_data=None):
        """运行推理
        
        Args:
            role: 角色
            enable_thinking: 是否启用思考
            prompt: 提示词 str
            multimodal_data: 多模态数据 dict, 包含 embeddings, n_image_tokens, image_width, image_height
        """
        rkllm_input = RKLLMInput()
        rkllm_input.role = role.encode("utf-8") if role else "user".encode("utf-8")
        rkllm_input.enable_thinking = ctypes.c_bool(enable_thinking)

        if multimodal_data:
            # 多模态输入 - 使用 RKLLM_INPUT_MULTIMODAL
            rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_MULTIMODAL

            embeddings = multimodal_data["embeddings"]
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings, dtype=np.float32)

            # 保持引用避免GC
            self._embeddings = embeddings

            # 1.3.0: 图片 embed 与起止标签都在 multimodal_input.image 子结构里
            mm = rkllm_input.input_data.multimodal_input
            mm.prompt = ctypes.c_char_p(prompt.encode("utf-8"))
            mm.image.image_embed = embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            mm.image.n_image_tokens = multimodal_data["n_image_tokens"]
            mm.image.n_image = multimodal_data.get("n_image", 1)
            mm.image.image_start = ctypes.c_char_p((args.img_start or "").encode("utf-8"))
            mm.image.image_end = ctypes.c_char_p((args.img_end or "").encode("utf-8"))
            mm.image.image_content = ctypes.c_char_p((args.img_content or "").encode("utf-8"))
            mm.image.image_width = multimodal_data["image_width"]
            mm.image.image_height = multimodal_data["image_height"]
        else:
            # 纯文本输入
            rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
            rkllm_input.input_data.prompt_input = ctypes.c_char_p(prompt.encode("utf-8"))

        self.rkllm_lib.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(self.rkllm_infer_params), None)

    def abort(self):
        """中止推理"""
        self.rkllm_lib.rkllm_abort(self.handle)

    def release(self):
        """释放资源"""
        self.rkllm_lib.rkllm_destroy(self.handle)


# ============================================================================
# RKNN 模型封装
# ============================================================================

class RKNNModel:
    """RKNN 模型封装"""

    def __init__(self, model_path: str, core_num: int = 3):
        if rknn_lib is None:
            raise RuntimeError("RKNN library not loaded")

        self.ctx = rknn_context(0)
        ret = rknn_lib.rknn_init(ctypes.byref(self.ctx), model_path.encode("utf-8"), 0, 0, None)
        if ret != RKNN_SUCC:
            raise RuntimeError(f"rknn_init failed with ret={ret}")

        # 设置 NPU 核心
        core_mask = RKNNCoreMask.RKNN_NPU_CORE_AUTO
        if core_num == 1:
            core_mask = RKNNCoreMask.RKNN_NPU_CORE_0
        elif core_num == 2:
            core_mask = RKNNCoreMask.RKNN_NPU_CORE_0_1
        elif core_num >= 3:
            core_mask = RKNNCoreMask.RKNN_NPU_CORE_0_1_2

        ret = rknn_lib.rknn_set_core_mask(self.ctx, core_mask)
        if ret != RKNN_SUCC:
            logger.warning("Failed to set NPU core mask, using AUTO mode")

        # 查询 IO 信息
        self.io_num = RKNNInputOutputNum()
        rknn_lib.rknn_query(self.ctx, RKNNQueryCmd.RKNN_QUERY_IN_OUT_NUM, ctypes.byref(self.io_num), ctypes.sizeof(self.io_num))

        # 查询输入属性
        self.input_attrs = []
        for i in range(self.io_num.n_input):
            attr = RKNNTensorAttr()
            attr.index = i
            rknn_lib.rknn_query(self.ctx, RKNNQueryCmd.RKNN_QUERY_INPUT_ATTR, ctypes.byref(attr), ctypes.sizeof(attr))
            self.input_attrs.append(attr)

        # 查询输出属性
        self.output_attrs = []
        for i in range(self.io_num.n_output):
            attr = RKNNTensorAttr()
            attr.index = i
            rknn_lib.rknn_query(self.ctx, RKNNQueryCmd.RKNN_QUERY_OUTPUT_ATTR, ctypes.byref(attr), ctypes.sizeof(attr))
            self.output_attrs.append(attr)

        # 获取模型尺寸
        if self.input_attrs[0].fmt == RKNNTensorFormat.RKNN_TENSOR_NCHW:
            self.model_channel = self.input_attrs[0].dims[1]
            self.model_height = self.input_attrs[0].dims[2]
            self.model_width = self.input_attrs[0].dims[3]
        else:
            self.model_height = self.input_attrs[0].dims[1]
            self.model_width = self.input_attrs[0].dims[2]
            self.model_channel = self.input_attrs[0].dims[3]

        # 获取嵌入信息
        self.model_image_token = 0
        self.model_embed_size = 0
        for i in range(4):
            if self.output_attrs[0].dims[i] > 1:
                self.model_image_token = self.output_attrs[0].dims[i]
                self.model_embed_size = self.output_attrs[0].dims[i + 1]
                break

        logger.info("RKNN model loaded: %dx%d, %d tokens, embed_size=%d",
                    self.model_width, self.model_height, self.model_image_token, self.model_embed_size)

    def run(self, img_data):
        """运行 RKNN 推理"""
        inputs = (RKNNInput * 1)()
        inputs[0].index = 0
        inputs[0].type = RKNNTensorType.RKNN_TENSOR_UINT8
        inputs[0].fmt = RKNNTensorFormat.RKNN_TENSOR_NHWC
        inputs[0].size = self.model_width * self.model_height * self.model_channel
        inputs[0].buf = ctypes.cast(img_data, ctypes.c_void_p)

        ret = rknn_lib.rknn_inputs_set(self.ctx, 1, inputs)
        if ret != RKNN_SUCC:
            raise RuntimeError(f"rknn_inputs_set failed with ret={ret}")

        ret = rknn_lib.rknn_run(self.ctx, None)
        if ret != RKNN_SUCC:
            raise RuntimeError(f"rknn_run failed with ret={ret}")

        outputs = (RKNNOutput * self.io_num.n_output)()
        for i in range(self.io_num.n_output):
            outputs[i].index = i
            outputs[i].want_float = 1
            outputs[i].is_prealloc = 0

        ret = rknn_lib.rknn_outputs_get(self.ctx, self.io_num.n_output, outputs, None)
        if ret != RKNN_SUCC:
            raise RuntimeError(f"rknn_outputs_get failed with ret={ret}")

        # 后处理
        res = []
        if self.io_num.n_output == 1:
            data = ctypes.string_at(outputs[0].buf, outputs[0].size)
            res = np.frombuffer(data, dtype=np.float32).copy()
        else:
            output_arrays = []
            for i in range(self.io_num.n_output):
                data = ctypes.string_at(outputs[i].buf, outputs[i].size)
                output_arrays.append(np.frombuffer(data, dtype=np.float32).reshape(self.model_image_token, self.model_embed_size))
            stacked = np.stack(output_arrays, axis=1)
            res = stacked.flatten().copy()

        rknn_lib.rknn_outputs_release(self.ctx, self.io_num.n_output, outputs)
        return res

    def release(self):
        """释放资源"""
        if self.ctx:
            rknn_lib.rknn_destroy(self.ctx)
            self.ctx = None


# ============================================================================
# 聊天服务
# ============================================================================

class ChatService:
    """聊天服务单例"""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.rkllm_model: Optional[RKLLMModel] = None
        self.rknn_model: Optional[RKNNModel] = None
        self.output_queue = queue.Queue()
        self._is_blocking = False
        self._token_count = 0  # 当前请求 Token 计数器
        self._request_start_time = 0  # 当前请求开始时间

        # ===== 全局统计 =====
        self._stats_lock = threading.Lock()
        self._total_requests = 0          # 总请求数
        self._total_tokens = 0            # 总 Token 数
        self._total_errors = 0            # 总错误数
        self._start_time = time.time()    # 服务启动时间
        self._current_request_id = None   # 当前请求ID
        self._current_request_prompt = "" # 当前请求提示词(截断)
        self._current_request_start = 0   # 当前请求开始时间
        self._is_processing = False       # 是否正在处理
        self._recent_logs = []            # 最近的推理日志(环形缓冲)
        self._max_recent_logs = 100       # 最大日志条数

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _callback(self, result, userdata, state):
        """RKLLM 推理回调"""
        if state == LLMCallState.RKLLM_RUN_NORMAL:
            text = result.contents.text.decode("utf-8") if result.contents.text else ""
            token_id = result.contents.token_id
            self._token_count += 1  # 统计生成的 token
            self.output_queue.put(("text", text))
        elif state == LLMCallState.RKLLM_RUN_FINISH:
            self.output_queue.put(("finish", None))
        elif state == LLMCallState.RKLLM_RUN_ERROR:
            self.output_queue.put(("error", None))
        return 0

    def initialize(self):
        """初始化模型"""
        logger.info("Loading RKLLM model from %s...", args.model_path)
        self.rkllm_model = RKLLMModel(
            model_path=args.model_path,
            platform=args.platform,
            callback_func=self._callback
        )
        logger.info("RKLLM model loaded successfully")

        if args.vision_model:
            logger.info("Loading Vision model from %s...", args.vision_model)
            try:
                self.rknn_model = RKNNModel(
                    model_path=args.vision_model,
                    core_num=args.rknn_cores
                )
                logger.info("Vision model loaded successfully")
            except Exception as e:
                logger.error("Failed to load Vision model: %s", e)

    def generate(self, user_prompt, system_prompt: str = "",
                 tools: str = None, enable_thinking: bool = False,
                 image_data: bytes = None) -> Generator[str, None, None]:
        """生成回复
        
        Args:
            user_prompt: 用户输入 (str 或 list 用于多模态)
            system_prompt: 系统提示词
            tools: 函数工具
            enable_thinking: 是否启用思考
            image_data: 图片二进制数据 (可选)
        """
        if self.rkllm_model is None:
            raise RuntimeError("Model not initialized")

        # 排队等待机制 - 如果服务忙，无限等待直到可以处理
        while True:
            with self._lock:
                if not self._is_blocking:
                    self._is_blocking = True
                    break
            logger.info("Server busy, waiting in queue...")
            time.sleep(1)

        # 记录开始时间并重置 token 计数
        self._request_start_time = time.time()
        self._token_count = 0

        # 更新全局统计
        with self._stats_lock:
            self._total_requests += 1
            self._is_processing = True
            self._current_request_start = self._request_start_time
            self._current_request_prompt = (user_prompt or "")[:80]

        logger.info("Starting inference, token counter reset")

        thread = None
        multimodal_data = None
        try:
            self.output_queue = queue.Queue()

            # 如果有图片，使用 Vision Encoder 编码
            if image_data and self.rknn_model:
                try:
                    import cv2
                    # 解码图片
                    nparr = np.frombuffer(image_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        # BGR -> RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # 展开为正方形
                        h, w = img.shape[:2]
                        size = max(w, h)
                        square = np.full((size, size, 3), 127.5, dtype=np.uint8)
                        x_off = (size - w) // 2
                        y_off = (size - h) // 2
                        square[y_off:y_off+h, x_off:x_off+w] = img
                        # 调整大小
                        img_resized = cv2.resize(square, (self.rknn_model.model_width, self.rknn_model.model_height))

                        # RKNN 编码
                        embeddings = self.rknn_model.run(img_resized.tobytes())
                        multimodal_data = {
                            "embeddings": embeddings,
                            "n_image_tokens": self.rknn_model.model_image_token,
                            "image_width": self.rknn_model.model_width,
                            "image_height": self.rknn_model.model_height,
                            "n_image": 1
                        }
                        logger.info("Image encoded: %dx%d, %d tokens",
                                    self.rknn_model.model_width, self.rknn_model.model_height,
                                    self.rknn_model.model_image_token)

                        # 多模态输入的 prompt 必须包含 <image> 标签
                        # 在 user_prompt 前插入 <image> 标签
                        if isinstance(user_prompt, list):
                            text_parts = []
                            for item in user_prompt:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    text_parts.append(item.get("text", ""))
                                elif isinstance(item, str):
                                    text_parts.append(item)
                            user_prompt = "".join(text_parts)
                        user_prompt = "<image>" + user_prompt
                except Exception as e:
                    logger.error("Image encoding failed: %s", e)

            # 设置函数工具
            self.rkllm_model.set_function_tools(
                system_prompt=system_prompt,
                tools=tools if tools else "[]",
                tool_response_str="tool_response"
            )

            # 启动推理线程
            thread = threading.Thread(
                target=self.rkllm_model.run,
                args=("user", enable_thinking, user_prompt, multimodal_data)
            )
            thread.start()

            # 收集输出
            while True:
                try:
                    msg_type, content = self.output_queue.get(timeout=300)
                    if msg_type == "text":
                        yield content
                    elif msg_type == "finish":
                        break
                    elif msg_type == "error":
                        raise RuntimeError("RKLLM Run Error")
                except queue.Empty:
                    logger.warning("Queue timeout, checking thread status...")
                    if not thread.is_alive():
                        # 线程已结束，再检查一下队列中是否有剩余
                        try:
                            msg_type, content = self.output_queue.get_nowait()
                            if msg_type == "text":
                                yield content
                            elif msg_type == "finish":
                                break
                        except queue.Empty:
                            break

        finally:
            # 确保推理线程结束
            if thread and thread.is_alive():
                logger.warning("Aborting inference thread...")
                try:
                    self.rkllm_model.abort()
                    thread.join(timeout=5)
                except Exception as e:
                    logger.error("Failed to abort inference: %s", e)

            # 打印统计信息
            elapsed = time.time() - self._request_start_time
            tokens = self._token_count
            tps = tokens / elapsed if elapsed > 0 else 0
            logger.info("=== Inference Stats ===")
            logger.info("  Completion tokens: %d", tokens)
            logger.info("  Time elapsed: %.2fs", elapsed)
            logger.info("  Tokens/sec: %.2f", tps)
            logger.info("=======================")

            # 更新全局统计
            with self._stats_lock:
                self._total_tokens += tokens
                self._is_processing = False
                self._current_request_id = None
                # 记录到最近日志
                self._recent_logs.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "tokens": tokens,
                    "elapsed": round(elapsed, 2),
                    "tps": round(tps, 2),
                    "prompt": self._current_request_prompt,
                    "has_image": image_data is not None
                })
                if len(self._recent_logs) > self._max_recent_logs:
                    self._recent_logs = self._recent_logs[-self._max_recent_logs:]

            # 确保释放锁，防止死锁
            with self._lock:
                self._is_blocking = False

    def get_stats(self) -> dict:
        """获取服务统计数据"""
        with self._stats_lock:
            uptime = time.time() - self._start_time
            current_elapsed = 0
            if self._is_processing and self._current_request_start > 0:
                current_elapsed = time.time() - self._current_request_start
            # 模型详细信息
            model_info = {
                "name": args.model_name,
                "path": args.model_path,
                "platform": args.platform,
                "quantization": "W4A16 G128" if "w4a16" in args.model_path.lower() else "未知",
                "max_context_len": args.max_context,
                "max_new_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "repeat_penalty": args.repeat_penalty,
            }
            vision_info = None
            if args.vision_model:
                vision_info = {
                    "path": args.vision_model,
                    "core_num": args.rknn_cores,
                    "input_size": f"{self.rknn_model.model_width}x{self.rknn_model.model_height}" if self.rknn_model else "未知",
                    "image_tokens": self.rknn_model.model_image_token if self.rknn_model else 0,
                }
            return {
                "uptime": round(uptime, 1),
                "uptime_str": self._format_uptime(uptime),
                "total_requests": self._total_requests,
                "total_tokens": self._total_tokens,
                "total_errors": self._total_errors,
                "is_processing": self._is_processing,
                "current_prompt": self._current_request_prompt if self._is_processing else None,
                "current_elapsed": round(current_elapsed, 1),
                "model_name": args.model_name,
                "model_ready": self.rkllm_model is not None,
                "vision_ready": self.rknn_model is not None,
                "model_info": model_info,
                "vision_info": vision_info,
                "api_port": args.port,
                "api_key": "sk-ytllm",
            }

    def get_recent_logs(self) -> list:
        """获取最近推理日志"""
        with self._stats_lock:
            return list(self._recent_logs)

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        """格式化运行时间"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")
        return " ".join(parts)

    def release(self):
        """释放资源"""
        if self.rkllm_model:
            self.rkllm_model.release()
        if self.rknn_model:
            self.rknn_model.release()


# 全局单例
chat_service = ChatService.get_instance()

# ============================================================================
# Flask API
# ============================================================================

app = Flask(__name__)


@app.after_request
def after_request(response):
    """添加 CORS 头"""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


@app.errorhandler(Exception)
def handle_exception(e):
    """全局异常处理"""
    logger.error("Unhandled exception: %s", e, exc_info=True)
    return jsonify({
        "error": {
            "message": str(e),
            "type": type(e).__name__
        }
    }), 500


@app.errorhandler(400)
def handle_400(e):
    return jsonify({"error": {"message": str(e), "type": "BadRequest"}}), 400


@app.errorhandler(404)
def handle_404(e):
    return jsonify({"error": {"message": "Not found", "type": "NotFound"}}), 404


@app.route("/", methods=["GET"])
@app.route("/v1", methods=["GET"])
def index():
    """返回首页"""
    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>ytllm - OpenAI API</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; border-bottom: 2px solid #7232f0; padding-bottom: 10px; }}
            .info {{ background: #f3e8ff; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .endpoint {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-left: 4px solid #7232f0; }}
            code {{ background: #e0e0e0; padding: 2px 6px; border-radius: 3px; }}
            .admin-link {{ display: inline-block; margin-top: 20px; padding: 10px 20px; background: #7232f0; color: white; text-decoration: none; border-radius: 5px; }}
        </style>
    </head>
    <body>
    <div class="container">
        <h1>🚀 ytllm</h1>
        <div class="info">
            <strong>模型:</strong> {args.model_name}<br>
            <strong>平台:</strong> {args.platform}<br>
            <strong>端口:</strong> {args.port}
        </div>
        <h2>OpenAI API 端点</h2>
        <div class="endpoint">
            <strong>POST /v1/chat/completions</strong><br>
            聊天补全接口，支持流式输出
        </div>
        <div class="endpoint">
            <strong>GET /v1/models</strong><br>
            获取模型列表
        </div>
        <h2>使用示例</h2>
        <pre style="background:#263238;color:#eee;padding:15px;border-radius:5px;overflow-x:auto;">
from openai import OpenAI

client = OpenAI(
    api_key="sk-ytllm",
    base_url="http://localhost:{args.port}/v1"
)

response = client.chat.completions.create(
    model="{args.model_name}",
    messages=[{{"role": "user", "content": "你好"}}]
)
print(response.choices[0].message.content)</pre>
    </div>
    </body>
    </html>
    '''
    return Response(html, mimetype="text/html; charset=utf-8")


@app.route("/health", methods=["GET"])
def health():
    """健康检查"""
    model_ready = chat_service.rkllm_model is not None
    return jsonify({
        "status": "ok" if model_ready else "initializing",
        "model": args.model_name,
        "model_ready": model_ready
    })


@app.route("/v1/models", methods=["GET"])
def list_models():
    """列出模型"""
    return jsonify({
        "object": "list",
        "data": [{
            "id": args.model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "rkllm"
        }]
    })


@app.route("/v1/chat/completions", methods=["POST", "OPTIONS"])
def chat_completions():
    """聊天补全"""
    # 处理 CORS 预检请求
    if request.method == "OPTIONS":
        return "", 204

    # 请求校验
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": {"message": "Invalid JSON body", "type": "BadRequest"}}), 400

    messages = data.get("messages")
    if not messages:
        return jsonify({"error": {"message": "messages is required", "type": "BadRequest"}}), 400

    stream = data.get("stream", False)
    model_name = data.get("model", args.model_name)
    extra_body = data.get("extra_body", {})
    request_id = f"req-{uuid.uuid4().hex[:8]}"

    logger.info("[%s] New request, stream=%s, model=%s", request_id, stream, model_name)

    # 提取提示词 (支持多模态消息)
    system_prompt = ""
    user_prompt = ""
    image_data = None
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                system_prompt = "".join(text_parts)
            else:
                system_prompt = content
        elif role == "user":
            if isinstance(content, list):
                # 多模态消息
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            # 提取图片 base64 数据
                            url = item.get("image_url", {}).get("url", "")
                            if url.startswith("data:"):
                                # data:image/jpeg;base64,<data>
                                parts = url.split(",", 1)
                                if len(parts) == 2:
                                    image_data = base64.b64decode(parts[1])
                                    logger.info("[%s] Received image: %d bytes", request_id, len(image_data))
                            else:
                                # URL 模式, 尝试下载
                                try:
                                    import urllib.request
                                    resp = urllib.request.urlopen(url, timeout=10)
                                    image_data = resp.read()
                                    logger.info("[%s] Downloaded image: %d bytes", request_id, len(image_data))
                                except Exception as e:
                                    logger.error("[%s] Failed to download image: %s", request_id, e)
                    elif isinstance(item, str):
                        text_parts.append(item)
                user_prompt = "".join(text_parts)
            else:
                user_prompt = content

    if not user_prompt:
        return jsonify({"error": {"message": "user prompt is required", "type": "BadRequest"}}), 400

    tools = extra_body.get("tools")
    enable_thinking = extra_body.get("enable_thinking", False)

    if stream:
        return Response(
            stream_response(user_prompt, system_prompt, tools, enable_thinking, model_name, image_data, request_id),
            mimetype="text/event-stream",
            headers={"X-Request-Id": request_id}
        )
    else:
        return jsonify(non_stream_response(user_prompt, system_prompt, tools, enable_thinking, model_name, image_data, request_id))


def stream_response(user_prompt, system_prompt, tools, enable_thinking, model_name, image_data=None, request_id=""):
    """流式响应"""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    # 发送初始角色 chunk
    first_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": ""},
            "finish_reason": None
        }]
    }
    yield f"data: {json.dumps(first_chunk)}\n\n"

    try:
        for content in chat_service.generate(user_prompt, system_prompt, tools, enable_thinking, image_data):
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        # 发送结束 chunk
        finish_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": chat_service._token_count,
                "total_tokens": chat_service._token_count
            }
        }
        yield f"data: {json.dumps(finish_chunk)}\n\n"
    except Exception as e:
        logger.error("[%s] Stream error: %s", request_id, e, exc_info=True)
        error_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"content": f"\n\n[Error: {str(e)}]"},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"

    yield "data: [DONE]\n\n"


def non_stream_response(user_prompt, system_prompt, tools, enable_thinking, model_name, image_data=None, request_id=""):
    """非流式响应"""
    full_content = ""
    try:
        for content in chat_service.generate(user_prompt, system_prompt, tools, enable_thinking, image_data):
            full_content += content
    except Exception as e:
        logger.error("[%s] Generation error: %s", request_id, e, exc_info=True)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": f"[Error: {str(e)}]"},
                "finish_reason": "error"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": full_content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": chat_service._token_count,
            "total_tokens": chat_service._token_count
        }
    }


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    import signal

    # 优雅关闭
    def shutdown_handler(signum, frame):
        logger.info("Received signal %s, shutting down gracefully...", signum)
        chat_service.release()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    logger.info("Starting ytllm RKLLM engine on %s:%d", args.host, args.port)
    logger.info("Model: %s", args.model_path)
    logger.info("Log file: %s", LOG_FILE)

    # 启动阶段即完成模型加载：失败立刻退出（退出码 3），
    # 父进程据此把日志中的原因直接反馈给用户，而不是拖到首次请求才暴露。
    try:
        chat_service.initialize()
    except Exception as e:
        logger.error("Model initialization failed: %s", e, exc_info=True)
        sys.exit(3)

    logger.info("Engine ready, serving on %s:%d", args.host, args.port)

    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        chat_service.release()
        logger.info("Server stopped.")
