Rockchip NPU 推理运行时库（Linux aarch64）
==========================================

- librkllmrt.so  — RKLLM 语言模型推理运行时，版本 v1.3.0
- librknnrt.so   — RKNN 通用模型推理运行时

来源：https://github.com/airockchip/rknn-llm （release v1.3.0，rkllm-runtime/Linux/librkllm_api/aarch64 与 rknpu driver 附带的 librknnrt.so）

版权归 Rockchip Electronics Co., Ltd. 所有，按其随发布包附带的许可条款再分发
（BSD 风格：允许源码/二进制形式的再分发，须保留版权声明）。

安装方式（二选一）：

  1. 复制到系统目录（默认搜索路径）：

     sudo cp lib/*.so /usr/local/lib/
     sudo ldconfig

  2. 或通过环境变量指定库路径，无需复制：

     export RKLLM_LIB=$(pwd)/lib/librkllmrt.so
     export RKNN_LIB=$(pwd)/lib/librknnrt.so

注意：librkllmrt.so v1.3.0 对应 rkllm_model_zoo v1.3.0 转出的模型；
不同版本的运行时与模型混用可能导致加载失败。
