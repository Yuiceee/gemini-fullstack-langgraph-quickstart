#!/bin/bash

# 确保使用agent环境
source /opt/mamba/etc/profile.d/conda.sh
conda activate agent

# 设置环境变量以增加稳定性
export HTTPX_TIMEOUT=300
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_MAX_RETRIES=10
export LANGCHAIN_API_RETRY_BASE=4
export PYTHONUNBUFFERED=1

# 输出调试信息
echo "启动稳定版LangGraph服务..."
echo "环境: $(conda info --envs | grep '*')"
echo "Python: $(python --version)"
echo "LangGraph版本: $(pip list | grep langgraph)"

# 使用单工作进程和较长超时启动服务
langgraph dev --verbose --workers 1 --timeout 300 2>&1 | tee langgraph_debug.log 