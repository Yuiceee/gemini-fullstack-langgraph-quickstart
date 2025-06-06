#!/bin/bash

# 终止已经运行的进程
echo "正在终止可能已经运行的相关进程..."
pkill -f "langgraph dev" || true
pkill -f "npm run dev" || true

# 设置代理环境变量
export HTTP_PROXY=http://10.0.1.158:8118
export HTTPS_PROXY=http://10.0.1.158:8118
export http_proxy=http://10.0.1.158:8118
export https_proxy=http://10.0.1.158:8118
export PYTHONHTTPSVERIFY=0

# 增加稳定性配置
export HTTPX_TIMEOUT=300
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_MAX_RETRIES=10
export LANGCHAIN_API_RETRY_BASE=4
export PYTHONUNBUFFERED=1

# 保存项目根目录
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# 启动后端
echo "启动后端服务..."
cd "$PROJECT_ROOT/backend"
source /opt/mamba/etc/profile.d/conda.sh
conda activate agent

# 后台启动langgraph，只用一个工作进程以减少资源使用
langgraph dev --verbose --workers 1 --timeout 300 > langgraph_debug.log 2>&1 &
BACKEND_PID=$!
echo "后端服务已启动，PID: $BACKEND_PID"

# 等待后端服务启动
echo "等待后端服务启动..."
sleep 5

# 启动前端
echo "启动前端服务..."
cd "$PROJECT_ROOT/frontend"
npm run dev > frontend_debug.log 2>&1 &
FRONTEND_PID=$!
echo "前端服务已启动，PID: $FRONTEND_PID"

echo "全栈应用已启动!"
echo "访问 http://localhost:5173 查看应用"
echo "按 Ctrl+C 停止所有服务"

# 捕获SIGINT信号(Ctrl+C)，优雅地关闭所有进程
trap 'echo "正在关闭服务..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0' SIGINT

# 等待子进程
wait 