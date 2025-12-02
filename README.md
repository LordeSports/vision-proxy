# Vision Proxy

一个为纯文字大模型增加多模态视觉能力的代理服务。通过 OCR 模型将图片转换为文字描述，使不支持图片的 LLM 也能"看到"图片内容。

## 工作原理

```
客户端请求(含图片) → Vision Proxy → OCR模型(提取图片描述) → 上游LLM(纯文字)
                                   ↓
                         移除图片 + 追加描述
                         "图片1内容：(描述)"
```

1. 代理接收 OpenAI 格式的聊天请求
2. 检测消息中是否包含图片
3. 若有图片，发送到 OCR 端点获取文字描述
4. 将描述追加到消息末尾，移除原图片
5. 转发处理后的纯文字请求到上游模型

## 快速开始

### Docker Compose（推荐）

```bash
# 1. 复制并编辑环境变量
cp .env.example .env

# 2. 启动服务
docker compose up -d

# 3. 查看日志
docker compose logs -f
```

### 手动运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件

# 3. 启动服务
python main.py
```

## 配置说明

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `OCR_ENDPOINT` | OCR 服务地址 | `http://localhost:8080/v1/chat/completions` |
| `OCR_API_KEY` | OCR 服务 API Key | 空 |
| `OCR_MODEL_NAME` | OCR 模型名称 | `deepseek-ocr` |
| `OCR_PARALLEL` | 多图片是否并行处理 | `true` |
| `OCR_PROMPT` | 发送给 OCR 的提示词 | `请详细描述这张图片的内容` |
| `OCR_TIMEOUT` | OCR 调用超时(秒) | `60` |
| `UPSTREAM_TIMEOUT` | 上游转发超时(秒) | `300` |
| `PROXY_HOST` | 代理监听地址 | `0.0.0.0` |
| `PROXY_PORT` | 代理监听端口 | `8000` |

## 使用方式

请求路径格式：`/{上游完整URL}`

### 示例

假设代理运行在 `localhost:8000`，上游模型地址为 `http://172.17.0.1:7000`：

```bash
# 发送聊天请求（图片会被 OCR 处理）
curl http://localhost:8000/http://172.17.0.1:7000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-xxx" \
  -d '{
    "model": "qwen",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "这张图片里有什么？"},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
        ]
      }
    ]
  }'

# 获取模型列表（直接透传）
curl http://localhost:8000/http://172.17.0.1:7000/v1/models \
  -H "Authorization: Bearer sk-xxx"
```

### 路由规则

- `*/v1/chat/completions` - 处理图片后转发
- 其他路径 - 直接透传（如 `/v1/models`）

## API 端点

| 端点 | 说明 |
|------|------|
| `GET /health` | 健康检查 |
| `/{path:path}` | 代理转发 |

## 特性

- ✅ 支持流式响应透传 (SSE)
- ✅ 多图片串行/并行处理可配置
- ✅ 图片描述带序号（图片1、图片2...）
- ✅ 非 chat/completions 路径直接透传
- ✅ 原始 Authorization 头转发到上游

## License

MIT
