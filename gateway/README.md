# AgentFlow Gateway

高性能 Go 语言网关，为 AgentFlow AI Agent 框架提供 HTTP/gRPC API 和 MCP 协议桥接。

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client                                   │
│                (Web App / Mobile / CLI)                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Gateway                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  HTTP/JSON  │  │    gRPC     │  │      WebSocket          │  │
│  │  REST API   │  │   Service   │  │   (Streaming)           │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
│         │                │                      │                │
│         └────────────────┴──────────────────────┘                │
│                          │                                       │
│    ┌─────────────────────┴─────────────────────┐                │
│    │  Rate Limit │ Auth │ CORS │ Logging       │                │
│    └─────────────────────┬─────────────────────┘                │
└──────────────────────────┼──────────────────────────────────────┘
                           │ gRPC
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Agent Gateway                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐   │
│  │   Agent Pool    │  │   MCP Manager   │  │    Monitor     │   │
│  │   (管理 Agent)   │  │  (MCP 协议桥接)  │  │  (Prometheus)  │   │
│  └────────┬────────┘  └────────┬────────┘  └───────┬────────┘   │
│           │                    │                    │            │
│           ▼                    ▼                    ▼            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐   │
│  │ Python Agent    │  │  MCP Servers    │  │   Metrics      │   │
│  │ (via HTTP/gRPC) │  │ (stdio/http)    │  │   Tracing      │   │
│  └─────────────────┘  └─────────────────┘  │   Alerting     │   │
│                                            └────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 功能

### API Gateway
- **HTTP/JSON REST API** - 标准 RESTful 接口
- **gRPC** - 高性能内部通信
- **WebSocket** - 实时双向通信
- **流式响应** - Server-Sent Events (SSE)
- **限流** - 令牌桶算法
- **认证** - API Key / JWT
- **CORS** - 跨域资源共享

### Agent Gateway
- **Agent 池** - 复用 Agent 实例，提高性能
- **MCP 协议桥接** - 支持 stdio/http 传输
- **会话管理** - 多轮对话状态管理
- **监控** - Prometheus 指标、分布式追踪、告警

## 快速开始

### 前置条件

- Go 1.22+
- protoc (Protocol Buffers 编译器)
- protoc-gen-go, protoc-gen-go-grpc

### 安装依赖

```bash
# 安装 protobuf 编译器
brew install protobuf

# 安装 Go protobuf 插件
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# 下载项目依赖
make deps
```

### 构建

```bash
# 生成 protobuf 代码
make proto

# 构建
make build
```

### 运行

```bash
# 运行所有网关
make run-all

# 或分别运行
make run-api    # API Gateway (:8080, :9090)
make run-agent  # Agent Gateway (:50051)

# 开发模式 (热重载)
make dev
```

## 配置

编辑 `config.yaml`:

```yaml
api_gateway:
  http_addr: ":8080"
  grpc_addr: ":9090"
  agent_addr: "localhost:50051"
  
  rate_limit:
    enabled: true
    rps: 100
    burst: 200
  
  auth:
    enabled: true
    api_keys:
      - "your-api-key"

agent_gateway:
  grpc_addr: ":50051"
  
  mcp_servers:
    - name: filesystem
      enabled: true
      transport: stdio
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
  
  pool:
    min_size: 2
    max_size: 100
  
  monitor:
    metrics:
      enabled: true
      addr: ":9091"
```

## API 文档

### REST API

#### 聊天

```bash
# 单轮对话
curl -X POST http://localhost:8080/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?"}'

# 带会话 ID
curl -X POST http://localhost:8080/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123", "message": "What did I just say?"}'
```

#### 流式响应

```bash
curl -N http://localhost:8080/api/v1/agent/stream?message=Hello
```

#### 运行 Agent 任务

```bash
curl -X POST http://localhost:8080/api/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{
    "task": "搜索并总结最近的 AI 新闻",
    "pattern": "react",
    "tools": ["search", "summarize"]
  }'
```

#### 会话管理

```bash
# 创建会话
curl -X POST http://localhost:8080/api/v1/session

# 获取会话
curl http://localhost:8080/api/v1/session/abc123

# 删除会话
curl -X DELETE http://localhost:8080/api/v1/session/abc123
```

#### 工具管理

```bash
# 列出工具
curl http://localhost:8080/api/v1/tools

# 执行工具
curl -X POST http://localhost:8080/api/v1/tools/search/execute \
  -H "Content-Type: application/json" \
  -d '{"query": "AI news"}'
```

### WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8080/ws');

// 发送消息
ws.send(JSON.stringify({
  type: 'chat',
  message: 'Hello!'
}));

// 流式消息
ws.send(JSON.stringify({
  type: 'stream',
  message: 'Tell me a story'
}));

// 接收消息
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.message);
};
```

## 监控

### Prometheus 指标

访问 `http://localhost:9091/metrics`

可用指标：
- `agentflow_requests_total` - 总请求数
- `agentflow_request_latency_seconds` - 请求延迟
- `agentflow_errors_total` - 错误数
- `agentflow_active` - 活跃连接数

### 健康检查

```bash
curl http://localhost:8080/api/v1/health
```

## 开发

### 项目结构

```
gateway/
├── cmd/
│   └── gateway/
│       └── main.go          # 入口
├── apigateway/
│   └── gateway.go           # API 网关实现
├── agentgateway/
│   ├── gateway.go           # Agent 网关实现
│   ├── pool.go              # Agent 池
│   ├── mcp.go               # MCP 客户端
│   └── monitor.go           # 监控系统
├── config/
│   └── config.go            # 配置定义
├── proto/
│   └── gateway.proto        # gRPC 定义
├── config.yaml              # 配置文件
├── Makefile
└── README.md
```

### 测试

```bash
make test
make bench
```

### Lint

```bash
make lint
```

## Docker

```bash
# 构建镜像
make docker-build

# 运行容器
make docker-run
```

## 许可证

MIT License
