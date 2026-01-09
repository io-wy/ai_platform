.PHONY: all build proto run-api run-agent run-all test clean

# 版本信息
VERSION ?= 0.1.0
COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_TIME := $(shell date -u '+%Y-%m-%d_%H:%M:%S')

# Go 参数
GOCMD=go
GOBUILD=$(GOCMD) build
GOTEST=$(GOCMD) test
GOCLEAN=$(GOCMD) clean
GOMOD=$(GOCMD) mod

# 构建参数
LDFLAGS=-ldflags "-X main.Version=$(VERSION) -X main.Commit=$(COMMIT) -X main.BuildTime=$(BUILD_TIME)"

# 输出目录
OUT_DIR=bin

all: proto build

# 生成 protobuf 代码
proto:
	@echo "生成 protobuf 代码..."
	protoc --go_out=. --go_opt=paths=source_relative \
		--go-grpc_out=. --go-grpc_opt=paths=source_relative \
		proto/gateway.proto

# 构建
build: 
	@echo "构建网关..."
	@mkdir -p $(OUT_DIR)
	$(GOBUILD) $(LDFLAGS) -o $(OUT_DIR)/gateway ./cmd/gateway

# 运行 API 网关
run-api:
	@echo "启动 API 网关..."
	$(GOCMD) run ./cmd/gateway -mode api -config config.yaml

# 运行 Agent 网关
run-agent:
	@echo "启动 Agent 网关..."
	$(GOCMD) run ./cmd/gateway -mode agent -config config.yaml

# 运行所有网关
run-all:
	@echo "启动所有网关..."
	$(GOCMD) run ./cmd/gateway -mode all -config config.yaml

# 开发模式运行 (带热重载)
dev:
	@echo "开发模式启动..."
	@which air > /dev/null || go install github.com/cosmtrek/air@latest
	air

# 测试
test:
	@echo "运行测试..."
	$(GOTEST) -v -race -cover ./...

# 基准测试
bench:
	@echo "运行基准测试..."
	$(GOTEST) -bench=. -benchmem ./...

# 清理
clean:
	@echo "清理..."
	$(GOCLEAN)
	rm -rf $(OUT_DIR)

# 依赖
deps:
	@echo "下载依赖..."
	$(GOMOD) download
	$(GOMOD) tidy

# Docker 构建
docker-build:
	@echo "构建 Docker 镜像..."
	docker build -t agentflow-gateway:$(VERSION) .

# Docker 运行
docker-run:
	@echo "运行 Docker 容器..."
	docker run -p 8080:8080 -p 9090:9090 -p 50051:50051 \
		-v $(PWD)/config.yaml:/app/config.yaml \
		agentflow-gateway:$(VERSION)

# 格式化
fmt:
	@echo "格式化代码..."
	$(GOCMD) fmt ./...
	goimports -w .

# Lint
lint:
	@echo "代码检查..."
	golangci-lint run ./...

# 帮助
help:
	@echo "AgentFlow Gateway Makefile"
	@echo ""
	@echo "使用方法:"
	@echo "  make proto      - 生成 protobuf 代码"
	@echo "  make build      - 构建网关"
	@echo "  make run-api    - 运行 API 网关"
	@echo "  make run-agent  - 运行 Agent 网关"
	@echo "  make run-all    - 运行所有网关"
	@echo "  make dev        - 开发模式 (热重载)"
	@echo "  make test       - 运行测试"
	@echo "  make bench      - 运行基准测试"
	@echo "  make clean      - 清理构建产物"
	@echo "  make deps       - 下载依赖"
	@echo "  make docker-build - 构建 Docker 镜像"
	@echo "  make fmt        - 格式化代码"
	@echo "  make lint       - 代码检查"
