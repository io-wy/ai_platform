/*
Package apigateway - API 网关

负责处理业务层请求：
- HTTP/JSON REST API
- gRPC 服务
- WebSocket 连接
- 请求路由到 Agent Gateway
*/
package apigateway

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/rs/zerolog/log"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/agentflow/gateway/config"
	pb "github.com/agentflow/gateway/proto"
)

// Gateway API 网关
type Gateway struct {
	cfg        config.APIGatewayConfig
	httpServer *http.Server
	grpcServer *grpc.Server
	agentConn  *grpc.ClientConn
	mu         sync.RWMutex
}

// New 创建 API 网关
func New(cfg config.APIGatewayConfig) *Gateway {
	return &Gateway{cfg: cfg}
}

// Start 启动网关
func (g *Gateway) Start(ctx context.Context) error {
	// 连接 Agent Gateway
	conn, err := grpc.Dial(
		g.cfg.AgentAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		return fmt.Errorf("连接 Agent Gateway 失败: %w", err)
	}
	g.agentConn = conn

	// 启动 HTTP 服务
	go g.startHTTP(ctx)

	// 启动 gRPC 服务
	go g.startGRPC(ctx)

	log.Info().
		Str("http", g.cfg.HTTPAddr).
		Str("grpc", g.cfg.GRPCAddr).
		Msg("API Gateway 已启动")

	<-ctx.Done()
	return g.shutdown()
}

// startHTTP 启动 HTTP 服务
func (g *Gateway) startHTTP(ctx context.Context) {
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Recovery())

	// 中间件
	r.Use(g.loggerMiddleware())
	r.Use(g.corsMiddleware())
	if g.cfg.RateLimit.Enabled {
		r.Use(g.rateLimitMiddleware())
	}
	if g.cfg.Auth.Enabled {
		r.Use(g.authMiddleware())
	}

	// 路由
	api := r.Group("/api/v1")
	{
		// Agent 相关
		api.POST("/agent/chat", g.handleChat)
		api.POST("/agent/run", g.handleRun)
		api.GET("/agent/stream", g.handleStream)

		// 会话管理
		api.POST("/session", g.handleCreateSession)
		api.GET("/session/:id", g.handleGetSession)
		api.DELETE("/session/:id", g.handleDeleteSession)

		// 工具管理
		api.GET("/tools", g.handleListTools)
		api.POST("/tools/:name/execute", g.handleExecuteTool)

		// 健康检查
		api.GET("/health", g.handleHealth)
	}

	// WebSocket
	r.GET("/ws", g.handleWebSocket)

	g.httpServer = &http.Server{
		Addr:    g.cfg.HTTPAddr,
		Handler: r,
	}

	if err := g.httpServer.ListenAndServe(); err != http.ErrServerClosed {
		log.Error().Err(err).Msg("HTTP 服务错误")
	}
}

// startGRPC 启动 gRPC 服务
func (g *Gateway) startGRPC(ctx context.Context) {
	lis, err := net.Listen("tcp", g.cfg.GRPCAddr)
	if err != nil {
		log.Fatal().Err(err).Msg("gRPC 监听失败")
	}

	g.grpcServer = grpc.NewServer()
	pb.RegisterAPIGatewayServer(g.grpcServer, &apiServer{gateway: g})

	if err := g.grpcServer.Serve(lis); err != nil {
		log.Error().Err(err).Msg("gRPC 服务错误")
	}
}

// shutdown 关闭服务
func (g *Gateway) shutdown() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if g.httpServer != nil {
		g.httpServer.Shutdown(ctx)
	}
	if g.grpcServer != nil {
		g.grpcServer.GracefulStop()
	}
	if g.agentConn != nil {
		g.agentConn.Close()
	}

	return nil
}

// ============================================================================
// HTTP 处理器
// ============================================================================

// ChatRequest 聊天请求
type ChatRequest struct {
	SessionID string `json:"session_id"`
	Message   string `json:"message"`
	Stream    bool   `json:"stream"`
}

// ChatResponse 聊天响应
type ChatResponse struct {
	SessionID string `json:"session_id"`
	Message   string `json:"message"`
	Done      bool   `json:"done"`
}

// handleChat 处理聊天请求
func (g *Gateway) handleChat(c *gin.Context) {
	var req ChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// 调用 Agent Gateway
	client := pb.NewAgentGatewayClient(g.agentConn)
	ctx, cancel := context.WithTimeout(c.Request.Context(), g.cfg.Timeout)
	defer cancel()

	resp, err := client.Chat(ctx, &pb.ChatRequest{
		SessionId: req.SessionID,
		Message:   req.Message,
	})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, ChatResponse{
		SessionID: resp.SessionId,
		Message:   resp.Message,
		Done:      true,
	})
}

// RunRequest 运行请求
type RunRequest struct {
	SessionID string            `json:"session_id"`
	Task      string            `json:"task"`
	Pattern   string            `json:"pattern"`
	Tools     []string          `json:"tools"`
	Config    map[string]string `json:"config"`
}

// RunResponse 运行响应
type RunResponse struct {
	SessionID string           `json:"session_id"`
	Output    string           `json:"output"`
	Success   bool             `json:"success"`
	Steps     []map[string]any `json:"steps"`
	Meta      map[string]any   `json:"meta"`
}

// handleRun 处理运行请求
func (g *Gateway) handleRun(c *gin.Context) {
	var req RunRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	client := pb.NewAgentGatewayClient(g.agentConn)
	ctx, cancel := context.WithTimeout(c.Request.Context(), g.cfg.Timeout)
	defer cancel()

	resp, err := client.Run(ctx, &pb.RunRequest{
		SessionId: req.SessionID,
		Task:      req.Task,
		Pattern:   req.Pattern,
		Tools:     req.Tools,
	})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, RunResponse{
		SessionID: resp.SessionId,
		Output:    resp.Output,
		Success:   resp.Success,
	})
}

// handleStream 处理流式请求
func (g *Gateway) handleStream(c *gin.Context) {
	sessionID := c.Query("session_id")
	message := c.Query("message")

	if message == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "message is required"})
		return
	}

	client := pb.NewAgentGatewayClient(g.agentConn)
	stream, err := client.ChatStream(c.Request.Context(), &pb.ChatRequest{
		SessionId: sessionID,
		Message:   message,
	})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")

	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "streaming not supported"})
		return
	}

	for {
		resp, err := stream.Recv()
		if err != nil {
			break
		}

		data, _ := json.Marshal(ChatResponse{
			SessionID: resp.SessionId,
			Message:   resp.Message,
			Done:      resp.Done,
		})

		fmt.Fprintf(c.Writer, "data: %s\n\n", data)
		flusher.Flush()

		if resp.Done {
			break
		}
	}
}

// handleCreateSession 创建会话
func (g *Gateway) handleCreateSession(c *gin.Context) {
	var req struct {
		AgentConfig map[string]string `json:"agent_config"`
	}
	c.ShouldBindJSON(&req)

	client := pb.NewAgentGatewayClient(g.agentConn)
	resp, err := client.CreateSession(c.Request.Context(), &pb.CreateSessionRequest{
		Config: req.AgentConfig,
	})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"session_id": resp.SessionId,
	})
}

// handleGetSession 获取会话
func (g *Gateway) handleGetSession(c *gin.Context) {
	sessionID := c.Param("id")

	client := pb.NewAgentGatewayClient(g.agentConn)
	resp, err := client.GetSession(c.Request.Context(), &pb.GetSessionRequest{
		SessionId: sessionID,
	})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"session_id": resp.SessionId,
		"created_at": resp.CreatedAt,
		"messages":   resp.Messages,
	})
}

// handleDeleteSession 删除会话
func (g *Gateway) handleDeleteSession(c *gin.Context) {
	sessionID := c.Param("id")

	client := pb.NewAgentGatewayClient(g.agentConn)
	_, err := client.DeleteSession(c.Request.Context(), &pb.DeleteSessionRequest{
		SessionId: sessionID,
	})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"deleted": true})
}

// handleListTools 列出工具
func (g *Gateway) handleListTools(c *gin.Context) {
	client := pb.NewAgentGatewayClient(g.agentConn)
	resp, err := client.ListTools(c.Request.Context(), &pb.ListToolsRequest{})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"tools": resp.Tools})
}

// handleExecuteTool 执行工具
func (g *Gateway) handleExecuteTool(c *gin.Context) {
	name := c.Param("name")

	var args map[string]any
	c.ShouldBindJSON(&args)

	argsJSON, _ := json.Marshal(args)

	client := pb.NewAgentGatewayClient(g.agentConn)
	resp, err := client.ExecuteTool(c.Request.Context(), &pb.ExecuteToolRequest{
		Name:      name,
		Arguments: string(argsJSON),
	})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"result": resp.Result,
		"error":  resp.Error,
	})
}

// handleHealth 健康检查
func (g *Gateway) handleHealth(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status": "ok",
		"time":   time.Now().Unix(),
	})
}

// ============================================================================
// WebSocket 处理
// ============================================================================

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

// handleWebSocket 处理 WebSocket 连接
func (g *Gateway) handleWebSocket(c *gin.Context) {
	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		return
	}
	defer conn.Close()

	client := pb.NewAgentGatewayClient(g.agentConn)
	sessionID := ""

	for {
		_, message, err := conn.ReadMessage()
		if err != nil {
			break
		}

		var req struct {
			Type      string `json:"type"`
			SessionID string `json:"session_id"`
			Message   string `json:"message"`
		}
		if err := json.Unmarshal(message, &req); err != nil {
			continue
		}

		if req.SessionID != "" {
			sessionID = req.SessionID
		}

		switch req.Type {
		case "chat":
			resp, err := client.Chat(context.Background(), &pb.ChatRequest{
				SessionId: sessionID,
				Message:   req.Message,
			})
			if err != nil {
				conn.WriteJSON(gin.H{"error": err.Error()})
				continue
			}

			sessionID = resp.SessionId
			conn.WriteJSON(gin.H{
				"type":       "response",
				"session_id": resp.SessionId,
				"message":    resp.Message,
			})

		case "stream":
			stream, err := client.ChatStream(context.Background(), &pb.ChatRequest{
				SessionId: sessionID,
				Message:   req.Message,
			})
			if err != nil {
				conn.WriteJSON(gin.H{"error": err.Error()})
				continue
			}

			for {
				resp, err := stream.Recv()
				if err != nil {
					break
				}

				conn.WriteJSON(gin.H{
					"type":       "chunk",
					"session_id": resp.SessionId,
					"message":    resp.Message,
					"done":       resp.Done,
				})

				if resp.Done {
					break
				}
			}
		}
	}
}

// ============================================================================
// 中间件
// ============================================================================

func (g *Gateway) loggerMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		c.Next()
		log.Info().
			Str("method", c.Request.Method).
			Str("path", c.Request.URL.Path).
			Int("status", c.Writer.Status()).
			Dur("latency", time.Since(start)).
			Msg("请求")
	}
}

func (g *Gateway) corsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}
		c.Next()
	}
}

func (g *Gateway) rateLimitMiddleware() gin.HandlerFunc {
	// 简单令牌桶限流
	type bucket struct {
		tokens   int
		lastFill time.Time
	}
	buckets := make(map[string]*bucket)
	var mu sync.Mutex

	return func(c *gin.Context) {
		ip := c.ClientIP()

		mu.Lock()
		b, exists := buckets[ip]
		if !exists {
			b = &bucket{tokens: g.cfg.RateLimit.Burst, lastFill: time.Now()}
			buckets[ip] = b
		}

		// 补充令牌
		elapsed := time.Since(b.lastFill)
		b.tokens += int(elapsed.Seconds() * float64(g.cfg.RateLimit.RPS))
		if b.tokens > g.cfg.RateLimit.Burst {
			b.tokens = g.cfg.RateLimit.Burst
		}
		b.lastFill = time.Now()

		if b.tokens <= 0 {
			mu.Unlock()
			c.JSON(http.StatusTooManyRequests, gin.H{"error": "rate limit exceeded"})
			c.Abort()
			return
		}

		b.tokens--
		mu.Unlock()

		c.Next()
	}
}

func (g *Gateway) authMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// API Key 认证
		apiKey := c.GetHeader("X-API-Key")
		if apiKey == "" {
			apiKey = c.Query("api_key")
		}

		if apiKey != "" {
			for _, key := range g.cfg.Auth.APIKeys {
				if apiKey == key {
					c.Next()
					return
				}
			}
		}

		// JWT 认证 (简化实现)
		token := c.GetHeader("Authorization")
		if token != "" && g.validateJWT(token) {
			c.Next()
			return
		}

		c.JSON(http.StatusUnauthorized, gin.H{"error": "unauthorized"})
		c.Abort()
	}
}

func (g *Gateway) validateJWT(token string) bool {
	// TODO: 实现 JWT 验证
	return false
}

// ============================================================================
// gRPC 服务
// ============================================================================

type apiServer struct {
	pb.UnimplementedAPIGatewayServer
	gateway *Gateway
}

func (s *apiServer) Health(ctx context.Context, req *pb.HealthRequest) (*pb.HealthResponse, error) {
	return &pb.HealthResponse{
		Status: "ok",
		Time:   time.Now().Unix(),
	}, nil
}
