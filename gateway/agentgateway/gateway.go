/*
Package agentgateway - Agent 网关

负责：
- Agent 池管理
- MCP 协议桥接
- Agent 监控与指标收集
- 会话状态管理
*/
package agentgateway

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/rs/zerolog/log"
	"google.golang.org/grpc"

	"github.com/agentflow/gateway/config"
	pb "github.com/agentflow/gateway/proto"
)

// Gateway Agent 网关
type Gateway struct {
	cfg        config.AgentGatewayConfig
	grpcServer *grpc.Server
	pool       *AgentPool
	monitor    *Monitor
	mcpManager *MCPManager
	sessions   *SessionManager
	mu         sync.RWMutex
}

// New 创建 Agent 网关
func New(cfg config.AgentGatewayConfig) *Gateway {
	return &Gateway{
		cfg:        cfg,
		pool:       NewAgentPool(cfg.Pool),
		monitor:    NewMonitor(cfg.Monitor),
		mcpManager: NewMCPManager(cfg.MCPServers),
		sessions:   NewSessionManager(),
	}
}

// Start 启动网关
func (g *Gateway) Start(ctx context.Context) error {
	// 启动监控
	go g.monitor.Start(ctx)

	// 初始化 MCP 连接
	if err := g.mcpManager.Initialize(ctx); err != nil {
		log.Warn().Err(err).Msg("MCP 初始化警告")
	}

	// 启动 Agent 池
	if err := g.pool.Start(ctx); err != nil {
		return fmt.Errorf("Agent 池启动失败: %w", err)
	}

	// 启动 gRPC 服务
	lis, err := net.Listen("tcp", g.cfg.GRPCAddr)
	if err != nil {
		return fmt.Errorf("监听失败: %w", err)
	}

	g.grpcServer = grpc.NewServer()
	pb.RegisterAgentGatewayServer(g.grpcServer, &agentServer{gateway: g})
	pb.RegisterMCPBridgeServer(g.grpcServer, &mcpServer{gateway: g})

	log.Info().
		Str("addr", g.cfg.GRPCAddr).
		Int("pool_size", g.cfg.Pool.MinSize).
		Int("mcp_servers", len(g.cfg.MCPServers)).
		Msg("Agent Gateway 已启动")

	go func() {
		<-ctx.Done()
		g.grpcServer.GracefulStop()
	}()

	return g.grpcServer.Serve(lis)
}

// ============================================================================
// gRPC Agent Service 实现
// ============================================================================

type agentServer struct {
	pb.UnimplementedAgentGatewayServer
	gateway *Gateway
}

// Chat 处理聊天请求
func (s *agentServer) Chat(ctx context.Context, req *pb.ChatRequest) (*pb.ChatResponse, error) {
	start := time.Now()
	defer func() {
		s.gateway.monitor.RecordLatency("chat", time.Since(start))
	}()

	// 获取或创建会话
	session := s.gateway.sessions.GetOrCreate(req.SessionId)

	// 从池中获取 Agent
	agent, err := s.gateway.pool.Acquire(ctx)
	if err != nil {
		s.gateway.monitor.RecordError("chat", err)
		return nil, err
	}
	defer s.gateway.pool.Release(agent)

	// 添加消息到会话
	session.AddMessage("user", req.Message)

	// 调用 Agent
	response, err := agent.Chat(ctx, session.Messages)
	if err != nil {
		s.gateway.monitor.RecordError("chat", err)
		return nil, err
	}

	// 添加响应到会话
	session.AddMessage("assistant", response)

	s.gateway.monitor.RecordRequest("chat")

	return &pb.ChatResponse{
		SessionId: session.ID,
		Message:   response,
		Done:      true,
	}, nil
}

// ChatStream 处理流式聊天
func (s *agentServer) ChatStream(req *pb.ChatRequest, stream pb.AgentGateway_ChatStreamServer) error {
	session := s.gateway.sessions.GetOrCreate(req.SessionId)

	agent, err := s.gateway.pool.Acquire(stream.Context())
	if err != nil {
		return err
	}
	defer s.gateway.pool.Release(agent)

	session.AddMessage("user", req.Message)

	// 流式响应
	fullResponse := ""
	err = agent.ChatStream(stream.Context(), session.Messages, func(chunk string, done bool) error {
		fullResponse += chunk
		return stream.Send(&pb.ChatResponse{
			SessionId: session.ID,
			Message:   chunk,
			Done:      done,
		})
	})

	if err != nil {
		return err
	}

	session.AddMessage("assistant", fullResponse)
	return nil
}

// Run 运行 Agent 任务
func (s *agentServer) Run(ctx context.Context, req *pb.RunRequest) (*pb.RunResponse, error) {
	start := time.Now()
	defer func() {
		s.gateway.monitor.RecordLatency("run", time.Since(start))
	}()

	session := s.gateway.sessions.GetOrCreate(req.SessionId)

	agent, err := s.gateway.pool.Acquire(ctx)
	if err != nil {
		return nil, err
	}
	defer s.gateway.pool.Release(agent)

	// 收集 MCP 工具
	mcpTools := s.gateway.mcpManager.GetTools(req.Tools)

	// 运行 Agent
	result, err := agent.Run(ctx, AgentTask{
		Task:    req.Task,
		Pattern: req.Pattern,
		Tools:   mcpTools,
		Config:  req.Config,
	})
	if err != nil {
		s.gateway.monitor.RecordError("run", err)
		return nil, err
	}

	// 转换步骤信息
	steps := make([]*pb.StepInfo, len(result.Steps))
	for i, step := range result.Steps {
		steps[i] = &pb.StepInfo{
			Index:     int32(i),
			Type:      step.Type,
			Content:   step.Content,
			Timestamp: step.Timestamp.Unix(),
		}
	}

	s.gateway.monitor.RecordRequest("run")

	return &pb.RunResponse{
		SessionId: session.ID,
		Output:    result.Output,
		Success:   result.Success,
		Steps:     steps,
	}, nil
}

// RunStream 流式运行 Agent 任务
func (s *agentServer) RunStream(req *pb.RunRequest, stream pb.AgentGateway_RunStreamServer) error {
	// 类似 Run，但流式返回每个步骤
	agent, err := s.gateway.pool.Acquire(stream.Context())
	if err != nil {
		return err
	}
	defer s.gateway.pool.Release(agent)

	mcpTools := s.gateway.mcpManager.GetTools(req.Tools)

	return agent.RunStream(stream.Context(), AgentTask{
		Task:    req.Task,
		Pattern: req.Pattern,
		Tools:   mcpTools,
	}, func(step AgentStep) error {
		return stream.Send(&pb.RunResponse{
			Output:  step.Content,
			Success: step.Type != "error",
			Steps: []*pb.StepInfo{{
				Type:      step.Type,
				Content:   step.Content,
				Timestamp: step.Timestamp.Unix(),
			}},
		})
	})
}

// CreateSession 创建会话
func (s *agentServer) CreateSession(ctx context.Context, req *pb.CreateSessionRequest) (*pb.CreateSessionResponse, error) {
	session := s.gateway.sessions.Create(req.Config)
	return &pb.CreateSessionResponse{SessionId: session.ID}, nil
}

// GetSession 获取会话
func (s *agentServer) GetSession(ctx context.Context, req *pb.GetSessionRequest) (*pb.GetSessionResponse, error) {
	session := s.gateway.sessions.Get(req.SessionId)
	if session == nil {
		return nil, fmt.Errorf("session not found: %s", req.SessionId)
	}

	messages := make([]*pb.MessageInfo, len(session.Messages))
	for i, msg := range session.Messages {
		messages[i] = &pb.MessageInfo{
			Role:      msg.Role,
			Content:   msg.Content,
			Timestamp: msg.Timestamp.Unix(),
		}
	}

	return &pb.GetSessionResponse{
		SessionId: session.ID,
		CreatedAt: session.CreatedAt.Unix(),
		Messages:  messages,
	}, nil
}

// DeleteSession 删除会话
func (s *agentServer) DeleteSession(ctx context.Context, req *pb.DeleteSessionRequest) (*pb.DeleteSessionResponse, error) {
	s.gateway.sessions.Delete(req.SessionId)
	return &pb.DeleteSessionResponse{Success: true}, nil
}

// ListTools 列出所有工具
func (s *agentServer) ListTools(ctx context.Context, req *pb.ListToolsRequest) (*pb.ListToolsResponse, error) {
	tools := s.gateway.mcpManager.ListAllTools()

	pbTools := make([]*pb.ToolInfo, len(tools))
	for i, t := range tools {
		pbTools[i] = &pb.ToolInfo{
			Name:             t.Name,
			Description:      t.Description,
			ParametersSchema: t.Schema,
		}
	}

	return &pb.ListToolsResponse{Tools: pbTools}, nil
}

// ExecuteTool 执行工具
func (s *agentServer) ExecuteTool(ctx context.Context, req *pb.ExecuteToolRequest) (*pb.ExecuteToolResponse, error) {
	var args map[string]any
	if err := json.Unmarshal([]byte(req.Arguments), &args); err != nil {
		return nil, err
	}

	result, err := s.gateway.mcpManager.CallTool(ctx, req.Name, args)
	if err != nil {
		return &pb.ExecuteToolResponse{Error: err.Error()}, nil
	}

	resultJSON, _ := json.Marshal(result)
	return &pb.ExecuteToolResponse{Result: string(resultJSON)}, nil
}

// GetMetrics 获取指标
func (s *agentServer) GetMetrics(ctx context.Context, req *pb.GetMetricsRequest) (*pb.GetMetricsResponse, error) {
	metrics := s.gateway.monitor.GetMetrics(req.MetricNames)
	return &pb.GetMetricsResponse{Metrics: metrics}, nil
}

// GetAgentStatus 获取 Agent 状态
func (s *agentServer) GetAgentStatus(ctx context.Context, req *pb.GetAgentStatusRequest) (*pb.GetAgentStatusResponse, error) {
	status := s.gateway.pool.GetStatus(req.AgentId)
	return &pb.GetAgentStatusResponse{
		AgentId:        status.ID,
		Status:         status.Status,
		ActiveSessions: status.ActiveSessions,
		MemoryUsage:    status.MemoryUsage,
		Uptime:         status.Uptime,
	}, nil
}

// ============================================================================
// MCP Bridge Service 实现
// ============================================================================

type mcpServer struct {
	pb.UnimplementedMCPBridgeServer
	gateway *Gateway
}

func (s *mcpServer) Initialize(ctx context.Context, req *pb.InitializeRequest) (*pb.InitializeResponse, error) {
	return s.gateway.mcpManager.HandleInitialize(ctx, req)
}

func (s *mcpServer) ListResources(ctx context.Context, req *pb.ListResourcesRequest) (*pb.ListResourcesResponse, error) {
	return s.gateway.mcpManager.HandleListResources(ctx, req)
}

func (s *mcpServer) ReadResource(ctx context.Context, req *pb.ReadResourceRequest) (*pb.ReadResourceResponse, error) {
	return s.gateway.mcpManager.HandleReadResource(ctx, req)
}

func (s *mcpServer) ListMCPTools(ctx context.Context, req *pb.ListMCPToolsRequest) (*pb.ListMCPToolsResponse, error) {
	return s.gateway.mcpManager.HandleListTools(ctx, req)
}

func (s *mcpServer) CallTool(ctx context.Context, req *pb.CallToolRequest) (*pb.CallToolResponse, error) {
	return s.gateway.mcpManager.HandleCallTool(ctx, req)
}

func (s *mcpServer) ListPrompts(ctx context.Context, req *pb.ListPromptsRequest) (*pb.ListPromptsResponse, error) {
	return s.gateway.mcpManager.HandleListPrompts(ctx, req)
}

func (s *mcpServer) GetPrompt(ctx context.Context, req *pb.GetPromptRequest) (*pb.GetPromptResponse, error) {
	return s.gateway.mcpManager.HandleGetPrompt(ctx, req)
}

// ============================================================================
// 会话管理
// ============================================================================

// Message 消息
type Message struct {
	Role      string
	Content   string
	Timestamp time.Time
}

// Session 会话
type Session struct {
	ID        string
	Messages  []Message
	Config    map[string]string
	CreatedAt time.Time
	UpdatedAt time.Time
	mu        sync.RWMutex
}

func (s *Session) AddMessage(role, content string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Messages = append(s.Messages, Message{
		Role:      role,
		Content:   content,
		Timestamp: time.Now(),
	})
	s.UpdatedAt = time.Now()
}

// SessionManager 会话管理器
type SessionManager struct {
	sessions map[string]*Session
	mu       sync.RWMutex
}

func NewSessionManager() *SessionManager {
	return &SessionManager{
		sessions: make(map[string]*Session),
	}
}

func (m *SessionManager) Create(config map[string]string) *Session {
	m.mu.Lock()
	defer m.mu.Unlock()

	session := &Session{
		ID:        uuid.New().String(),
		Messages:  []Message{},
		Config:    config,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	m.sessions[session.ID] = session
	return session
}

func (m *SessionManager) Get(id string) *Session {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.sessions[id]
}

func (m *SessionManager) GetOrCreate(id string) *Session {
	if id == "" {
		return m.Create(nil)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if session, exists := m.sessions[id]; exists {
		return session
	}

	session := &Session{
		ID:        id,
		Messages:  []Message{},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	m.sessions[id] = session
	return session
}

func (m *SessionManager) Delete(id string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.sessions, id)
}
