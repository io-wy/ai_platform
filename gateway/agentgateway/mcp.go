package agentgateway

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"sync"
	"time"

	"github.com/rs/zerolog/log"

	"github.com/agentflow/gateway/config"
	pb "github.com/agentflow/gateway/proto"
)

// ============================================================================
// MCP 管理器
// ============================================================================

// MCPTool MCP 工具
type MCPTool struct {
	Name        string
	Description string
	Schema      string // JSON Schema
	ServerName  string
}

// MCPResource MCP 资源
type MCPResource struct {
	URI         string
	Name        string
	Description string
	MimeType    string
	ServerName  string
}

// MCPPrompt MCP Prompt
type MCPPrompt struct {
	Name        string
	Description string
	Arguments   []MCPPromptArg
	ServerName  string
}

// MCPPromptArg Prompt 参数
type MCPPromptArg struct {
	Name        string
	Description string
	Required    bool
}

// MCPManager MCP 服务管理器
type MCPManager struct {
	configs   []config.MCPServerConfig
	clients   map[string]*MCPClient
	tools     map[string]MCPTool
	resources map[string]MCPResource
	prompts   map[string]MCPPrompt
	mu        sync.RWMutex
}

// NewMCPManager 创建 MCP 管理器
func NewMCPManager(configs []config.MCPServerConfig) *MCPManager {
	return &MCPManager{
		configs:   configs,
		clients:   make(map[string]*MCPClient),
		tools:     make(map[string]MCPTool),
		resources: make(map[string]MCPResource),
		prompts:   make(map[string]MCPPrompt),
	}
}

// Initialize 初始化所有 MCP 连接
func (m *MCPManager) Initialize(ctx context.Context) error {
	for _, cfg := range m.configs {
		if !cfg.Enabled {
			continue
		}

		client, err := NewMCPClient(cfg)
		if err != nil {
			log.Warn().Str("name", cfg.Name).Err(err).Msg("MCP 客户端创建失败")
			continue
		}

		if err := client.Initialize(ctx); err != nil {
			log.Warn().Str("name", cfg.Name).Err(err).Msg("MCP 初始化失败")
			continue
		}

		m.mu.Lock()
		m.clients[cfg.Name] = client

		// 收集工具
		for _, tool := range client.tools {
			tool.ServerName = cfg.Name
			m.tools[tool.Name] = tool
		}

		// 收集资源
		for _, res := range client.resources {
			res.ServerName = cfg.Name
			m.resources[res.URI] = res
		}

		// 收集 Prompts
		for _, prompt := range client.prompts {
			prompt.ServerName = cfg.Name
			m.prompts[prompt.Name] = prompt
		}
		m.mu.Unlock()

		log.Info().
			Str("name", cfg.Name).
			Int("tools", len(client.tools)).
			Int("resources", len(client.resources)).
			Msg("MCP 服务已连接")
	}

	return nil
}

// GetTools 获取指定的工具
func (m *MCPManager) GetTools(names []string) []MCPTool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if len(names) == 0 {
		// 返回所有工具
		tools := make([]MCPTool, 0, len(m.tools))
		for _, t := range m.tools {
			tools = append(tools, t)
		}
		return tools
	}

	tools := make([]MCPTool, 0, len(names))
	for _, name := range names {
		if tool, exists := m.tools[name]; exists {
			tools = append(tools, tool)
		}
	}
	return tools
}

// ListAllTools 列出所有工具
func (m *MCPManager) ListAllTools() []MCPTool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	tools := make([]MCPTool, 0, len(m.tools))
	for _, t := range m.tools {
		tools = append(tools, t)
	}
	return tools
}

// CallTool 调用工具
func (m *MCPManager) CallTool(ctx context.Context, name string, args map[string]any) (any, error) {
	m.mu.RLock()
	tool, exists := m.tools[name]
	if !exists {
		m.mu.RUnlock()
		return nil, fmt.Errorf("工具不存在: %s", name)
	}

	client, exists := m.clients[tool.ServerName]
	m.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("MCP 服务不可用: %s", tool.ServerName)
	}

	return client.CallTool(ctx, name, args)
}

// ============================================================================
// gRPC 处理器实现
// ============================================================================

func (m *MCPManager) HandleInitialize(ctx context.Context, req *pb.InitializeRequest) (*pb.InitializeResponse, error) {
	return &pb.InitializeResponse{
		ProtocolVersion: "2024-11-05",
		Capabilities: &pb.ServerCapabilities{
			Tools:     &pb.ToolsCapability{ListChanged: true},
			Resources: &pb.ResourcesCapability{Subscribe: true, ListChanged: true},
			Prompts:   &pb.PromptsCapability{ListChanged: true},
		},
		ServerInfo: &pb.ServerInfo{
			Name:    "agentflow-mcp-bridge",
			Version: "1.0.0",
		},
	}, nil
}

func (m *MCPManager) HandleListResources(ctx context.Context, req *pb.ListResourcesRequest) (*pb.ListResourcesResponse, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	resources := make([]*pb.ResourceInfo, 0, len(m.resources))
	for _, r := range m.resources {
		resources = append(resources, &pb.ResourceInfo{
			Uri:         r.URI,
			Name:        r.Name,
			Description: r.Description,
			MimeType:    r.MimeType,
		})
	}

	return &pb.ListResourcesResponse{Resources: resources}, nil
}

func (m *MCPManager) HandleReadResource(ctx context.Context, req *pb.ReadResourceRequest) (*pb.ReadResourceResponse, error) {
	m.mu.RLock()
	res, exists := m.resources[req.Uri]
	if !exists {
		m.mu.RUnlock()
		return nil, fmt.Errorf("资源不存在: %s", req.Uri)
	}

	client := m.clients[res.ServerName]
	m.mu.RUnlock()

	content, err := client.ReadResource(ctx, req.Uri)
	if err != nil {
		return nil, err
	}

	return &pb.ReadResourceResponse{
		Contents: []*pb.ResourceContent{{
			Uri:      req.Uri,
			MimeType: res.MimeType,
			Text:     &content,
		}},
	}, nil
}

func (m *MCPManager) HandleListTools(ctx context.Context, req *pb.ListMCPToolsRequest) (*pb.ListMCPToolsResponse, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	tools := make([]*pb.MCPToolInfo, 0, len(m.tools))
	for _, t := range m.tools {
		tools = append(tools, &pb.MCPToolInfo{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.Schema,
		})
	}

	return &pb.ListMCPToolsResponse{Tools: tools}, nil
}

func (m *MCPManager) HandleCallTool(ctx context.Context, req *pb.CallToolRequest) (*pb.CallToolResponse, error) {
	var args map[string]any
	if err := json.Unmarshal([]byte(req.Arguments), &args); err != nil {
		return nil, err
	}

	result, err := m.CallTool(ctx, req.Name, args)
	if err != nil {
		return &pb.CallToolResponse{
			IsError: true,
			Content: []*pb.ToolContent{{Type: "text", Text: err.Error()}},
		}, nil
	}

	text, _ := json.Marshal(result)
	return &pb.CallToolResponse{
		Content: []*pb.ToolContent{{Type: "text", Text: string(text)}},
	}, nil
}

func (m *MCPManager) HandleListPrompts(ctx context.Context, req *pb.ListPromptsRequest) (*pb.ListPromptsResponse, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	prompts := make([]*pb.PromptInfo, 0, len(m.prompts))
	for _, p := range m.prompts {
		args := make([]*pb.PromptArgument, len(p.Arguments))
		for i, a := range p.Arguments {
			args[i] = &pb.PromptArgument{
				Name:        a.Name,
				Description: a.Description,
				Required:    a.Required,
			}
		}
		prompts = append(prompts, &pb.PromptInfo{
			Name:        p.Name,
			Description: p.Description,
			Arguments:   args,
		})
	}

	return &pb.ListPromptsResponse{Prompts: prompts}, nil
}

func (m *MCPManager) HandleGetPrompt(ctx context.Context, req *pb.GetPromptRequest) (*pb.GetPromptResponse, error) {
	m.mu.RLock()
	prompt, exists := m.prompts[req.Name]
	if !exists {
		m.mu.RUnlock()
		return nil, fmt.Errorf("Prompt 不存在: %s", req.Name)
	}

	client := m.clients[prompt.ServerName]
	m.mu.RUnlock()

	return client.GetPrompt(ctx, req.Name, req.Arguments)
}

// ============================================================================
// MCP 客户端 (支持 stdio 和 HTTP)
// ============================================================================

// MCPClient MCP 客户端
type MCPClient struct {
	cfg       config.MCPServerConfig
	transport MCPTransport
	tools     []MCPTool
	resources []MCPResource
	prompts   []MCPPrompt
}

// MCPTransport MCP 传输层
type MCPTransport interface {
	Send(ctx context.Context, method string, params any) (json.RawMessage, error)
	Close() error
}

// NewMCPClient 创建 MCP 客户端
func NewMCPClient(cfg config.MCPServerConfig) (*MCPClient, error) {
	client := &MCPClient{cfg: cfg}

	switch cfg.Transport {
	case "stdio":
		transport, err := NewStdioTransport(cfg.Command, cfg.Args, cfg.Env)
		if err != nil {
			return nil, err
		}
		client.transport = transport
	case "http", "sse":
		client.transport = NewHTTPTransport(cfg.URL)
	default:
		return nil, fmt.Errorf("不支持的传输类型: %s", cfg.Transport)
	}

	return client, nil
}

// Initialize 初始化连接
func (c *MCPClient) Initialize(ctx context.Context) error {
	// 发送初始化请求
	_, err := c.transport.Send(ctx, "initialize", map[string]any{
		"protocolVersion": "2024-11-05",
		"capabilities": map[string]any{
			"roots": map[string]any{"listChanged": true},
		},
		"clientInfo": map[string]any{
			"name":    "agentflow-gateway",
			"version": "1.0.0",
		},
	})
	if err != nil {
		return err
	}

	// 发送 initialized 通知
	c.transport.Send(ctx, "notifications/initialized", nil)

	// 获取工具列表
	toolsResp, err := c.transport.Send(ctx, "tools/list", nil)
	if err == nil {
		var result struct {
			Tools []struct {
				Name        string          `json:"name"`
				Description string          `json:"description"`
				InputSchema json.RawMessage `json:"inputSchema"`
			} `json:"tools"`
		}
		if json.Unmarshal(toolsResp, &result) == nil {
			for _, t := range result.Tools {
				c.tools = append(c.tools, MCPTool{
					Name:        t.Name,
					Description: t.Description,
					Schema:      string(t.InputSchema),
				})
			}
		}
	}

	// 获取资源列表
	resResp, err := c.transport.Send(ctx, "resources/list", nil)
	if err == nil {
		var result struct {
			Resources []struct {
				URI         string `json:"uri"`
				Name        string `json:"name"`
				Description string `json:"description"`
				MimeType    string `json:"mimeType"`
			} `json:"resources"`
		}
		if json.Unmarshal(resResp, &result) == nil {
			for _, r := range result.Resources {
				c.resources = append(c.resources, MCPResource{
					URI:         r.URI,
					Name:        r.Name,
					Description: r.Description,
					MimeType:    r.MimeType,
				})
			}
		}
	}

	// 获取 Prompts 列表
	promptsResp, err := c.transport.Send(ctx, "prompts/list", nil)
	if err == nil {
		var result struct {
			Prompts []struct {
				Name        string `json:"name"`
				Description string `json:"description"`
				Arguments   []struct {
					Name        string `json:"name"`
					Description string `json:"description"`
					Required    bool   `json:"required"`
				} `json:"arguments"`
			} `json:"prompts"`
		}
		if json.Unmarshal(promptsResp, &result) == nil {
			for _, p := range result.Prompts {
				args := make([]MCPPromptArg, len(p.Arguments))
				for i, a := range p.Arguments {
					args[i] = MCPPromptArg{
						Name:        a.Name,
						Description: a.Description,
						Required:    a.Required,
					}
				}
				c.prompts = append(c.prompts, MCPPrompt{
					Name:        p.Name,
					Description: p.Description,
					Arguments:   args,
				})
			}
		}
	}

	return nil
}

// CallTool 调用工具
func (c *MCPClient) CallTool(ctx context.Context, name string, args map[string]any) (any, error) {
	resp, err := c.transport.Send(ctx, "tools/call", map[string]any{
		"name":      name,
		"arguments": args,
	})
	if err != nil {
		return nil, err
	}

	var result struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		IsError bool `json:"isError"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, err
	}

	if result.IsError && len(result.Content) > 0 {
		return nil, fmt.Errorf(result.Content[0].Text)
	}

	if len(result.Content) > 0 {
		return result.Content[0].Text, nil
	}

	return nil, nil
}

// ReadResource 读取资源
func (c *MCPClient) ReadResource(ctx context.Context, uri string) (string, error) {
	resp, err := c.transport.Send(ctx, "resources/read", map[string]any{
		"uri": uri,
	})
	if err != nil {
		return "", err
	}

	var result struct {
		Contents []struct {
			Text string `json:"text"`
		} `json:"contents"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return "", err
	}

	if len(result.Contents) > 0 {
		return result.Contents[0].Text, nil
	}

	return "", nil
}

// GetPrompt 获取 Prompt
func (c *MCPClient) GetPrompt(ctx context.Context, name string, args map[string]string) (*pb.GetPromptResponse, error) {
	resp, err := c.transport.Send(ctx, "prompts/get", map[string]any{
		"name":      name,
		"arguments": args,
	})
	if err != nil {
		return nil, err
	}

	var result struct {
		Description string `json:"description"`
		Messages    []struct {
			Role    string `json:"role"`
			Content struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, err
	}

	messages := make([]*pb.PromptMessage, len(result.Messages))
	for i, m := range result.Messages {
		messages[i] = &pb.PromptMessage{
			Role: m.Role,
			Content: &pb.PromptContent{
				Type: m.Content.Type,
				Text: m.Content.Text,
			},
		}
	}

	return &pb.GetPromptResponse{
		Description: result.Description,
		Messages:    messages,
	}, nil
}

// ============================================================================
// Stdio 传输 (用于本地 MCP Server)
// ============================================================================

// StdioTransport Stdio 传输
type StdioTransport struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout io.ReadCloser
	mu     sync.Mutex
	msgID  int64
}

// NewStdioTransport 创建 Stdio 传输
func NewStdioTransport(command string, args []string, env map[string]string) (*StdioTransport, error) {
	cmd := exec.Command(command, args...)

	for k, v := range env {
		cmd.Env = append(cmd.Env, fmt.Sprintf("%s=%s", k, v))
	}

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}

	if err := cmd.Start(); err != nil {
		return nil, err
	}

	return &StdioTransport{
		cmd:    cmd,
		stdin:  stdin,
		stdout: stdout,
	}, nil
}

// Send 发送请求
func (t *StdioTransport) Send(ctx context.Context, method string, params any) (json.RawMessage, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.msgID++
	id := t.msgID

	// 构建 JSON-RPC 请求
	req := map[string]any{
		"jsonrpc": "2.0",
		"id":      id,
		"method":  method,
	}
	if params != nil {
		req["params"] = params
	}

	data, _ := json.Marshal(req)
	data = append(data, '\n')

	if _, err := t.stdin.Write(data); err != nil {
		return nil, err
	}

	// 读取响应
	decoder := json.NewDecoder(t.stdout)
	var resp struct {
		ID     int64           `json:"id"`
		Result json.RawMessage `json:"result"`
		Error  *struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}

	// 设置超时
	done := make(chan error, 1)
	go func() {
		done <- decoder.Decode(&resp)
	}()

	select {
	case err := <-done:
		if err != nil {
			return nil, err
		}
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(30 * time.Second):
		return nil, fmt.Errorf("请求超时")
	}

	if resp.Error != nil {
		return nil, fmt.Errorf("MCP 错误 %d: %s", resp.Error.Code, resp.Error.Message)
	}

	return resp.Result, nil
}

// Close 关闭连接
func (t *StdioTransport) Close() error {
	t.stdin.Close()
	return t.cmd.Process.Kill()
}

// ============================================================================
// HTTP 传输 (用于远程 MCP Server)
// ============================================================================

// HTTPTransport HTTP 传输
type HTTPTransport struct {
	baseURL string
	client  *http.Client
	mu      sync.Mutex
	msgID   int64
}

// NewHTTPTransport 创建 HTTP 传输
func NewHTTPTransport(baseURL string) *HTTPTransport {
	return &HTTPTransport{
		baseURL: baseURL,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// Send 发送请求
func (t *HTTPTransport) Send(ctx context.Context, method string, params any) (json.RawMessage, error) {
	t.mu.Lock()
	t.msgID++
	id := t.msgID
	t.mu.Unlock()

	req := map[string]any{
		"jsonrpc": "2.0",
		"id":      id,
		"method":  method,
	}
	if params != nil {
		req["params"] = params
	}

	data, _ := json.Marshal(req)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", t.baseURL, nil)
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	// 设置请求体
	httpReq.Body = io.NopCloser(jsonReader(data))
	httpReq.ContentLength = int64(len(data))

	httpResp, err := t.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer httpResp.Body.Close()

	var resp struct {
		Result json.RawMessage `json:"result"`
		Error  *struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}

	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return nil, err
	}

	if resp.Error != nil {
		return nil, fmt.Errorf("MCP 错误 %d: %s", resp.Error.Code, resp.Error.Message)
	}

	return resp.Result, nil
}

// Close 关闭连接
func (t *HTTPTransport) Close() error {
	return nil
}

// jsonReader 用于创建 io.Reader
type jsonReaderImpl struct {
	data []byte
	pos  int
}

func jsonReader(data []byte) io.Reader {
	return &jsonReaderImpl{data: data}
}

func (r *jsonReaderImpl) Read(p []byte) (n int, err error) {
	if r.pos >= len(r.data) {
		return 0, io.EOF
	}
	n = copy(p, r.data[r.pos:])
	r.pos += n
	return
}
