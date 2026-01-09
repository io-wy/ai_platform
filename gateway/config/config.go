package config

import (
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

// Config 网关总配置
type Config struct {
	APIGateway   APIGatewayConfig   `yaml:"api_gateway"`
	AgentGateway AgentGatewayConfig `yaml:"agent_gateway"`
}

// APIGatewayConfig API 网关配置
type APIGatewayConfig struct {
	// HTTP 服务
	HTTPAddr string `yaml:"http_addr" default:":8080"`

	// gRPC 服务
	GRPCAddr string `yaml:"grpc_addr" default:":9090"`

	// Agent 服务地址
	AgentAddr string `yaml:"agent_addr" default:"localhost:50051"`

	// 超时设置
	Timeout time.Duration `yaml:"timeout" default:"30s"`

	// 限流配置
	RateLimit RateLimitConfig `yaml:"rate_limit"`

	// CORS 配置
	CORS CORSConfig `yaml:"cors"`

	// 认证配置
	Auth AuthConfig `yaml:"auth"`
}

// AgentGatewayConfig Agent 网关配置
type AgentGatewayConfig struct {
	// gRPC 服务地址 (接收来自 API Gateway 的请求)
	GRPCAddr string `yaml:"grpc_addr" default:":50051"`

	// MCP 服务器配置
	MCPServers []MCPServerConfig `yaml:"mcp_servers"`

	// 监控配置
	Monitor MonitorConfig `yaml:"monitor"`

	// Agent 池配置
	Pool PoolConfig `yaml:"pool"`
}

// MCPServerConfig MCP 服务器配置
type MCPServerConfig struct {
	Name      string            `yaml:"name"`
	Enabled   bool              `yaml:"enabled" default:"true"`
	Transport string            `yaml:"transport"` // stdio, http, sse
	Command   string            `yaml:"command,omitempty"`
	Args      []string          `yaml:"args,omitempty"`
	Env       map[string]string `yaml:"env,omitempty"`
	URL       string            `yaml:"url,omitempty"`
	Timeout   time.Duration     `yaml:"timeout" default:"30s"`
	Retry     int               `yaml:"retry" default:"3"`
}

// MonitorConfig 监控配置
type MonitorConfig struct {
	// Metrics 配置
	Metrics MetricsConfig `yaml:"metrics"`

	// 追踪配置
	Tracing TracingConfig `yaml:"tracing"`

	// 告警配置
	Alerting AlertingConfig `yaml:"alerting"`
}

// MetricsConfig Prometheus metrics 配置
type MetricsConfig struct {
	Enabled bool   `yaml:"enabled" default:"true"`
	Addr    string `yaml:"addr" default:":9091"`
	Path    string `yaml:"path" default:"/metrics"`
}

// TracingConfig 追踪配置
type TracingConfig struct {
	Enabled  bool    `yaml:"enabled" default:"false"`
	Endpoint string  `yaml:"endpoint"`
	Sampler  float64 `yaml:"sampler" default:"0.1"`
}

// AlertingConfig 告警配置
type AlertingConfig struct {
	Enabled bool `yaml:"enabled" default:"false"`

	// 延迟阈值 (ms)
	LatencyThreshold int64 `yaml:"latency_threshold" default:"5000"`

	// 错误率阈值 (%)
	ErrorRateThreshold float64 `yaml:"error_rate_threshold" default:"5.0"`

	// Webhook URL
	WebhookURL string `yaml:"webhook_url"`
}

// PoolConfig Agent 池配置
type PoolConfig struct {
	// 最小 Agent 数量
	MinSize int `yaml:"min_size" default:"2"`

	// 最大 Agent 数量
	MaxSize int `yaml:"max_size" default:"100"`

	// Agent 空闲超时
	IdleTime time.Duration `yaml:"idle_time" default:"5m"`

	// 等待获取 Agent 超时
	WaitTime time.Duration `yaml:"wait_time" default:"30s"`

	// 健康检查间隔
	HealthTime time.Duration `yaml:"health_time" default:"30s"`
}

// RateLimitConfig 限流配置
type RateLimitConfig struct {
	Enabled bool `yaml:"enabled" default:"true"`

	// 每秒请求数
	RPS int `yaml:"rps" default:"100"`

	// 突发请求数
	Burst int `yaml:"burst" default:"200"`
}

// CORSConfig CORS 配置
type CORSConfig struct {
	AllowOrigins []string `yaml:"allow_origins"`
	AllowMethods []string `yaml:"allow_methods"`
	AllowHeaders []string `yaml:"allow_headers"`
}

// AuthConfig 认证配置
type AuthConfig struct {
	Enabled bool `yaml:"enabled" default:"false"`

	// API Key 认证
	APIKeys []string `yaml:"api_keys"`

	// JWT 配置
	JWT JWTConfig `yaml:"jwt"`
}

// JWTConfig JWT 配置
type JWTConfig struct {
	Secret     string        `yaml:"secret"`
	Expiration time.Duration `yaml:"expiration" default:"24h"`
}

// Load 加载配置文件
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		// 返回默认配置
		return DefaultConfig(), nil
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}

	return &cfg, nil
}

// DefaultConfig 返回默认配置
func DefaultConfig() *Config {
	return &Config{
		APIGateway: APIGatewayConfig{
			HTTPAddr:  ":8080",
			GRPCAddr:  ":9090",
			AgentAddr: "localhost:50051",
			Timeout:   30 * time.Second,
			RateLimit: RateLimitConfig{
				Enabled: true,
				RPS:     100,
				Burst:   200,
			},
		},
		AgentGateway: AgentGatewayConfig{
			GRPCAddr: ":50051",
			Monitor: MonitorConfig{
				MetricsAddr: ":9091",
				Verbose:     false,
			},
			Pool: PoolConfig{
				MaxAgents:           100,
				IdleTimeout:         5 * time.Minute,
				HealthCheckInterval: 30 * time.Second,
			},
		},
	}
}
