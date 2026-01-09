package agentgateway

import (
	"context"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/rs/zerolog/log"

	"github.com/agentflow/gateway/config"
)

// ============================================================================
// Agent 监控系统
// ============================================================================

// Monitor Agent 监控器
type Monitor struct {
	cfg config.MonitorConfig

	// Prometheus 指标
	requestCounter   *prometheus.CounterVec
	latencyHistogram *prometheus.HistogramVec
	errorCounter     *prometheus.CounterVec
	activeGauge      *prometheus.GaugeVec

	// 内部统计
	stats     *Stats
	alerts    []Alert
	alertsMu  sync.RWMutex
	listeners []AlertListener
}

// Stats 统计数据
type Stats struct {
	TotalRequests  int64
	TotalErrors    int64
	ActiveSessions int64
	ActiveAgents   int64
	AvgLatencyMs   float64
	P99LatencyMs   float64
	StartTime      time.Time

	// 滑动窗口统计
	windowRequests []windowSample
	windowLatency  []time.Duration
	windowMu       sync.RWMutex
}

type windowSample struct {
	time  time.Time
	count int64
}

// Alert 告警
type Alert struct {
	ID        string
	Level     string // info, warning, critical
	Type      string
	Message   string
	Timestamp time.Time
	Resolved  bool
}

// AlertListener 告警监听器
type AlertListener func(alert Alert)

// NewMonitor 创建监控器
func NewMonitor(cfg config.MonitorConfig) *Monitor {
	m := &Monitor{
		cfg:   cfg,
		stats: &Stats{StartTime: time.Now()},
	}

	if cfg.Metrics.Enabled {
		m.initPrometheus()
	}

	return m
}

// initPrometheus 初始化 Prometheus 指标
func (m *Monitor) initPrometheus() {
	m.requestCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "agentflow_requests_total",
			Help: "Total number of requests",
		},
		[]string{"method", "status"},
	)

	m.latencyHistogram = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "agentflow_request_latency_seconds",
			Help:    "Request latency in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"method"},
	)

	m.errorCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "agentflow_errors_total",
			Help: "Total number of errors",
		},
		[]string{"method", "type"},
	)

	m.activeGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "agentflow_active",
			Help: "Current active counts",
		},
		[]string{"type"},
	)

	prometheus.MustRegister(
		m.requestCounter,
		m.latencyHistogram,
		m.errorCounter,
		m.activeGauge,
	)
}

// Start 启动监控
func (m *Monitor) Start(ctx context.Context) {
	// 启动 Prometheus HTTP 服务
	if m.cfg.Metrics.Enabled {
		go m.serveMetrics()
	}

	// 启动定期统计
	go m.collectStats(ctx)

	// 启动告警检查
	if m.cfg.Alerting.Enabled {
		go m.checkAlerts(ctx)
	}

	log.Info().
		Bool("metrics", m.cfg.Metrics.Enabled).
		Bool("tracing", m.cfg.Tracing.Enabled).
		Bool("alerting", m.cfg.Alerting.Enabled).
		Msg("监控系统已启动")
}

// serveMetrics 提供 Prometheus 指标
func (m *Monitor) serveMetrics() {
	mux := http.NewServeMux()
	mux.Handle(m.cfg.Metrics.Path, promhttp.Handler())

	server := &http.Server{
		Addr:    m.cfg.Metrics.Addr,
		Handler: mux,
	}

	log.Info().Str("addr", m.cfg.Metrics.Addr).Msg("Prometheus 指标服务启动")

	if err := server.ListenAndServe(); err != http.ErrServerClosed {
		log.Error().Err(err).Msg("指标服务错误")
	}
}

// collectStats 收集统计数据
func (m *Monitor) collectStats(ctx context.Context) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			m.updateStats()
		}
	}
}

// updateStats 更新统计
func (m *Monitor) updateStats() {
	m.stats.windowMu.Lock()
	defer m.stats.windowMu.Unlock()

	now := time.Now()
	windowSize := 5 * time.Minute

	// 清理过期的样本
	var validSamples []windowSample
	for _, s := range m.stats.windowRequests {
		if now.Sub(s.time) < windowSize {
			validSamples = append(validSamples, s)
		}
	}
	m.stats.windowRequests = validSamples

	// 计算平均延迟
	var validLatency []time.Duration
	for _, l := range m.stats.windowLatency {
		validLatency = append(validLatency, l)
	}

	if len(validLatency) > 0 {
		var sum time.Duration
		for _, l := range validLatency {
			sum += l
		}
		m.stats.AvgLatencyMs = float64(sum.Milliseconds()) / float64(len(validLatency))

		// P99
		if len(validLatency) >= 100 {
			// 简单实现：取第99个百分位
			idx := int(float64(len(validLatency)) * 0.99)
			m.stats.P99LatencyMs = float64(validLatency[idx].Milliseconds())
		}
	}
}

// checkAlerts 检查告警
func (m *Monitor) checkAlerts(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			m.evaluateAlertRules()
		}
	}
}

// evaluateAlertRules 评估告警规则
func (m *Monitor) evaluateAlertRules() {
	// 错误率告警
	errorRate := float64(m.stats.TotalErrors) / float64(m.stats.TotalRequests+1)
	if errorRate > 0.1 { // 10% 错误率
		m.triggerAlert(Alert{
			Level:   "warning",
			Type:    "high_error_rate",
			Message: "错误率超过 10%",
		})
	}

	// 延迟告警
	if m.stats.P99LatencyMs > 5000 { // P99 > 5s
		m.triggerAlert(Alert{
			Level:   "warning",
			Type:    "high_latency",
			Message: "P99 延迟超过 5 秒",
		})
	}
}

// triggerAlert 触发告警
func (m *Monitor) triggerAlert(alert Alert) {
	alert.ID = time.Now().Format("20060102150405")
	alert.Timestamp = time.Now()

	m.alertsMu.Lock()
	m.alerts = append(m.alerts, alert)
	m.alertsMu.Unlock()

	// 通知监听器
	for _, listener := range m.listeners {
		go listener(alert)
	}

	log.Warn().
		Str("level", alert.Level).
		Str("type", alert.Type).
		Str("message", alert.Message).
		Msg("告警触发")
}

// ============================================================================
// 记录方法
// ============================================================================

// RecordRequest 记录请求
func (m *Monitor) RecordRequest(method string) {
	atomic.AddInt64(&m.stats.TotalRequests, 1)

	if m.requestCounter != nil {
		m.requestCounter.WithLabelValues(method, "success").Inc()
	}

	m.stats.windowMu.Lock()
	m.stats.windowRequests = append(m.stats.windowRequests, windowSample{
		time:  time.Now(),
		count: 1,
	})
	m.stats.windowMu.Unlock()
}

// RecordError 记录错误
func (m *Monitor) RecordError(method string, err error) {
	atomic.AddInt64(&m.stats.TotalErrors, 1)

	if m.errorCounter != nil {
		m.errorCounter.WithLabelValues(method, "error").Inc()
	}

	log.Error().Err(err).Str("method", method).Msg("请求错误")
}

// RecordLatency 记录延迟
func (m *Monitor) RecordLatency(method string, latency time.Duration) {
	if m.latencyHistogram != nil {
		m.latencyHistogram.WithLabelValues(method).Observe(latency.Seconds())
	}

	m.stats.windowMu.Lock()
	m.stats.windowLatency = append(m.stats.windowLatency, latency)
	// 保持窗口大小
	if len(m.stats.windowLatency) > 1000 {
		m.stats.windowLatency = m.stats.windowLatency[500:]
	}
	m.stats.windowMu.Unlock()
}

// SetActiveCount 设置活跃数
func (m *Monitor) SetActiveCount(typ string, count int64) {
	switch typ {
	case "sessions":
		atomic.StoreInt64(&m.stats.ActiveSessions, count)
	case "agents":
		atomic.StoreInt64(&m.stats.ActiveAgents, count)
	}

	if m.activeGauge != nil {
		m.activeGauge.WithLabelValues(typ).Set(float64(count))
	}
}

// ============================================================================
// 查询方法
// ============================================================================

// GetMetrics 获取指标
func (m *Monitor) GetMetrics(names []string) map[string]float64 {
	metrics := make(map[string]float64)

	// 如果没有指定，返回所有基本指标
	if len(names) == 0 {
		names = []string{
			"total_requests",
			"total_errors",
			"active_sessions",
			"active_agents",
			"avg_latency_ms",
			"p99_latency_ms",
			"uptime_seconds",
		}
	}

	for _, name := range names {
		switch name {
		case "total_requests":
			metrics[name] = float64(atomic.LoadInt64(&m.stats.TotalRequests))
		case "total_errors":
			metrics[name] = float64(atomic.LoadInt64(&m.stats.TotalErrors))
		case "active_sessions":
			metrics[name] = float64(atomic.LoadInt64(&m.stats.ActiveSessions))
		case "active_agents":
			metrics[name] = float64(atomic.LoadInt64(&m.stats.ActiveAgents))
		case "avg_latency_ms":
			metrics[name] = m.stats.AvgLatencyMs
		case "p99_latency_ms":
			metrics[name] = m.stats.P99LatencyMs
		case "uptime_seconds":
			metrics[name] = time.Since(m.stats.StartTime).Seconds()
		case "error_rate":
			total := float64(atomic.LoadInt64(&m.stats.TotalRequests))
			if total > 0 {
				errors := float64(atomic.LoadInt64(&m.stats.TotalErrors))
				metrics[name] = errors / total
			}
		}
	}

	return metrics
}

// GetAlerts 获取告警
func (m *Monitor) GetAlerts(resolved bool) []Alert {
	m.alertsMu.RLock()
	defer m.alertsMu.RUnlock()

	var result []Alert
	for _, a := range m.alerts {
		if a.Resolved == resolved {
			result = append(result, a)
		}
	}
	return result
}

// AddAlertListener 添加告警监听器
func (m *Monitor) AddAlertListener(listener AlertListener) {
	m.listeners = append(m.listeners, listener)
}

// ============================================================================
// 分布式追踪 (简化实现)
// ============================================================================

// Span 追踪 Span
type Span struct {
	TraceID   string
	SpanID    string
	ParentID  string
	Operation string
	StartTime time.Time
	EndTime   time.Time
	Tags      map[string]string
	Logs      []SpanLog
}

// SpanLog Span 日志
type SpanLog struct {
	Timestamp time.Time
	Message   string
}

// StartSpan 开始 Span
func (m *Monitor) StartSpan(ctx context.Context, operation string) (*Span, context.Context) {
	span := &Span{
		TraceID:   generateID(),
		SpanID:    generateID(),
		Operation: operation,
		StartTime: time.Now(),
		Tags:      make(map[string]string),
	}

	// 从 context 获取父 Span
	if parentSpan, ok := ctx.Value(spanKey{}).(*Span); ok {
		span.ParentID = parentSpan.SpanID
		span.TraceID = parentSpan.TraceID
	}

	return span, context.WithValue(ctx, spanKey{}, span)
}

// FinishSpan 结束 Span
func (m *Monitor) FinishSpan(span *Span) {
	span.EndTime = time.Now()

	if m.cfg.Tracing.Enabled {
		// TODO: 发送到追踪后端 (Jaeger/Zipkin)
		log.Debug().
			Str("trace_id", span.TraceID).
			Str("span_id", span.SpanID).
			Str("operation", span.Operation).
			Dur("duration", span.EndTime.Sub(span.StartTime)).
			Msg("Span 完成")
	}
}

type spanKey struct{}

func generateID() string {
	return time.Now().Format("20060102150405.000000")
}

// ============================================================================
// 健康检查
// ============================================================================

// HealthStatus 健康状态
type HealthStatus struct {
	Status    string           `json:"status"`
	Checks    map[string]Check `json:"checks"`
	Timestamp time.Time        `json:"timestamp"`
}

// Check 检查项
type Check struct {
	Status  string `json:"status"`
	Message string `json:"message,omitempty"`
}

// GetHealth 获取健康状态
func (m *Monitor) GetHealth() HealthStatus {
	status := HealthStatus{
		Status:    "healthy",
		Checks:    make(map[string]Check),
		Timestamp: time.Now(),
	}

	// 检查错误率
	errorRate := float64(m.stats.TotalErrors) / float64(m.stats.TotalRequests+1)
	if errorRate > 0.5 {
		status.Status = "unhealthy"
		status.Checks["error_rate"] = Check{Status: "fail", Message: "错误率过高"}
	} else if errorRate > 0.1 {
		status.Status = "degraded"
		status.Checks["error_rate"] = Check{Status: "warn", Message: "错误率偏高"}
	} else {
		status.Checks["error_rate"] = Check{Status: "pass"}
	}

	// 检查延迟
	if m.stats.P99LatencyMs > 10000 {
		status.Status = "degraded"
		status.Checks["latency"] = Check{Status: "warn", Message: "延迟过高"}
	} else {
		status.Checks["latency"] = Check{Status: "pass"}
	}

	return status
}
