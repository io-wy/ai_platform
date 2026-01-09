package main

import (
	"context"
	"flag"
	"os"
	"os/signal"
	"syscall"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/agentflow/gateway/agentgateway"
	"github.com/agentflow/gateway/apigateway"
	"github.com/agentflow/gateway/config"
)

func main() {
	// 命令行参数
	configPath := flag.String("config", "config.yaml", "配置文件路径")
	mode := flag.String("mode", "all", "运行模式: api, agent, all")
	flag.Parse()

	// 配置日志
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	// 加载配置
	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Fatal().Err(err).Msg("加载配置失败")
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 启动服务
	switch *mode {
	case "api":
		startAPIGateway(ctx, cfg)
	case "agent":
		startAgentGateway(ctx, cfg)
	case "all":
		go startAPIGateway(ctx, cfg)
		startAgentGateway(ctx, cfg)
	default:
		log.Fatal().Str("mode", *mode).Msg("未知的运行模式")
	}

	// 优雅关闭
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Info().Msg("正在关闭服务...")
	cancel()
}

func startAPIGateway(ctx context.Context, cfg *config.Config) {
	gw := apigateway.New(cfg.APIGateway)
	if err := gw.Start(ctx); err != nil {
		log.Fatal().Err(err).Msg("API Gateway 启动失败")
	}
}

func startAgentGateway(ctx context.Context, cfg *config.Config) {
	gw := agentgateway.New(cfg.AgentGateway)
	if err := gw.Start(ctx); err != nil {
		log.Fatal().Err(err).Msg("Agent Gateway 启动失败")
	}
}
