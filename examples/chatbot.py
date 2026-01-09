"""
对话机器人示例
===============

展示如何使用 AgentFlow 创建一个多轮对话机器人，支持：
- 会话持久化
- 上下文记忆
- 多种对话风格
- 通过 .env 文件配置 LLM

配置文件: .env.chatbot
"""

import asyncio
import os
from datetime import datetime
from typing import Optional
from pathlib import Path

from agentflow import Agent, AgentConfig, LLMConfig, ReasoningPattern
from agentflow.core.config import MemoryConfig
from agentflow.memory.database import ConversationStore
from agentflow.llm.config_loader import LLMConfigLoader, load_llm_config


class ChatBot:
    """交互式对话机器人."""
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        env_file: Optional[str] = None,
        persona: str = "helpful_assistant",
        db_path: str = "conversations.db",
    ):
        """初始化对话机器人.
        
        Args:
            llm_config: LLM 配置对象（优先级最高）
            env_file: 环境配置文件路径（如 .env.chatbot）
            persona: 角色名称
            db_path: 数据库路径
        """
        # 加载 LLM 配置
        if llm_config:
            self.llm_config = llm_config
        elif env_file:
            self.llm_config = LLMConfigLoader.from_env_file(env_file)
        else:
            # 默认从 .env.chatbot 加载，如果不存在则使用默认配置
            self.llm_config = load_llm_config(task="chatbot")
        
        self.persona = persona
        self.db_path = db_path
        
        self.personas = {
            "helpful_assistant": """你是一个友好、专业的AI助手。你的特点是：
- 回答准确、有条理
- 善于解释复杂概念
- 主动提供相关建议
- 语气温和、耐心""",
            
            "coding_expert": """你是一位资深的软件工程师和编程导师。你的特点是：
- 精通多种编程语言和框架
- 代码示例清晰、规范
- 善于解释技术原理
- 注重最佳实践和代码质量""",
            
            "creative_writer": """你是一位富有创意的作家和内容创作者。你的特点是：
- 语言生动、富有感染力
- 善于讲故事和比喻
- 创意丰富、思维开阔
- 能够适应不同的写作风格""",
            
            "business_analyst": """你是一位经验丰富的商业分析师。你的特点是：
- 逻辑清晰、分析严谨
- 熟悉各行业商业模式
- 善于数据驱动决策
- 提供可执行的建议""",
        }
        
        self.agent: Agent = None
        self.store: ConversationStore = None
        self.session_id: str = None
    
    async def initialize(self, session_id: str = None):
        """初始化聊天机器人."""
        # 配置 Agent - 使用加载的 LLM 配置
        config = AgentConfig(
            name=f"ChatBot-{self.persona}",
            llm=self.llm_config,
            memory=MemoryConfig(
                max_short_term_messages=30,
                enable_long_term=False,  # 使用数据库代替
            ),
            pattern=ReasoningPattern.COT,
            system_prompt=self.personas.get(self.persona, self.personas["helpful_assistant"]),
        )
        
        self.agent = Agent(config=config)
        
        # 初始化会话存储
        self.store = ConversationStore(self.db_path)
        
        # 创建或恢复会话
        if session_id:
            self.session_id = session_id
            await self._restore_session()
        else:
            self.session_id = await self.store.create_session(
                name=f"Chat-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                metadata={
                    "persona": self.persona, 
                    "model": self.llm_config.model,
                    "provider": self.llm_config.provider.value,
                },
            )
    
    async def _restore_session(self):
        """从数据库恢复会话历史."""
        messages = await self.store.get_session_messages(self.session_id, limit=20)
        
        for msg in messages:
            if msg["role"] == "user":
                self.agent.conversation.add_user(msg["content"])
            elif msg["role"] == "assistant":
                self.agent.conversation.add_assistant(msg["content"])
        
        print(f"✓ 已恢复 {len(messages)} 条历史消息")
    
    async def chat(self, message: str) -> str:
        """发送消息并获取回复."""
        # 保存用户消息
        await self.store.add_message(self.session_id, "user", message)
        
        # 获取 AI 回复
        response = await self.agent.chat(message)
        
        # 保存 AI 回复
        await self.store.add_message(self.session_id, "assistant", response)
        
        return response
    
    async def get_history(self) -> list[dict]:
        """获取会话历史."""
        return await self.store.get_session_messages(self.session_id)
    
    async def list_sessions(self) -> list[dict]:
        """列出所有会话."""
        return await self.store.get_sessions()
    
    async def search_history(self, query: str) -> list[dict]:
        """搜索历史消息."""
        return await self.store.search_messages(query)
    
    async def close(self):
        """关闭资源."""
        if self.agent:
            await self.agent.close()
        if self.store:
            await self.store.close()
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


async def interactive_chat(env_file: Optional[str] = None):
    """交互式聊天演示.
    
    Args:
        env_file: 可选的环境配置文件路径
    """
    print("=" * 60)
    print("AgentFlow 对话机器人")
    print("=" * 60)
    
    # 显示当前 LLM 配置
    llm_config = load_llm_config(env_file=env_file) if env_file else load_llm_config(task="chatbot")
    print(f"\n当前 LLM 配置:")
    print(f"  Provider: {llm_config.provider.value}")
    print(f"  Model: {llm_config.model}")
    print(f"  Temperature: {llm_config.temperature}")
    
    print("\n可用的角色:")
    print("1. helpful_assistant - 通用助手")
    print("2. coding_expert - 编程专家")
    print("3. creative_writer - 创意写手")
    print("4. business_analyst - 商业分析师")
    
    persona = input("\n选择角色 (1-4, 默认1): ").strip() or "1"
    persona_map = {
        "1": "helpful_assistant",
        "2": "coding_expert",
        "3": "creative_writer",
        "4": "business_analyst",
    }
    
    bot = ChatBot(
        llm_config=llm_config,
        persona=persona_map.get(persona, "helpful_assistant"),
    )
    
    async with bot:
        print(f"\n✓ 已启动 {bot.persona} 模式")
        print("输入 'quit' 退出, 'history' 查看历史, 'search:关键词' 搜索\n")
        
        while True:
            try:
                user_input = input("你: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == "quit":
                    print("再见!")
                    break
                
                if user_input.lower() == "history":
                    history = await bot.get_history()
                    print("\n--- 会话历史 ---")
                    for msg in history[-10:]:  # 最近10条
                        role = "你" if msg["role"] == "user" else "AI"
                        print(f"{role}: {msg['content'][:100]}...")
                    print("----------------\n")
                    continue
                
                if user_input.lower().startswith("search:"):
                    query = user_input[7:].strip()
                    results = await bot.search_history(query)
                    print(f"\n--- 搜索结果: {query} ---")
                    for r in results[:5]:
                        print(f"[{r['session_name']}] {r['role']}: {r['content'][:80]}...")
                    print("------------------------\n")
                    continue
                
                response = await bot.chat(user_input)
                print(f"AI: {response}\n")
                
            except KeyboardInterrupt:
                print("\n再见!")
                break


async def batch_conversation(env_file: Optional[str] = None):
    """批量对话演示."""
    llm_config = load_llm_config(env_file=env_file) if env_file else load_llm_config(task="chatbot")
    
    conversations = [
        ("helpful_assistant", [
            "你好，我想了解一下机器学习的基础知识",
            "监督学习和无监督学习有什么区别？",
            "能推荐一些入门资源吗？",
        ]),
        ("coding_expert", [
            "如何用 Python 实现一个简单的 HTTP 服务器？",
            "能加上路由功能吗？",
            "如何处理 POST 请求？",
        ]),
    ]
    
    for persona, messages in conversations:
        print(f"\n{'=' * 50}")
        print(f"场景: {persona}")
        print("=" * 50)
        
        bot = ChatBot(llm_config=llm_config, persona=persona)
        async with bot:
            for msg in messages:
                print(f"\n用户: {msg}")
                response = await bot.chat(msg)
                print(f"AI: {response[:300]}...")


async def resume_session(env_file: Optional[str] = None):
    """恢复历史会话演示."""
    llm_config = load_llm_config(env_file=env_file) if env_file else load_llm_config(task="chatbot")
    store = ConversationStore("conversations.db")
    
    try:
        sessions = await store.get_sessions()
        
        if not sessions:
            print("没有历史会话，创建新会话...")
            bot = ChatBot(llm_config=llm_config)
            async with bot:
                await bot.chat("这是一条测试消息")
                print(f"创建了会话: {bot.session_id}")
            return
        
        print("历史会话:")
        for i, session in enumerate(sessions):
            print(f"{i + 1}. {session['name']} ({session['updated_at'][:10]})")
        
        choice = input("\n选择要恢复的会话 (数字): ").strip()
        try:
            idx = int(choice) - 1
            session_id = sessions[idx]["id"]
            
            bot = ChatBot(llm_config=llm_config)
            async with bot:
                await bot.initialize(session_id=session_id)
                
                print("\n继续对话:")
                response = await bot.chat("你还记得我们之前聊的内容吗？")
                print(f"AI: {response}")
                
        except (ValueError, IndexError):
            print("无效选择")
    
    finally:
        await store.close()


if __name__ == "__main__":
    import sys
    
    # 解析命令行参数
    env_file = None
    mode = "interactive"
    
    for arg in sys.argv[1:]:
        if arg.startswith("--env="):
            env_file = arg.split("=", 1)[1]
        elif arg in ("batch", "resume", "interactive"):
            mode = arg
    
    print(f"使用配置文件: {env_file or '.env.chatbot (默认)'}")
    
    if mode == "batch":
        asyncio.run(batch_conversation(env_file))
    elif mode == "resume":
        asyncio.run(resume_session(env_file))
    else:
        asyncio.run(interactive_chat(env_file))
