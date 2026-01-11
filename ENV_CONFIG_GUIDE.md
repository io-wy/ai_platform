# 环境配置说明

## 🔐 重要提示

**所有 `.env*` 文件都包含敏感的 API Key，已被 `.gitignore` 排除在版本控制之外。**

⚠️ 请勿将包含真实 API Key 的 `.env` 文件提交到 git！

---

## 📋 配置步骤

### 1. 复制模板文件

```bash
# 复制 .env.example 为你需要的配置文件
cp .env.example .env.chatbot
cp .env.example .env.qa
cp .env.example .env.form
```

### 2. 填入你的配置

编辑对应的 `.env.*` 文件，填入你的 API 配置：

```ini
LLM_PROVIDER=openai
LLM_MODEL=your-model-name
LLM_API_KEY=your-api-key-here        # ⚠️ 真实的 API Key
LLM_API_BASE=https://api.xxx.com/v1
```

### 3. 验证配置

```bash
# 测试连接
uv run --no-sync python test_connection.py

# 运行应用
uv run --no-sync python run_chatbot.py --env=.env.chatbot
```

---

## 📁 配置文件说明

| 文件 | 用途 |
|------|------|
| `.env.example` | 配置模板（可提交到 git） |
| `.env.chatbot` | 对话机器人配置 |
| `.env.qa` | 问答系统配置 |
| `.env.form` | 表单提取配置 |
| `.env.executor` | 执行器配置 |
| `.env.planner` | 规划器配置 |

---

## 🛡️ 安全检查清单

- [x] `.gitignore` 包含 `.env*` 规则
- [x] 已移除 git 追踪的 .env 文件
- [x] `.env.example` 不包含真实密钥
- [ ] 确认 `.env.*` 文件不在 git status 中

---

## 🔍 验证是否被忽略

```bash
# 检查文件是否被 .gitignore 忽略
git check-ignore -v .env.chatbot

# 查看 git 追踪的文件
git ls-files | grep ".env"

# 应该只看到 .env.example
```

---

## 🚨 如果不小心提交了密钥

如果你已经提交了包含真实 API Key 的文件：

```bash
# 1. 从历史记录中移除
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env.chatbot" \
  --prune-empty --tag-name-filter cat -- --all

# 2. 强制推送
git push origin --force --all

# 3. 立即更换被泄露的 API Key
```

---

## 📝 本地文件状态

你的本地 `.env.*` 文件会保留，但不会被 git 追踪：

```bash
$ ls -la | grep .env
-rw-r--r--  .env.chatbot    # 本地存在 ✓
-rw-r--r--  .env.example    # git 追踪 ✓
```

```bash
$ git status
# .env.chatbot 不会出现在 git status 中 ✓
```
