# MCP 配置

MCP（Model Context Protocol）服务器给 agent 提供额外工具。配置文件是当前工作目录下的
`mcp_config.json`，独立于 `config.json` 与 CLI 参数。

## 运行中工具刷新

对于在初始化响应中声明 `tools.listChanged` 的 stdio 或支持 SSE 的 HTTP MCP Server，若它在长任务中发送
`notifications/tools/list_changed`，Runtime 会重新请求 `tools/list`，替换该 Server 的工具包装器，
并在下一次模型请求前把更新后的工具集合交给当前 Agent。刷新事件会记录为
`mcp_tools_refreshed`。

该机制不会中断已经发出的模型请求或工具调用；当前一步完成后，下一步才使用新工具目录。
HTTP MCP 会在该能力存在时以 `GET` 建立常驻 SSE 监听；断线后以 500ms 间隔重连。若 Server 不支持
SSE `GET`，客户端保持普通请求/响应模式，仍支持显式刷新与重连刷新。

## 快速开始

### 1. 准备工作

确保你已经安装了 Node.js 和 npm（因为大部分 MCP 服务器使用 Node.js 运行）：

```bash
# 检查是否已安装
node --version
npm --version
```

### 2. 配置 Playwright MCP（已预置）

项目已经预置了 Playwright MCP 配置。只需确保 `mcp_config.json` 文件存在：

```json
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": [
        "@playwright/mcp@latest"
      ],
      "enabled": true
    }
  }
}
```

### 3. 启动系统

运行 Agent 系统，MCP 服务器会自动启动：

```bash
python main.py
```

你将会看到类似输出：

```
正在加载 MCP 服务器...
✅ MCP 服务器 'playwright' 启动成功，提供 5 个工具
✓ 成功启动 1 个 MCP 服务器
✓ 加载了 5 个 MCP 工具
```

---

## Playwright MCP 使用示例

Playwright MCP 提供以下工具（工具名称会自动加上 `mcp_playwright_` 前缀）：

1. **mcp_playwright_navigate** - 导航到网页
2. **mcp_playwright_screenshot** - 截取网页截图
3. **mcp_playwright_click** - 点击页面元素
4. **mcp_playwright_fill** - 填写表单
5. **mcp_playwright_evaluate** - 执行 JavaScript 代码

## Context7 MCP 使用示例 ⭐ 新增

Context7 MCP 提供智能上下文管理功能（工具名称会自动加上 `mcp_context7_` 前缀）：

1. **mcp_context7_store** - 存储上下文信息
2. **mcp_context7_search** - 语义搜索上下文
3. **mcp_context7_retrieve** - 检索相关上下文
4. **mcp_context7_analyze** - 分析上下文关系

### Context7 配置说明

Context7 需要 Upstash API Key：
1. 访问 https://console.upstash.com/
2. 注册并创建账号
3. 创建 Context7 项目
4. 复制 API Key
5. 配置到 `mcp_config.json` 或 `.env` 文件

**配置示例**：
```json
{
  "context7": {
    "command": "npx",
    "args": ["-y", "@upstash/context7-mcp"],
    "env": {
      "UPSTASH_API_KEY": "ctx7sk-your-api-key"
    },
    "enabled": true
  }
}
```

### 示例任务

#### 示例 1：Playwright - 打开网页并截图

```bash
python main.py "打开 https://www.example.com 并截图保存为 example.png"
```

Agent 会自动调用 Playwright MCP 工具来完成任务。

#### 示例 2：Playwright - 自动化表单填写

```
"打开 https://example.com/login，在用户名框中输入 'testuser'，在密码框中输入 'password123'，然后点击登录按钮"
```

#### 示例 3：Playwright - 网页数据提取

```
"访问 https://news.ycombinator.com，提取前 10 条新闻标题"
```

#### 示例 4：Context7 - 存储项目上下文 ⭐ 新增

```bash
python main.py "分析当前项目的架构，并将关键信息存储到 Context7"
```

#### 示例 5：Context7 - 语义搜索 ⭐ 新增

```bash
python main.py "在 Context7 中搜索所有与数据库设计相关的上下文"
```

#### 示例 6：Context7 - 智能检索 ⭐ 新增

```bash
python main.py "从 Context7 获取与当前任务相关的历史决策和代码片段"
```

---

## GitHub 远程 MCP（只读）

项目预置了 `github_readonly`，使用 GitHub 托管的 Streamable HTTP 服务。它只暴露
仓库、Issue 与 Pull Request 的读取工具，可为代码维护任务补充 Issue 需求、历史变更和
PR Diff；返回内容应作为外部观察记录，不能替代本地测试验证。

配置默认启用；首次启动时才会连接 GitHub，未授权前不会暴露 GitHub 工具：

```json
{
  "github_readonly": {
    "transport": "streamable-http",
    "url": "https://api.githubcopilot.com/mcp/readonly",
    "headers": {
      "X-MCP-Toolsets": "repos,issues,pull_requests",
      "X-MCP-Readonly": "true",
      "X-MCP-Lockdown": "true"
    },
    "oauth": true,
    "enabled": true,
    "timeout": 30
  }
}
```

首次连接收到 `401` OAuth Challenge 后，Runtime 会：

1. 发现远程 MCP 的 OAuth 元数据并动态注册客户端；
2. 在本机 `127.0.0.1` 随机端口接收回调，打开浏览器让用户在 GitHub 完成授权；
3. 使用 PKCE 交换授权码；
4. 通过 Windows DPAPI 将 Token 和客户端注册信息加密保存到
   `%LOCALAPPDATA%\\dm-code-agent\\mcp-oauth`，之后自动刷新即将到期的 Token。

Token 不会写入 `mcp_config.json`、Trace 或仓库。若授权服务器不支持动态注册，Runtime 会
停止并提示使用预注册 OAuth App。此时在 `mcp_config.json` 的该服务配置内加入客户端 ID，
并把密钥只放入当前用户环境变量：

```json
"oauth_client_id": "你的 GitHub OAuth App Client ID",
"oauth_client_secret_env": "GITHUB_MCP_OAUTH_CLIENT_SECRET"
```

```cmd
setx GITHUB_MCP_OAUTH_CLIENT_SECRET "你的 OAuth App Client Secret"
```

重新打开终端后再运行 Agent。Client Secret 不会进入配置文件或 Trace。

---

## 如何接入新的 MCP 服务器

### 方法 1：通过配置文件接入（推荐）

这是最简单的方法，适用于大部分场景。

#### 步骤 1：获取 MCP 服务器的配置信息

每个 MCP 服务器都会提供一个 JSON 配置，通常包含：

- `command`: 启动命令（如 `npx`, `node`, `python`）
- `args`: 命令参数列表
- `env`: 环境变量（可选）
- `timeout`: 单次工具调用的超时秒数（可选，默认 5；慢速服务器如浏览器自动化可调大）

服务器进程意外退出时，agent 在下一次工具调用会自动做一次重连；仍失败才把错误反馈给模型。

**示例**：假设你找到了一个新的 MCP 服务器，它的配置如下：

```json
{
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-filesystem@latest"]
}
```

#### 步骤 2：添加到 `mcp_config.json`

编辑项目根目录下的 `mcp_config.json` 文件，在 `mcpServers` 中添加新服务器：

```json
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["@playwright/mcp@latest"],
      "enabled": true
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem@latest"],
      "enabled": true
    }
  }
}
```

#### 步骤 3：重启 Agent 系统

```bash
python main.py
```

系统会自动加载新的 MCP 服务器！

### 方法 2：带环境变量的 MCP 服务器

某些 MCP 服务器需要环境变量（如 API 密钥）：

```json
{
  "mcpServers": {
    "weather-api": {
      "command": "node",
      "args": ["/path/to/weather-mcp-server.js"],
      "env": {
        "WEATHER_API_KEY": "your_api_key_here",
        "DEBUG": "true"
      },
      "enabled": true
    }
  }
}
```

### 方法 3：本地开发的 MCP 服务器

如果你自己开发了一个 MCP 服务器：

```json
{
  "mcpServers": {
    "my-custom-server": {
      "command": "python",
      "args": ["/absolute/path/to/my_mcp_server.py"],
      "enabled": true
    }
  }
}
```

---

## MCP 配置详解

### 配置文件结构

```json
{
  "mcpServers": {
    "<服务器名称>": {
      "command": "<启动命令>",
      "args": ["<参数1>", "<参数2>", ...],
      "env": {
        "<环境变量名>": "<环境变量值>"
      },
      "enabled": true/false
    }
  }
}
```

### 字段说明

| 字段 | 必填 | 说明 | 示例 |
|------|------|------|------|
| `command` | ✅ | 启动命令 | `"npx"`, `"node"`, `"python"` |
| `args` | ✅ | 命令参数列表 | `["@playwright/mcp@latest"]` |
| `env` | ❌ | 环境变量字典 | `{"API_KEY": "xxx"}` |
| `enabled` | ❌ | 是否启用（默认 `true`） | `true` 或 `false` |

### 禁用某个 MCP 服务器

临时禁用某个服务器而不删除配置：

```json
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["@playwright/mcp@latest"],
      "enabled": false  // 设置为 false 即可禁用
    }
  }
}
```

---

## 常见 MCP 服务器推荐

### 1. Playwright（浏览器自动化）✅ 已预置

```json
{
  "playwright": {
    "command": "npx",
    "args": ["@playwright/mcp@latest"],
    "enabled": true
  }
}
```

**用途**：网页自动化、截图、表单填写、数据爬取

### 2. Context7（智能上下文管理）⭐ 已预置

```json
{
  "context7": {
    "command": "npx",
    "args": ["-y", "@upstash/context7-mcp"],
    "env": {
      "UPSTASH_API_KEY": "your_upstash_api_key"
    },
    "enabled": true
  }
}
```

**用途**：智能上下文存储、语义搜索、知识管理、历史决策追溯

**获取 API Key**：https://console.upstash.com/

### 3. Filesystem（文件系统操作）

```json
{
  "filesystem": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem@latest"],
    "enabled": true
  }
}
```

**用途**：高级文件操作、目录遍历、文件监控

### 4. Puppeteer（浏览器自动化）

```json
{
  "puppeteer": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-puppeteer@latest"],
    "enabled": true
  }
}
```

**用途**：与 Playwright 类似，但使用 Puppeteer 引擎

### 5. SQLite（数据库操作）

```json
{
  "sqlite": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-sqlite@latest"],
    "enabled": true
  }
}
```

**用途**：SQLite 数据库查询和管理

---

## 实战示例：接入新的 MCP 服务器

### 场景：接入 GitHub MCP 服务器

根目录 `mcp_config.json` 已配置 `github_readonly`。启动时，首次 GitHub MCP 请求会打开
浏览器进行 OAuth 授权，不需要在配置或 `.env` 中填写 Personal Access Token。

```bash
dm-agent "读取仓库中的 GitHub Issue 与最近关联 PR，梳理修复需求；不要修改远程仓库。"
```

授权完成后，Runtime 只加载仓库、Issue 与 Pull Request 的读取工具。不要将 GitHub MCP
检索结果作为完成验收；仍需对本地工作区运行对应测试。

---

## 常见问题

### Q1: MCP 服务器启动失败怎么办？

**可能原因**：
1. Node.js 未安装或版本过低
2. 网络问题，无法下载 MCP 包
3. 配置文件格式错误

**解决方法**：
```bash
# 检查 Node.js
node --version  # 需要 v14+

# 手动安装 MCP 包
npx @playwright/mcp@latest

# 检查配置文件
python -c "import json; json.load(open('mcp_config.json'))"
```

### Q2: 如何查看 MCP 服务器提供了哪些工具？

启动系统后，选择菜单选项 `3. 查看可用工具列表`，所有 MCP 工具会显示为 `mcp_<服务器名>_<工具名>` 格式。

### Q3: 能否在命令行模式使用 MCP 工具？

可以！MCP 工具在交互模式和命令行模式都可用：

```bash
python main.py "使用 Playwright 打开 https://example.com 并截图"
```

### Q4: MCP 服务器会占用很多资源吗？

MCP 服务器是按需启动的独立进程，只有在系统启动时才会加载。如果不需要某个服务器，可以在配置中设置 `"enabled": false`。

### Q5: 如何调试 MCP 服务器？

MCP 客户端会输出详细的错误信息。你也可以手动测试 MCP 服务器：

```bash
# 手动启动 MCP 服务器
npx @playwright/mcp@latest

# 查看输出
```


---

**需要帮助？** 查看 [文档索引](README.md) 或提交 Issue！
