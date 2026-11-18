# Ego Pod (`v0.2.0`)

`ego-pod` 是 Ego 项目本地、隐私优先的个性化 AI 模型修炼管线（Dojo）：从人类原型（Human Prime）的五类个人数据出发，经由 **Collector → Compiler（深层专精度蒸馏）→ Trainer（多规格 QLoRA）→ Vault（双层完整性签名与版本仓库）** 流水线，复刻产出可供终端宿主（Host）按需装载的多维稀疏网格 `Ego_Package`。

## 安装

```bash
pip install ./release/dev
```

安装后在当前 Python 环境注册 `ego` 命令。

## 快速开始

```bash
# 1. 生成默认配置（或使用 --all-editions 生成四档规格，或 --interactive 交互式向导）
ego init
ego init --all-editions --force
ego init --interactive

# 2. 执行端到端修炼（支持 --update 增量采集编译与 --parallel-editions 多规格并行训练）
ego train
ego train --update

# 3. 按指定规格与功能分区深度加载对话
ego run --edition normal
ego run --edition plus --zone coding:3 --zone knowledge:1
ego run --edition lite --kernel-only

# 4. 查看历史版本快照、验签回滚或彻底清理缓存
ego vault list
ego vault rollback 0.2.0-0001
ego vault purge --all
```

## 命令一览

| 命令 | 说明 |
| --- | --- |
| `ego init [--force] [--all-editions] [--interactive]` | 初始化工作目录 `ego-config.yaml`（默认非交互单规格；支持全规格模板与交互向导） |
| `ego train [--config PATH] [--update] [--parallel-editions]` | 执行采集 → 深层专精度网格蒸馏 → 多规格 QLoRA 训练 → 签名与历史归档 |
| `ego run [--edition NAME] [--zone ZONE:DEPTH ...] [--kernel-only]` | 加载指定 Edition 与选定网格深度并启动交互式 REPL |
| `ego vault {list,rollback,purge}` | 管理 `ego-output/.history/` 版本快照、验签原子回滚与隐私缓存清理 |

全局参数支持 `ego --version` 与 `ego --offline`（或环境变量 `EGO_OFFLINE=1`）。

## 多规格预设与受控基座白名单

### 四档内置 Edition 预设（支持自定义 Edition）

| Edition | 默认基座 | 默认 LoRA (`r / alpha`) | 网格最大深度 | 产物体积上限 |
| --- | --- | --- | --- | --- |
| `lite` | `qwen3.5-0.8B` | `8 / 16` | `L1`（`Kernel + L1`） | `<= 64 MB` |
| `normal` | `qwen3.5-2B`（`ego init` 默认模板使用 `qwen3.5-0.8B`） | `16 / 32` | `L2`（`Kernel + L1..L2`） | `<= 256 MB` |
| `plus` | `qwen3.5-4B` | `32 / 64` | `L3`（`Kernel + L1..L3`） | `<= 1024 MB` |
| `super` | `qwen3.5-27B` | `64 / 128` | `L3`（`Kernel + L1..L3`） | `<= 4096 MB` |

此外可在 `editions:` 下声明满足 `^[a-z0-9][a-z0-9-]{0,31}$` 的自定义规格，并覆盖 `base`、`lora_rank`、`lora_alpha`、`max_level`、`max_size_mb` 与 `zones` 子集。

### 受控基座模型白名单

- **Qwen 3.5**：`qwen3.5-0.8B`, `qwen3.5-2B`, `qwen3.5-4B`, `qwen3.5-9B`, `qwen3.5-27B`
- **Gemma 4**：`gemma4-2B`, `gemma4-9B`, `gemma4-27B`
- **Phi-4**：`phi4-mini`, `phi4-14B`

训练期间若遭遇显存或内存不足（OOM），编排层会自动将 `batch_size` 减半重试（最多 3 次），并在产物配置中记录 `effective_batch_size` 与 `oom_retries`。

## 数据源与增量采集

- **`local`**：本地目录递归采集，配合 `.ego/collection-manifest.json` 内容 SHA-256 追踪实现 `--update` 增量缓存复用。
- **`url`**：受离线模式守卫与 `allowed_hosts` 白名单约束的 HTTPS 数据源，支持 `.ego/collector-checkpoints.json` ETag / HTTP Range 断点续采。
- **`s3` / `oss`**：云端对象存储适配器。云凭证**严禁**写入 `ego-config.yaml`（内嵌明文密钥会被配置校验器直接以退出码 2 拒绝），仅从环境变量 `EGO_CLOUD_ACCESS_KEY_ID`、`EGO_CLOUD_SECRET_ACCESS_KEY`、`EGO_CLOUD_ENDPOINT_URL` 读取，并在落盘前主动擦除环境变量敏感串。
- **插件扩展**：支持通过 Python `entry_points(group="ego_pod.collectors")` 或 `register_collector_plugin` 注册自定义数据源适配器。

## 深层二维稀疏网格（Sparse_Grid）与供应链签名

- **无泄漏专精度分层（`L1..L3`）**：Compiler 在切分确定性留出验证集之后，**仅在训练子集上**拟合各功能分区词表 IDF 与分位数切点，结合结构/代码/推理特征将分区样本分层为连续无跳号的 `1..K`（`K <= max_level`）。
- **两级完整性与签名**：`manifest.json` 内置 Tier 1 `integrity.cell_digests`（覆盖所有已构建 Edition 的 `.safetensors` 适配器 SHA-256）；配置 `EGO_SIGNING_SEED_HEX` 与 `EGO_SIGNER_FAY_ID` 时，自动生成与 `FayGer` 装载协议逐字节对齐的 Tier 2 RFC 8785 JCS + Ed25519 签名（域 `ifay.ego.package-signing.v1`）。`EGO_SIGNER_FAY_ID` 必须为规范 iFay FayID（`ifay_` + 至少 16 位小写 base32 `[a-z2-7]`），否则签名阶段即报错。
- **跨 Cell 一致性评分**：训练完成后，`ego train` 在本地留出集提示上比较每个 Edition 的 Kernel-only 与全量加载回复，取各 Edition 最小值写入 `fidelity_report.cell_consistency_score` 并重新签名（`FayGer` 生产准入要求 `>= 0.70`）。Stub 后端产物不评分（保持 `null`），因为占位回复不代表模型行为。
- **本地保真度报告**：自动生成纯聚合标量的 `fidelity_report`（留出集损失、困惑度、跨 Cell 一致性与版本间漂移），零原始语料入包。

## 真实训练后端安装（Optional Extras）

默认安装仅含轻量核心依赖并以 `Stub_Backend` 跑通端到端流程；按目标硬件可安装对应真实训练后端：

```bash
pip install "./release/dev[cuda]"      # NVIDIA CUDA QLoRA (peft + bitsandbytes)
pip install "./release/dev[unsloth]"   # NVIDIA CUDA Unsloth 加速后端
pip install "./release/dev[mlx]"       # Apple Silicon MLX 原生 LoRA 训练 + PEFT 转置导出
pip install "./release/dev[peft]"      # CPU / 跨平台 PEFT 训练
pip install "./release/dev[all]"       # 安装全部可选训练后端
```

探测优先级固定为：`unsloth` → `peft-cuda` → `mlx` → `peft-cpu` → `stub`。

## 开发与测试

```bash
pip install -e "./release/dev[test]"
pytest release/dev/tests
```
