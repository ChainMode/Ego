# Ego Pod

Ego Pod MVP 是 Ego 项目蓝图 **Phase 1（最小可用版本）** 的实现：一个开箱即用、全自动、隐私优先的个性化 AI 模型训练管线。

## 安装

```bash
pip install ./release/dev
```

安装后会在当前 Python 环境注册 `ego` 可执行命令。

## 快速开始

```bash
# 1. 在工作目录生成默认配置 ego-config.yaml
ego init

# 2. 编辑 ego-config.yaml，填入本地数据源目录，然后训练
ego train

# 3. 与训练产出的 Ego 模型对话
ego run
```

## 命令一览

| 命令 | 说明 |
| --- | --- |
| `ego init` | 在当前目录生成默认 `ego-config.yaml` |
| `ego train` | 依次执行采集 → 语料构建 → QLoRA 训练 → 产出 Ego 包 |
| `ego run` | 加载 `ego-output/` 中的 Ego 包并进入交互式对话 |

更多用法见 `ego --help` 与各子命令的 `ego <subcommand> --help`。

## 真实训练后端 / Real Training Backends

默认安装（`pip install ./release/dev`）只引入轻量核心依赖，端到端可跑通的训练路径为 **Stub_Backend**（模拟训练，不加载真实权重）。要启用真实 QLoRA 训练，需按硬件额外安装对应的可选依赖分组（extras）。

### 安装可选后端 / Install extras

```bash
# NVIDIA CUDA：QLoRA 4-bit 真实训练（peft + transformers + accelerate + datasets + bitsandbytes）
pip install "./release/dev[cuda]"

# CPU / Apple Silicon：PEFT 真实训练（不含仅 CUDA 的 bitsandbytes，以受支持精度训练）
pip install "./release/dev[peft]"

# 仅单独补 4-bit 量化库（NVIDIA CUDA）
pip install "./release/dev[bitsandbytes]"

# Unsloth 加速后端（NVIDIA CUDA，自带 transformers/peft/bitsandbytes 依赖链）
pip install "./release/dev[unsloth]"

# Apple MLX 后端（仅供探测；MLX 真实训练本特性暂未落地，见下方说明）
pip install "./release/dev[mlx]"

# 一次性安装全部可选后端
pip install "./release/dev[all]"
```

### 后端选择优先级 / Backend selection priority

`ego train` 按固定优先级探测可用后端，选中第一个「依赖已安装且目标硬件可用」的后端；全部不可用时回退到始终可用的 Stub_Backend，使训练端到端跑通至退出码 0：

| 优先级 | 后端 | `training_backend` 标识 | 目标硬件 | 所需 Training_Dependencies |
| --- | --- | --- | --- | --- |
| 1 | Unsloth | `unsloth` | NVIDIA CUDA | `unsloth`（含 transformers/peft/bitsandbytes） |
| 2 | HuggingFace PEFT (CUDA) | `peft-cuda` | NVIDIA CUDA | `peft` + `bitsandbytes`（即 `[cuda]` 分组） |
| 3 | Apple MLX | `mlx` | Apple Silicon | `mlx-lm`（仅探测，见说明） |
| 4 | HuggingFace PEFT (CPU) | `peft-cpu` | CPU | `peft`（无需 bitsandbytes） |
| 5 | Stub | `stub` | 任意（始终可用） | 无 |

选中后端的标识会写入产物 `ego-output/manifest.json` 的 `training_backend` 字段，`ego run` 据此选择匹配的推理路径。

### 最低硬件要求 / Hardware requirements

- **Peft_Cuda_Backend（`peft-cuda`）**：需要一块支持 CUDA 的 NVIDIA GPU 与可用的 CUDA 运行时。基座为 Qwen 3.5 0.8B，启用 4-bit 量化后显存占用较低，但仍建议具备数 GB 可用显存以容纳量化权重、LoRA 适配器与训练激活值。
- **Peft_Cpu_Backend（`peft-cpu`）**：无需 GPU，可在普通 CPU 上运行（含 Apple Silicon）。以全精度/受支持精度加载基座，内存占用与训练耗时显著高于 CUDA 路径，适合小语料的验证性训练。

### 4-bit 量化在 CPU 与 CUDA 上的可用性差异 / 4-bit availability

- **CUDA**：通过 `bitsandbytes` 提供 4-bit（NF4）量化，按 QLoRA 在量化后的基座骨干上训练 LoRA 适配器，显著降低显存占用。需安装 `bitsandbytes`（包含在 `[cuda]` 分组中）。
- **CPU**：`bitsandbytes` 的 4-bit 量化在纯 CPU / macOS 平台通常不可用。此时 Peft_Cpu_Backend 会**自动回退**为以 CPU 受支持的精度加载基座（不使用 bitsandbytes 4-bit 量化），并在标准输出记录实际采用的加载精度。因此 `[peft]` 分组**不包含** `bitsandbytes`，CPU/Mac 用户安装 `peft` 即可工作。

### CUDA 上的非确定性与可复现性 / Determinism on CUDA

训练开始前会为 Python、NumPy 与训练框架（torch）的随机数发生器设定确定性种子，使得在相同基座、相同语料、相同超参与相同种子下重复训练产出语义等价的 LoRA 适配器。但在 CUDA 设备上：

- 部分 CUDA / cuDNN 算子内核存在非确定性实现（如某些归约、原子加法操作），即便固定种子也可能在逐位层面产生微小差异；
- 不同的 GPU 型号、驱动版本、CUDA / cuDNN 版本、以及混合精度计算路径都可能影响数值结果。

因此本系统在 CUDA 上**不保证逐位（bitwise）可复现**，仅保证在固定种子与相同环境下产出语义等价的适配器。需要更强可复现性时，可在 CPU 路径上训练，或固定完整的软硬件与库版本环境。

### Apple Silicon 说明 / Apple Silicon note

本特性范围内**未落地 MLX 真实训练**（mlx 张量栈与 torch/peft 不兼容，需独立实现）。由于探测优先级中 Apple MLX（第 3）先于 PEFT-CPU（第 4），若在 Apple Silicon 上安装了 `mlx` extra，MLX 后端会被优先选中并以明确的 `TrainerError`（退出码 1）显式失败，而不会静默产出无效适配器。

因此，**欲在 Apple Silicon 上用 PEFT-CPU 进行真实训练，请安装 `peft` extra 而不要安装 `mlx` extra**（直至 MLX 真实训练落地）：

```bash
pip install "./release/dev[peft]"   # 正确：选中 Peft_Cpu_Backend
# 不要执行 pip install "./release/dev[mlx]"，否则 MLX 会被优先选中并显式失败
```

## 二维分层人格 / Layered Adapter（二维网格）

除"单个扁平 LoRA 适配器"外，Ego 还支持把人格组织成一张**二维、类脑的稀疏网格（Sparse_Grid）**，使产出的 Ego 包能自描述其结构，从而支持"按需加载"——既能小内核 + 全分区浅加载（小而广），也能单分区深加载（大而专）。

### 二维类脑人格模型 / Brain-like 2D model

人格由两条正交的轴加上一个共享中心定义：

- **Kernel（内核，Level 0，与分区无关）**：唯一、最基础、最一般、**与分区无关（zone-agnostic）**的基础细胞，被所有功能分区共享，在**任何一次加载中恒被加载**。
- **Functional_Zone（功能分区，角向轴，相互独立）**：类比大脑脑区的领域分区，每个分区主管一个不同的领域（如 `coding` / `knowledge`），分区之间彼此独立。每个分区由唯一的 **Zone_Name** 标识，取值由小写字母、数字与连字符组成（`[a-z0-9-]`）。
- **Level（层级，径向轴，coarse→fine）**：在**同一分区内部**由内向外的深度，从 `1` 开始。Level 1 在该领域内最粗、最一般；Level 越大越细、越专精，但始终服务于同一领域。Level 是**每分区独立**的。

网格的最小可寻址单元是 **Cell = (Functional_Zone, Level)**，每个 Cell 对应一个独立的 LoRA 适配器文件；Kernel 是特殊的与分区无关的 Level-0 Cell。网格是**稀疏的**——不同分区可拥有不同的最大 Level，但**在一个存在的分区内部，Level 从 1 到该分区最大 Level 连续**（每分区层级前缀）。

### Zone_Config：可配置的领域分类法 / Zone configuration

功能分区分类法与语料的 Data_Category（`sense` / `action` / `thought` / `skill` / `credentials`）**解耦**，通过 `ego-config.yaml` 中可选的 `zones:` 块声明：

```yaml
# 可选：功能分区分类法（Zone_Config）。省略则使用内置默认基线（见下）。
zones:
  # 声明的功能分区集合（Zone_Name 由小写字母、数字、连字符组成且唯一）。
  zones: [coding, knowledge]
  # Data_Category → 分区路由基线；未列出的类别路由到 Kernel。
  routing:
    skill: coding
    action: coding
    thought: knowledge
    # sense / credentials 未列出 → 归入 Kernel
```

- `zones.zones`：声明的 Functional_Zone 名称集合；每个 Zone_Name 须匹配 `^[a-z0-9-]+$` 且唯一。
- `zones.routing`：Data_Category → Zone_Name 的路由基线；未在 `routing` 中出现的类别（以及未匹配任何分区规则的样本）一律归入 **Kernel**。
- 校验失败（非法 Zone_Name、重复 Zone_Name、`routing` 取值指向未声明分区）会在检测到第一个错误时以**退出码 2** 结束。

**省略 `zones:` 块时使用内置默认基线**（最小且可辩护，全部可经 Zone_Config 覆盖）：

| 默认分区 | 含义（示例领域） | 由哪些 Data_Category 路由进来 |
| --- | --- | --- |
| `coding` | 编程/技能与操作 | `skill`、`action` |
| `knowledge` | 知识/思考 | `thought` |
| （Kernel，Level 0） | 与分区无关的共享基础 | `sense`、`credentials`，以及**未匹配任何分区**的样本 |

该默认基线天然保证 **Kernel 非空**（`sense` / `credentials` 恒路由到 Kernel）。现有不含 `zones:` 的配置照常工作，无需修改。

### `manifest.json` 的 Grid_Descriptor 结构 / Grid descriptor

网格包在 `manifest.json` 顶层新增**可选** `grid` 键来自描述这张网格——**`grid` 键缺省即视为 Flat_Package**。其值按 Edition 组织，每个 Edition 描述其 Kernel 与各分区的 Level 列表与各 Cell 适配器文件**相对该 Edition 目录**的相对路径，使加载方仅凭 manifest 即可枚举全部 Cell 而无需扫描 `adapters/` 目录：

```json
{
  "version": "0.1.0",
  "created_at": "2025-11-25T08:00:00Z",
  "base_model": "qwen3.5-0.8B",
  "editions": ["normal"],
  "training_backend": "peft-cpu",
  "grid": {
    "normal": {
      "kernel": { "path": "adapters/kernel.safetensors" },
      "zones": {
        "coding": [
          { "level": 1, "path": "adapters/coding/L1.safetensors" }
        ],
        "knowledge": [
          { "level": 1, "path": "adapters/knowledge/L1.safetensors" }
        ]
      }
    }
  }
}
```

每个分区的数组按 `level` 升序且连续 `1..max`；不含任何 Cell 的分区表达为空数组 `[]`。既有的五个 manifest 字段（`version` / `created_at` / `base_model` / `editions` / `training_backend`）名称、类型、取值约定与字段顺序均不变，`grid` 仅在**存在至少一个分区 Cell**时追加。

### 磁盘布局 / On-disk layout

| Cell | 磁盘相对路径（相对 `<edition>/`） |
| --- | --- |
| Kernel（网格包） | `adapters/kernel.safetensors` |
| 分区 Cell | `adapters/<zone>/L<level>.safetensors` |
| 退化单 Cell（纯 Flat / kernel-only） | `adapters/ego-lora.safetensors` |

`Zone_Name` 被约束为 `[a-z0-9-]`，因此可直接安全地用作目录名，无需任何转义；`L<level>` 前缀避免纯数字文件名并使层级在文件系统中自描述。一个含分区的网格包磁盘布局示例：

```
ego-output/
├── manifest.json
└── normal/
    ├── config.json
    └── adapters/
        ├── kernel.safetensors
        ├── coding/
        │   └── L1.safetensors
        └── knowledge/
            └── L1.safetensors
```

### Flat_Package 与 Grid_Package 的兼容关系 / Compatibility

- **缺 `grid` 键 ⇒ 单个 Kernel Cell**：一个不含 `grid` 键的包（Flat_Package）被加载为一个**单 Kernel Cell** 的网格，其 Kernel 适配器为 `adapters/ego-lora.safetensors`——与今天 MVP / real-trainer 的产物**逐字节一致**。退化为仅含 Kernel 的网格时，Vault 直接产出这种纯 Flat_Package（不写 `grid` 键），因此现有 `ego-output/` 无需迁移即可被加载。
- **默认 Full_In_Load**：`ego run` 默认执行 **Full_In_Load**，加载 Kernel + 每个分区到其最大 Level。各 Cell 适配器在冻结的 Base_Model 之上以 **Coexist_Activate** 加性并存激活（顺序为 Kernel → 各分区 Level 由低到高），且**可逆**（激活某 Cell 后可停用回到先前状态）。
- **按需加载（API 层）**：Nested_In_Load（仅加载分区子集、或各分区只加载到选定深度）在 API 层经 `Load_Selection` 支持（`load_inference_session(edition, selection=...)`）；对应的 `ego run` CLI 选择参数本特性暂未暴露（默认 Full_In_Load），留待后续特性交付。

### MVP 范围说明 / MVP scope

MVP 的路由对**每个被填充分区只产 Level 1**（即 Kernel + level-1-per-zone）。契约层（manifest schema、加载选择、网格往返）完整**支持任意深度的 Level**，更深 Level 的内容级细分类留待后续特性生成。

## 范围

MVP 严格限定为：本地文件系统数据源、Qwen 3.5 0.8B 基座、`normal` edition、QLoRA 训练方法、对话格式语料。所有训练流程在本地设备完成，不向外部网络发送原始数据。

## 开发

```bash
pip install -e "./release/dev[test]"
pytest release/dev/tests
```
