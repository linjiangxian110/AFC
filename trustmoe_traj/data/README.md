# TrustMoE-Traj 数据层说明

这个目录可以把它理解为：**项目里专门负责“读数据、整理数据、打包数据”的地方**。

当前我们已经做的事情是：

1. 先规定了一套统一的数据格式；
2. 再把 ETH 五个子集的原始 txt 数据，按这套格式切成可直接使用的样本；
3. 提供 batch 拼接能力，方便后面直接接训练或推理；
4. 提供主缓存 pickle 的生成与读取接口，支持“一次预处理，多次复用”。

---

## 当前目录结构

```text
trustmoe_traj/data/
├─ __init__.py
├─ schema.py
├─ collate.py
├─ README.md
├─ adapters/
│  ├─ __init__.py
│  └─ eth.py
└─ ETH/
   └─ README.md
```

---

## 每个 Python 文件是做什么的

### 1. `schema.py`

这个文件可以理解为：**先规定“标准盒子长什么样”**。

通俗一点说，后面不管是哪种数据、哪个模型，都不能乱传数据，必须按这里定义好的字段来。

当前它主要定义了：

- `past_traj`：历史轨迹
- `future_traj`：未来轨迹
- `agent_mask`：哪些 agent 是真实的，哪些是补零补出来的
- `scene_meta`：这个样本来自哪个数据集、哪个子集、哪个文件、哪个时间段

你可以把它理解成：**数据层的统一语言和统一包装标准**。

---

### 2. `collate.py`

这个文件可以理解为：**把很多大小不一的小样本，整理成一个能一起送进模型的大 batch**。

为什么需要它？

因为不同场景里 agent 数量不一样：

- 有的样本可能只有 1 个 agent；
- 有的样本可能有几十个 agent。

模型训练时，通常希望一个 batch 里的数据形状统一，所以这个文件负责：

- 自动 padding
- 生成统一形状的 batch
- 保留 `agent_mask`

你可以把它理解成：**训练前的数据打包工人**。

---

### 3. `__init__.py`

这个文件本身不负责复杂逻辑，它更像是：**数据层的统一出口**。

作用是把常用的数据类、配置、函数统一导出，这样外部模块在使用时不需要记很多深层路径。

比如以后外面代码可以直接从：

```python
from trustmoe_traj.data import ETHTrajectoryDataset
```

而不一定要写很长的内部路径。

---

## `adapters/` 目录是做什么的

这个目录可以理解为：**不同数据集的“翻译器”目录**。

因为不同数据集原始格式不同，所以要给每种数据集写一个 adapter，把它们翻译成我们统一的数据格式。

### 4. `adapters/eth.py`

这是当前最重要的实现文件。

它负责把 ETH 原始 txt 文件：

- 扫描出来；
- 读进来；
- 按 `obs=8 / pred=12 / skip=1` 切成训练样本；
- 只保留在完整窗口内持续出现的 agent；
- 组织成统一的多 agent 样本格式。

你可以把它理解成：**ETH 数据的专属翻译器 + 切片机**。

目前它已经支持 ETH 五个子集：

- `eth`
- `hotel`
- `univ`
- `zara1`
- `zara2`

并且可以分别处理：

- `train`
- `val`
- `test`

---

### 5. `adapters/__init__.py`

这个文件和数据层根目录下的 `__init__.py` 类似，作用是：**把 adapter 相关的类和函数统一导出**。

这样以后外部模块要调用 ETH adapter 时会更方便。

---

## `ETH/` 目录是做什么的

这个目录用于存放 ETH 原始数据。

这里面放的是：

- 五个子集目录
- 每个子集下的 `train / val / test`
- 原始 txt 文件

可以理解为：**原材料仓库**。

而 `adapters/eth.py` 的工作，就是去这个仓库里取原材料，再加工成模型能吃的数据。

---

## 当前已经做到哪一步了？

目前已经完成的是：

1. 定义了统一数据标准；
2. 实现了 ETH 五个子集的原始 txt 读取与样本切分；
3. 实现了 batch 拼接；
4. 已经可以把原始 txt 数据转成 **内存中的标准样本 / batch**；
5. 已经支持把标准样本落盘为主缓存 pickle 文件。

### 当前主缓存是什么

当前默认主缓存路径为：

```text
trustmoe_traj/data/ETH/processed/{subset}_{split}.pkl
```

缓存中会保存：

- 文件级 `cache_meta`
- 文件级 `cache_stats`
- 样本级 `samples`

其中每个样本至少包含：

- `past_traj`
- `future_traj`
- `agent_mask`
- `scene_meta`
- `extras(agent_ids, frame_ids, num_agents)`

### 当前默认行为说明

也就是说，现在已经能：

- 读取原始数据
- 转成统一样本
- 拼成 batch
- 保存为主缓存 pickle
- 从主缓存重新加载

但 **不会在每次初始化数据集时自动偷偷重建缓存**。

推荐做法是：

1. 先显式运行一次预处理脚本；
2. 后续训练 / 推理时通过 `prefer_cache=True` 优先读取 pickle。

对应脚本建议使用：

```bash
python -m trustmoe_traj.scripts.prepare_eth_cache --subset all --split all
```

对应数据接口示例：

```python
from trustmoe_traj.data import ETHAdapterConfig, ETHTrajectoryDataset

config = ETHAdapterConfig(subset="eth", split="train", prefer_cache=True)
dataset = ETHTrajectoryDataset(config)
```

---

## 下一步最自然要做什么？

当前数据层已经像是：**把原材料洗干净、切整齐、装箱好了**。

下一步最自然要做的是：**把这些标准数据真正接到模型里**。

建议顺序：

1. 把当前统一数据格式映射到 MoFlow teacher / student 的输入格式；
2. 先跑通 Slow（teacher）基线；
3. 再跑通 Fast（student）基线；
4. 做第一版 baseline 对比；
5. 再继续做 evaluator、Experts、Router。

换句话说：

**数据层地基已经打好，下一步要开始搭第一层楼，也就是 baseline。**

---

## 新增脚本说明

### `trustmoe_traj/scripts/prepare_eth_cache.py`

这个脚本可以理解为：**把 ETH 原始数据统一预处理成主缓存 pickle 的一键工具**。

它的作用是：

- 扫描 ETH 五个子集的原始 txt；
- 按统一 schema 切样本；
- 生成 `processed/*.pkl`；
- 让后续训练和实验直接复用缓存，而不是每次重新处理原始文本。
