# ETH 主缓存目录说明

这个目录用于存放经过 TrustMoE-Traj 统一 schema 预处理后的主缓存文件。

默认文件命名方式：

```text
{subset}_{split}.pkl
```

例如：

- `eth_train.pkl`
- `eth_val.pkl`
- `hotel_test.pkl`

这些文件属于**可重复生成的缓存产物**，因此默认不纳入版本管理。

推荐通过以下命令生成：

```bash
python -m trustmoe_traj.scripts.prepare_eth_cache --subset all --split all
```