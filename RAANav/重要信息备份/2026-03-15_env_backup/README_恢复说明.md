# 环境信息备份说明（2026-03-15）

本目录用于完整备份当前机器上的关键环境信息，便于后续恢复或对比。

## 1) 备份范围

- 目标环境：`agentrag`、`sem-map`
- 管理器：`conda` / `mamba`
- 系统信息：Linux 内核、Ubuntu 版本、conda/mamba 版本
- 全局配置：`conda` 全局配置与环境列表、`~/.condarc`

## 2) 目录结构

- `conda/`
  - `conda_info_all.txt`
  - `conda_config_show.txt`
  - `conda_config_show_sources.txt`
  - `conda_env_list.txt`
  - `base_conda_list.txt`
  - `condarc.backup`
- `system/`
  - `uname.txt`
  - `os-release.txt`
  - `conda_version.txt`
  - `mamba_version.txt`
- `agentrag/`、`sem-map/`
  - `environment_full.yml`（含 build 信息）
  - `environment_no_builds.yml`（不含 build，更易跨机器）
  - `explicit_spec.txt`（最严格复现）
  - `conda_list.txt`
  - `revisions.txt`
  - `python_version.txt`
  - `pip_freeze.txt`
  - `pip_list_freeze.txt`
  - `pip_check.txt`

## 3) 恢复建议（按优先级）

### A. 跨机器优先（推荐）

```bash
conda env create -n agentrag_restore -f agentrag/environment_no_builds.yml
conda env create -n sem-map_restore -f sem-map/environment_no_builds.yml
```

### B. 强一致复现（同平台/同镜像）

```bash
conda create -n agentrag_exact --file agentrag/explicit_spec.txt
conda create -n sem-map_exact --file sem-map/explicit_spec.txt
```

### C. 已有环境覆盖更新

```bash
conda env update -n agentrag -f agentrag/environment_full.yml --prune
conda env update -n sem-map -f sem-map/environment_full.yml --prune
```

## 4) 已知提示

- `pip_check.txt` 中记录了当前环境的依赖冲突提示（例如 `salesforce-lavis` 与 `timm` 版本约束），这是“现状快照”，不影响备份文件有效性。

## 5) 日常防误删建议

- 每次升级前先执行：

```bash
conda env export -n <env> --no-builds > <env>_$(date +%F).yml
conda list -n <env> --explicit > <env>_$(date +%F)_explicit.txt
```

- 将本目录加入你的云盘/异地备份策略。
