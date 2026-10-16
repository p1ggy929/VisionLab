# 视界智研 —— 基于双目视觉测距与目标检测算法实验平台

> **VisionLab-D455 · v1.0.0**
>
> 一个统一接入 Intel RealSense D455 双目相机、同时承载 **YOLO 多模型目标检测** 与
> **改进 U-Net 道路语义分割** 两条算法路线的软件实验平台。
> 覆盖「采集 — 标注 — 训练 — 推理 — 测距 — 误差分析 — 可视化 — 报告」完整闭环。

---

## 一、30 秒上手

```powershell
cd E:\2Project\2026dachuang\visionlab
.\start.ps1                 # 一键启动（自动使用原项目 A 的虚拟环境）
```

浏览器会自动打开 <http://127.0.0.1:5000/>，默认账号 **admin / admin123**（登录后请立即修改）。

| 场景 | 命令 |
| --- | --- |
| 双击启动（推荐给不熟悉命令行的使用者） | 双击 `start.bat` |
| PowerShell 启动 | `.\start.ps1` |
| **无相机（合成场景演示）** | `.\start.ps1 -NoCamera` |
| 换端口 | `.\start.ps1 -Port 8080` |
| 指定解释器 | `.\start.ps1 -Python "C:\Python312\python.exe"` |
| 只做环境自检、不启动服务 | `.\start.ps1 -Check` |
| 指定数据源 | `.\start.ps1 -Backend mock`（或 `realsense` / `replay`） |

> 无相机时平台自动切换到内置的 **合成 RGB-D 场景（MockBackend）**，
> 检测、分割、测距、误差分析、可视化、报告全部功能仍可完整演示。

### 两个启动脚本的区别与排错

`start.bat` 与 `start.ps1` 功能等价，都是为了适配不同使用习惯而保留的两套入口。
两者的文件编码是**刻意不同**的，修改时请勿随意转换：

| 文件 | 编码 | 换行 | 说明 |
| --- | --- | --- | --- |
| `start.bat` | **纯 ASCII**（0 个非 ASCII 字节） | CRLF | `cmd.exe` 用控制台代码页读取批处理，并按字节偏移反复重读同一文件；一旦写入中文或改成 LF 换行，解析就会错位（典型症状：`'ab-D455' is not recognized`、`'ck' is not recognized`）。因此脚本内**不出现任何中文**，路径用通配符推导而非硬编码。 |
| `start.ps1` | **UTF-8 带 BOM** | LF | Windows PowerShell 5.1 在**没有 BOM** 时按 ANSI（简体中文机器上即 GBK）解码 `.ps1`；直接保存成 UTF-8 无 BOM 会让中文字符串错位，典型症状是「字符串缺少终止符」「缺少右 `}`」「赋值表达式无效」等语法报错。必须用「UTF-8 with BOM」保存。 |

**脚本被禁用时**（提示 `无法加载文件 ... 因为在此系统上禁止运行脚本`）：

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

或直接双击 `start.bat` —— 批处理不受 PowerShell 执行策略限制。

**找不到 Python 解释器时**：两个脚本都会扫描本目录的兄弟目录（`*\.venv\Scripts\python.exe`、`*\venv\...`），
并对每个候选执行一次真实导入探测（`import flask, numpy, cv2, yaml, sqlite3, PIL`），
只有**探测通过**的解释器才会被采用。这一点很关键：同级目录下可能存在其它项目的虚拟环境
（例如 Python 3.14 环境，没有装 Flask），仅凭「文件存在」判断会选错解释器并报
`ModuleNotFoundError: No module named 'flask'`。

> 平台默认复用原项目 A 的虚拟环境 `Yolo_dis_v3源代码\.venv`（Python 3.12.1），
> 其中已装齐全部依赖；如果你希望使用自己的环境，先 `pip install -r requirements.txt`，
> 再用 `-Python` 显式指定。

**正常停止 vs 启动失败**：按 Ctrl+C 停止服务时 Python 也会返回非 0 退出码，因此脚本不靠退出码判断成败，
而是回查日志里有没有本次启动时刻之后的「平台启动」横幅：

| 情况 | 输出 |
| --- | --- |
| 正常停止（Ctrl+C） | `平台已停止（退出码 …；Ctrl+C 主动停止属正常）。` |
| 确实没起来 | `[错误] 平台未能启动，退出码 …` + **自动运行环境自检**并给出排查建议 |

这样避免每次正常关服务都被误报成「启动失败」。若真失败，自检结果会直接指向原因
（例如存储目录不可写会明确报出 `storage.root` 的取值与修复步骤，而不是丢一段 Python traceback）。

**不想看黑窗口**：启动脚本本身就是控制台程序，窗口保持打开即服务在运行；关闭窗口即停止服务。

---

## 二、技术栈（沿用两个原项目）

| 层 | 选型 | 来源 |
| --- | --- | --- |
| Web 框架 | Flask 3.1 + Flask-SocketIO 5.6 + eventlet | 项目 A |
| 页面 | Jinja2 模板 + 原生 JavaScript + 手写 CSS | 项目 A 风格延续 |
| 图表 | Plotly 6.7（本地 vendor，无需外网） | 项目 A |
| 检测 | Ultralytics 8.4（YOLOv8/v9/v10/v11/v12） | 项目 A |
| 分割 | PyTorch 2.4 + 自研 U-Net 家族（含 PVTv2 骨干） | 项目 B |
| 图像 | OpenCV 4.13 + Pillow（中文文字渲染） | 项目 A / B |
| 相机 | pyrealsense2 2.58 | 项目 A / B |
| 数值 | NumPy / Pandas / SciPy | 项目 A / B |
| 报告 | ReportLab（PDF）+ python-docx 思路 + Markdown | 项目 A |
| 元数据库 | **Python 标准库 sqlite3**（20 张表，零额外依赖） | 平台新增 |
| 运行环境 | 直接复用 `Yolo_dis_v3源代码\.venv`（Python 3.12.1） | 项目 A |

**未引入任何新框架**：没有 FastAPI、没有 SQLAlchemy、没有 Vue、没有 Docker 依赖。

---

## 三、功能模块（15 个 / 18 个页面）

| 模块 | 页面 | 说明 |
| --- | --- | --- || M01 项目管理与实验管理 | `/projects` `/experiments` `/experiments/<id>` | 项目容器、实验状态机、配置快照、克隆归档、复现清单、多实验对比 |
| M02 用户与权限管理 | `/login` `/settings` | 四角色 RBAC（admin/teacher/researcher/viewer）、PBKDF2 口令、操作审计 |
| M03 双目相机接入与标定 | `/camera/calibration` `/camera/capture` | 设备枚举、流配置、内参读取、深度平面校验、标定版本管理；**三种数据源**：RealSense 实时 / 合成场景 / **离线回放** |
| M04 RGB-D 数据采集 | `/camera/capture` | 双窗预览、抓拍、**16 位深度 PNG 落盘**、双路录制、标准距离点真值录入、一键生成回放样本 |
| M05 数据集管理与标注管理 | `/datasets` `/annotations` | 版本管理、五种格式导入、完整性校验、前景值口径检查、划分冻结 |
| M06 图像预处理与数据增强 |（配置驱动） | 预处理链与增强策略（albumentations 兼容），作为实验变量 |
| M07 模型仓库与模型选择 | `/models` | 模型卡片、权重哈希校验、引用保护、历史产物批量导入、ONNX 导出 |
| M08 YOLO 多模型目标检测实验 | `/experiments/detection` | 20 个配置、类别过滤、大/小目标分组、dual 权重融合去重、实时结果 |
| M09 改进 U-Net 道路语义分割实验 | `/experiments/segmentation` | 四个结构统一工厂、阈值与形态学、多模型对比、批量评测 |
| M10 目标边界定位与边界距离测量 | `/experiments/ranging` | 检测框中心局部中位数 + 边界多点采样 → 最近/最远/平均 |
| M11 道路边界距离测量 | `/experiments/ranging` | 掩膜轮廓采样 + **左右分离** + **针孔反投影** → 走廊宽度 |
| M12 双目深度误差分析与校正 | `/analysis` | 六维归因、异常样本、线性/比率/分段校正、参数回灌 |
| M13 可视化大屏与图表分析 | `/bigscreen` `/analysis` | KPI、排行榜、距离—误差曲线、热力图、样本画廊、模型对比 |
| M14 实验报告导出 | `/reports` | 五类模板，PDF / Markdown / CSV / JSON，分享链接 |
| M15 系统日志与运行监控 | `/monitor` `/settings` | 日志中心、GPU/磁盘/队列/相机健康、一键自检 |

---

## 四、实测数据（已迁移并验证）

平台已完成两个原项目历史数据的迁移与回归验证：

| 指标 | 平台实测 | 原项目报告 | 结论 |
| --- | --- | --- | --- |
| 历史测距记录 | 1,462 条 | 1,460 条（分析口径） | ✅ |
| 缺检率 | **17.44%** | 17.40% | ✅ 复现 |
| 平均绝对误差 | **0.4454 m** | 0.446 m | ✅ 复现 |
| 中位误差 | **0.190 m** | 0.190 m | ✅ 复现 |
| 最大误差 | **6.57 m** | 6.57 m | ✅ 复现 |
| 距离分组 MAE（1/2/3/4/5/10/15/20 m） | 0.2155 / 0.1501 / 0.2187 / 0.3033 / 0.2628 / 0.5497 / 1.2256 / 4.2624 | 0.216 / 0.150 / 0.219 / 0.303 / 0.263 / 0.550 / 1.226 / 4.262 | ✅ **八个距离点逐点一致** |
| 车辆类别 MAE | 1.555 m | 1.555 m | ✅ 复现 |
| 分割模型（4 个结构） | 全部成功加载项目 B 的 `*.pth` 并推理 | — | ✅ |
| YOLO 检测模型 | **20 个配置全部在真实道路图上推理成功**，11 类英文名全部映射到平台中文类别 | — | ✅ |
| 模型仓库 | **120 个模型**（40 检测 = 20 预训练 + 20 自训练；80 分割，10.9 GB） | 40 + 80 | ✅ |
| 数据集 | **5,814 张图像**（检测 2,694：train 2,301 + val 393；分割 3,120：train 2,496 + val 312 + test 312），**原始划分保留** | 5,814 张 | ✅ |
| 标注 | 2,693 份 LabelMe 标注导入并可校验 | — | ✅ |

**误差校正实测**（在 1,207 条带真值的历史记录上拟合）：

| 校正方法 | 校正前 RMSE | 校正后 RMSE | 下降 |
| --- | ---: | ---: | ---: |
| 线性 | 0.9925 m | 0.6654 m | 33.0% |
| 比率 | 0.9925 m | 0.5460 m | 45.0% |
| **分段（近/中/远）** | 0.9925 m | **0.4579 m** | **53.9%** |
| 分段 · 车辆类别单拟合 | 2.5042 m | **0.1152 m** | **95.4%** |

**统一测距内核验证**（合成场景，真值已知）：

| 量 | 平台实测 | 仿真真值 | 误差 |
| --- | ---: | ---: | ---: |
| 可通行走廊宽度 | 6.204 m | 6.200 m | 0.06% |
| 目标边界测距 MAE | 0.261 m | — | 与合成深度偏置模型一致 |

---

## 五、目录结构

```text
visionlab/
├── app.py                     启动入口（应用工厂 + 首次初始化 + 合成场景兜底）
├── start.ps1 / start.bat      一键启动脚本
├── requirements.txt           依赖说明（默认复用项目 A 的 .venv）
├── configs/                   全部可调参数（YAML，零硬编码）
│   ├── default.yaml           服务/相机/测距/检测/分割/训练/分析/报告/日志
│   ├── classes.yaml           检测 11 类（含中英别名与大小目标分组）+ 分割 2 类与前景值白名单
│   └── models.yaml            YOLO 五个系列 + U-Net 四个结构
├── core/                      公共核心层
│   ├── config.py              配置加载（YAML + VISIONLAB_* 环境变量覆盖）
│   ├── db.py / schema.py      sqlite3 访问层与 20 张表 DDL
│   ├── storage.py             相对 URI、内容哈希、磁盘统计
│   ├── geometry.py            针孔模型、反投影、内参
│   ├── media.py               编解码、深度伪彩、叠加渲染（PIL 中文文字）
│   ├── logging_setup.py       双通道滚动日志 + 内存环形缓冲
│   ├── exceptions.py          统一异常体系
│   └── camera/                base / service / realsense_backend / mock_backend / replay_backend
├── algorithms/                算法插件层（可插拔）
│   ├── base.py                AlgorithmPlugin 协议 + Prediction 统一输出
│   ├── registry.py            插件注册中心
│   ├── ground.py              地面平面估计（两个兜底算法共用）
│   ├── detection/             yolo_adapter / depth_adapter / sim_adapter
│   ├── segmentation/          unet_adapter / classical_adapter / models/ / backbones/
│   ├── ranging/               base / depth_sampler / box_boundary / mask_boundary
│   └── calibration/           strategies（linear / ratio / piecewise）
├── services/                  业务服务层（9 个）
│   auth / project / dataset / model_registry / pipeline / capture /
│   analysis / report / health
├── api/                       接口层（12 个模块）
│   common / auth / projects / camera / datasets / models / inference /
│   analysis / reports / system / media / ws / pages
├── templates/                 23 个 Jinja2 模板（18 个页面 + base/login/error）
├── static/                    css / js（api.js, ui.js, pages.js）/ vendor/plotly
├── storage/                   运行期数据（数据集索引、采集、模型、结果、报告、日志、导出）
└── tools/
    ├── import_legacy.py       历史数据一次性迁移（测距 CSV / 权重 / 数据集）
    ├── rebuild_datasets.py    按原始 train/val/test 划分重建数据集索引
    ├── scan_models.py         登记 40 个检测权重 + 80 个分割权重
    ├── fit_calibration.py     在真实历史数据上拟合并启用误差校正
    ├── e2e_check.py           端到端 HTTP 验收（15 组、80+ 项断言）
    ├── check_yolo.py          20 个 YOLO 配置的真实权重加载与推理验证
    ├── check_seg_models.py    四个分割结构的权重加载验证
    ├── check_seg_compare.py   真实图像上四模型分割对比（输出对比图）
    ├── check_algorithms.py    算法层离线全链路验证
    ├── check_replay.py        离线回放全链路验证
    ├── check_api.py           API 离线冒烟（无需启动服务）
    ├── tune_seg_threshold.py  分割阈值标定
    ├── self_check.py          命令行环境自检
    ├── status.py              数据状态与回归口径速览
    ├── tidy_state.py          收敛校正方案到交付状态
    ├── cleanup_demo.py        清理演示实验
    └── export_soft_copyright.py  导出软著源代码文档（前 30 页 + 后 30 页）
```

---

## 六、核心设计说明

### 6.1 统一测距内核（二合一的技术枢纽）

平台把「目标边界测距」与「道路边界测距」识别为**同一算法本质的两种实例**：
在指定图像区域内采样深度、剔除无效值、换算为物理距离。差异只在采样点来自检测框还是分割掩膜。

```
DepthSampler（共用）         BoxBoundaryStrategy  → 目标边界 min/max/mean
  局部中位数核 21×21    ┌──→ MaskBoundaryStrategy → 道路左/右边界 min/max + 走廊宽度
  有效点 >0 且 ≤20 m   ─┤
  量程裁剪 / 校正回灌   └──→ RangingResult（统一契约，统一落库）
```

`RangingResult` 同时给出：沿光轴深度 `depth_m`、横向偏移 `lateral_m = (u−ppx)·Z/fx`、
欧氏距离 `euclidean_m`、校正前后值、有效点数与采样点坐标。

**走廊宽度**在同一参考深度（默认 5 m）上取左右横向差——若直接相减不同深度的两个边界点，
会因近处受视场限制、远处更宽而严重偏小。

### 6.2 可插拔算法插件

新增一个算法只需实现 6 个方法（`describe / load / predict / predict_batch / export / release`）
并注册一份配置，**不需要改动任何业务代码**。平台内置 7 个插件：

| 任务 | 插件 | 说明 |
| --- | --- | --- |
| detection | `yolo` | Ultralytics，20 个配置，三种权重模式，dual 融合去重（IoU 0.55） |
| detection | `depth_obstacle` | 深度基线检测器（无需权重，几何方法，物理尺寸分类） |
| detection | `sim_ground_truth` | 仿真真值检测器（配合 MockBackend，验证测距与误差链路） |
| segmentation | `unet` / `resunet` / `att_resunet` / `unetv2` | 项目 B 的四个结构，权重键名完全兼容 |
| segmentation | `classical_road` | 传统道路分割器（地面平面 + 颜色，无需权重） |

### 6.3 实验可复现

每次实验生成 `run_manifest.json`，固化：代码提交号与工作区状态、依赖版本、CUDA 与 GPU 型号、
相机型号/序列号/固件/USB 速率、标定文件哈希、数据集版本与内容哈希、划分随机种子、
模型权重哈希与全部超参数。提供「验证复现」（逐项比对输出差异清单）与「重放」两种能力。

### 6.4 三种数据源共用一条流水线

| 后端 | 数据来源 | 用途 | 是否需硬件 |
| --- | --- | --- | --- |
| `realsense` | Intel RealSense D455 实时流 | 真实采集与实验 | 需要相机 |
| `mock` | 内置合成 RGB-D 场景（含随距离放大的深度偏置与深度缺失） | 无设备时开发、教学、答辩演示 | 不需要 |
| `replay` | **从落盘的真实 RGB-D 序列按帧重放** | 复现验证、教学分发、无相机时用真实数据做实验 | 不需要 |

三者在 `CameraBackend` 接口下完全等价：切换数据源不需要改动任何业务代码，
检测、分割、测距、误差分析与报告全部照常工作。回放帧还会携带采集时录入的真值，
因此回放数据同样能做误差统计与校正。

```powershell
# 用当前相机抓一个可回放的样本（含真值），存到 storage/captures/
curl -X POST http://127.0.0.1:5000/api/v1/camera/replay-sample -H "Content-Type: application/json" -d "{\"frames\":40}"
# 切换到回放后端
curl -X POST http://127.0.0.1:5000/api/v1/camera/stream/start -H "Content-Type: application/json" \
     -d "{\"backend\":\"replay\",\"replay_type\":\"capture\",\"replay_source\":\"<会话ID>\",\"fps\":20}"
```

界面入口：**数据采集页 → 数据源**（下拉选择来源 + 回放源 + 一键抓取回放样本）。

### 6.5 演示稳健性

- **MockBackend 合成场景**：无相机也能完整演示（USB 松动、驱动异常、设备被占用时的兜底）
- **自动降级**：未装 `ultralytics` → 深度基线检测器；未装 `torch` → 传统道路分割器；
  未装 `reportlab` → 报告降级为 Markdown；未装 `flask-socketio` → 实时推流降级为 HTTP 轮询
- **一键自检**：`/settings` 或 `python tools/self_check.py`，10 秒内确认环境
- **大屏适配**：1920×1080 与 2560×1440 均可全屏铺满

---

## 七、常用操作

### 迁移历史数据（一次性）

```powershell
# 方式一：一条命令完成全部迁移（推荐）
python tools\import_legacy.py `
  --project "校园道路视觉感知" `
  --measurements "E:\2Project\2026dachuang\Yolo_dis_v3源代码\data\results\measurements.csv" `
  --det-images   "E:\2Project\2026dachuang\Yolo_dis_v3源代码\datasets\images" `
  --det-labels   "E:\2Project\2026dachuang\Yolo_dis_v3源代码\datasets\labels" `
  --seg-root     "E:\2Project\2026dachuang\Unet" `
  --seg-weights  "E:\2Project\2026dachuang\Unet\models" `
  --best-model   "E:\2Project\2026dachuang\Unet\best_model" `
  --runs         "E:\2Project\2026dachuang\Yolo_dis_v3源代码\runs"

# 方式二：分步执行（便于观察每步结果）
python tools\rebuild_datasets.py --apply   # 5,814 张图像，保留原始划分
python tools\scan_models.py                # 40 检测权重 + 80 分割权重
python tools\fit_calibration.py            # 在 1,207 条真实记录上拟合校正
```

> `--det-images` 指向含 `train/`、`val/` 的目录时会自动按原始划分整体导入；
> `--seg-root` 指向含 `dataset/train|val|test` 的数据根目录，同样保留原始划分。
> 也可以在 Web 端「数据集管理 → 导入历史数据集」用表单完成同样的导入。

### 跑一次完整实验

1. `/experiments/ranging` → 顶栏确认算法路线为「双策略同跑」
2. 参数面板选择检测器（有 YOLO 权重则选 `yolo`，否则选 `仿真真值检测器` 或 `深度基线检测器`）
3. 点「开始实验」→ 观察左侧叠加画面与右侧深度伪彩
4. 走到标准距离点时点「暂停/恢复」，在记录列表点「记录」并填入卷尺实测值
5. 点「结束实验」→ 到 `/analysis` 看统计、归因与校正
6. 到 `/reports` 生成报告，到 `/bigscreen` 投屏演示

### 环境自检与验收

```powershell
python tools/self_check.py                       # 环境自检
python tools\e2e_check.py --base http://127.0.0.1:5000   # 端到端验收（服务需在运行）
python tools\export_soft_copyright.py            # 导出软著源代码文档
```

---

## 八、已知边界与后续迭代

| 项 | 现状 | 说明 |
| --- | --- | --- |
| UNetV2 (PVTv2) | 可用 | 需 `timm==0.9.16`（已安装）。缺失时该结构在界面标注为不可用，不影响其余三个模型 |
| GPU 训练 | 交由任务队列串行调度 | 平台负责调度、进度与产物归档；训练实现复用原项目脚本。训练执行器为骨架实现，需接入具体训练入口 |
| 在线标注 | 未实现 | 沿用 LabelMe / LabelImg 外部标注后导入；平台负责导入、校验、版本管理与浏览 |
| 边缘部署 | 只做 ONNX 导出 | TensorRT / Jetson 实测未做 |
| 夜间 / 雨天场景 | 不在范围 | 平台聚焦日间场景；`illumination` 字段已为扩展预留 |
| 官方预训练权重 | 已登记为对照基线 | 20 个 COCO 80 类权重，未针对本项目 11 类微调；用于与自训练权重做对照 |
| 分割模型默认阈值 | 0.30 | 实测各结构最优阈值不同（UNet 0.05 / ResUNet 0.40 / AttResUNet 0.05 / UNetV2 0.30），界面滑块可调 |
| Docker Compose | 未做 | 当前交付一键脚本 `start.ps1` / `start.bat`，更贴合答辩现场；容器化不影响任何业务代码 |

---

## 九、验证记录

| 验证项 | 结果 |
| --- | --- |
| 算法层冒烟测试 | 相机 → 检测 → 分割 → 双策略测距 → 三类校正 全链路通过 |
| API 冒烟测试 | 12 组接口 + 18 个页面路由全部 200 |
| 真实 HTTP 验收 | 登录、17 个页面、静态资源、实时 JPEG 帧流全部正常 |
| 端到端验收（`tools/e2e_check.py`） | **15 组、80+ 项断言全部通过** |
| 历史数据回归 | 8 个距离点 MAE 与原项目报告逐点一致 |
| 四分割模型权重加载 | UNet / ResUNet / AttResUNet / UNetV2 全部成功 |
| 真实图像分割对比 | UNet IoU 0.937 / ResUNet 0.963 / AttResUNet 0.963 / UNetV2 0.887（6 张测试图） |
| **20 个 YOLO 配置真实推理** | 全部加载自训练权重并在真实道路图上检出目标（`tools/check_yolo.py --all`） |
| **离线回放全链路** | 抓取 20 帧真实 RGB-D → 回放 → 检测/分割/测距全部跑通（`tools/check_replay.py`） |
| 数据资产 | 5,814 张图像、2,693 份标注、1,462 条测距记录、120 个模型权重 |
| 环境自检 | 18 项检查，0 错误、2 警告（均为硬件缺失提示，已自动降级） |

### 已修复的三个原实现缺陷

| # | 缺陷 | 原实现 | 平台处理 |
| --- | --- | --- | --- |
| 1 | **v11/v12 结构名拼错** | 权重解析用 `"yolo" + size`，得到 `yolon` 而非 `yolo11n`，导致 20 个配置里 8 个找不到权重 | 修正为 `yolo{version}{size}`；并兼容原项目两种命名惯例（官方 `yolo11n.pt` 与自训练 `yolov11n_cloud.pt`），20/20 全部命中 |
| 2 | **分割阈值语义不一致** | UNet/ResUNet/AttResUNet 输出概率，UNetV2 输出 logits，原脚本却统一用 `> 0.30` | 自动识别：输出超出 [0,1] 视为 logits 并套用 sigmoid，界面显示实际生效的激活方式 |
| 3 | **掩膜前景值口径冲突** | 训练用 `mask == 17`，部分评估用 `mask > 0` | 校验层强制统计前景值分布，多值混用时告警并提供「一键统一前景值」 |
| 4 | **GUI 切换模型不生效** | PyQt 下拉框存在但 `CameraThread` 固定构造 UNetV2 | 统一 `MODEL_FACTORY` 按名构造，服务端已验证四个结构分别被调用 |
