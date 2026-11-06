# Edge-KB · 浏览器本地向量化 + seekdb 库内混合检索

迷你知识库：**文本在浏览器里用 Transformers.js 本地向量化 → 写入 seekdb → 查询时同样在浏览器算查询向量 → 由 seekdb 的 `hybridSearch` 在库内做「向量 + 全文」RRF 融合 → 展示排序结果与原文**。后端**不计算任何向量**，只做校验与转发；无打包器，前端是原生 ESM。

> 本文数字均来自真机实测（Windows 11 + Chrome，seekdb 跑在 4 vCPU / 8 GiB 虚拟机），完整口径见 [TECH.md](./TECH.md)。未实测处均已标注。

---

## 1. 架构

```
┌─────────────────────────────────────────────────────────────────────┐
│ 浏览器（全部推理都在这里）                                          │
│ app.js：导入 UI（粘贴 / .txt / .md）→ 分批向量化 → 写库             │
│ compare.js（#btn-compare）：fp16@webgpu · q8@wasm · fp32@webgpu     │
│ Transformers.js 3.7.5 + Xenova/all-MiniLM-L6-v2（权重进浏览器缓存） │
│ device: webgpu → wasm      dtype: fp16 → q8 → fp32                  │
│ mean pooling + L2 normalize → 384 维 float32                        │
└─────────────────────────────────────────────────────────────────────┘
        │ HTTP JSON：只传向量与关键词，不传模型
        ▼
┌──────────────────────────────────────────────────────────────────┐
│ Node + Express（server/，不含任何 embedding 代码）               │
│ index.mjs：路由 · 384 维与有限性校验 · 集合名白名单 · 超时       │
│ seekdb.mjs：连接与集合生命周期（embedded | server）· 显式 Schema │
└──────────────────────────────────────────────────────────────────┘
        │ MySQL 协议（server 模式）
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│ seekdb（server 模式 127.0.0.1:2881，库 test）                         │
│ collection.upsert(ids, documents, embeddings, metadatas)              │
│ collection.hybridSearch({ whereDocument.$contains, knn, rank:{rrf} }) │
│ 全文分支：ik 全文索引（ik_mode=smart），输入去停用词的关键词串        │
│ 向量分支：HNSW / cosine / 384 维，输入原始问题向量                    │
│ RRF 融合在库内完成，客户端不合并两路结果                              │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 2. 快速开始

**前置**：Node.js **20+**（`engines: >=20`；`cds` 不适用，用 NodeSource 或官方 tarball）；一个连得上的 seekdb；Chrome/Edge 113+ 且经 `localhost` 或 `https` 访问。

**① 启动 seekdb（二选一）**

```bash
# 方式一（推荐）：Docker 官方镜像，默认 2881 / root / 空密码
docker run -d --name seekdb -p 2881:2881 oceanbase/seekdb

# 方式二：嵌入式，仅 Linux / macOS，不需要任何服务
SEEKDB_MODE=embedded SEEKDB_PATH=./data/seekdb npm start
```

装包（systemd）部署见 <https://docs.seekdb.ai/seekdb/zh-CN/V1.3.0/deploy-by-systemd/>。

**② 启动应用**

```bash
cd outputs/edge-kb
npm install --omit=dev --legacy-peer-deps   # 两个参数都是必需的，见 §7-1
npm start                                    # http://127.0.0.1:3000
```

`npm run dev` = `node --watch server/index.mjs`。

**③ 用起来**：打开页面 → 点「加载示例语料」（24 篇）或粘贴文本/上传 `.txt` `.md` → 点「向量化并导入」（徽标会显示 `推理后端` 与 `量化`）→ 输入问题点「检索」→ 结果卡片给出 `rank`、文档 id、`sim/dist`、高亮片段和可展开原文。首轮要下载权重 **22–86 MiB**（视精度），之后走浏览器缓存。

环境变量：`SEEKDB_MODE`（默认 server）、`SEEKDB_PATH`、`SEEKDB_HOST`（127.0.0.1）、`SEEKDB_PORT`（2881）、`SEEKDB_USER`（root）、`SEEKDB_PASSWORD`、`SEEKDB_TIMEOUT_MS`（15000）、`PORT`（3000）、`MAX_DOCS_PER_REQUEST`（512）。健康检查：`curl http://127.0.0.1:3000/api/health`。

---

## 3. 浏览器要求与 WebGPU

| 项 | 要求 / 说明 |
| --- | --- |
| 浏览器 | **实测只在 Chrome 上跑过**。Chrome / Edge 113+ 默认有 WebGPU；Firefox 141+（仅 Windows，Mozilla 官方博客 2025-07）也已启用；Safari 与其它组合未逐版本核实，可能直接走 WASM |
| 安全上下文 | `navigator.gpu` 与 Cache API 只在 **https / localhost / 127.0.0.1** 存在；用 `http://<局域网IP>:3000` 打开会**静默降级 WASM** |
| 网络 | 需要 `cdn.jsdelivr.net`（Transformers.js 3.7.5）与 `huggingface.co`（权重，失败自动退 `hf-mirror.com`） |
| 内存 | 建议 ≥ 4 GB 可用（经验值，未实测峰值） |

**降级链**（`app.js` → `ensureModel()`）：`webgpu+fp16 → webgpu+fp32 → wasm+q8 → wasm+fp32`，每个候选加载后都做一次探针推理（输出必须真为 384 维）才采纳，失败原因逐条写入页面日志；徽标绿=GPU、橙=CPU。`webgpu+q8` 不在链上，原因是 INT8 在 WebGPU 上质量会崩（[TECH.md §2](./TECH.md)）。

---

## 4. 功能清单

| 要求 | 实现 | 状态 |
| --- | --- | --- |
| 导入（粘贴/文件）、切分、文档身份 | `app.js` `splitIntoDocs()` / `makeDocId()` | ✅ 空行分段，超 4000 字符硬切；id = FNV-1a + 长度 |
| 浏览器本地向量化 | `ensureModel()` / `embedAll()` | ✅ 384 维，mean pooling + normalize |
| 量化（INT8/FP16）体积、压缩比、质量 | `compare.js` + `#btn-compare` | ✅ 见 §5 |
| WebGPU 优先 + WASM 降级 + 界面提示 | `detectWebGPU()` / `buildCandidates()` / `paintBackendPill()` | ✅ 徽标 + 日志说明降级原因 |
| seekdb 存储（向量 + 全文） | `server/seekdb.mjs` `buildSchema()` | ✅ ik 全文索引 + HNSW 余弦 + metadata |
| 库内混合检索（RRF） | `POST /api/collections/:name/search` | ✅ 融合在库内，客户端不合并 |
| 增量导入 / 索引刷新 | `runImport()` + `collection.upsert` | ✅ 重复导入条数不增（实测仍 24 条） |
| 集合重置 / 切换 | `POST /api/collections/:name/reset`、下拉框 | ✅ 删除重建，检索目标同步 |
| 可复现 | 本文件 + TECH.md + `samples/` | ✅ 24 篇语料、12 条查询含 qrels |

**后端接口**（前端只调这 5 个）：`GET /api/health`、`GET /api/collections`、`POST /api/collections/:name/documents`（走 `upsert`）、`POST /api/collections/:name/search`（走 `hybridSearch`）、`POST /api/collections/:name/reset`（删除重建）。向量全部由请求体传入，后端只校验 384 维与有限性，不传向量直接 400——从接口层面杜绝「后端代算向量」。

---

## 5. 可复现的实验步骤

**数据**：`samples/corpus.jsonl` 24 篇英文短文（`id`/`title`/`text`）；`samples/queries.jsonl` 12 条查询（`id`/`query`/`relevant`，qrels 为手写的小样本基准，只用于同语料下的横向对比，不代表公开榜单分数）。

**复现体积与 Recall（全部在浏览器，不依赖 Node 侧脚本）**

1. `npm start`，用 **localhost** 打开（保证有 WebGPU）；
2. 「加载示例语料」→「向量化并导入」，等徽标变成 `WebGPU` + `FP16`；
3. 点「量化对比评测」（`#btn-compare`）：`compare.js` 依次以 `fp16@webgpu`、`q8@wasm`、`fp32@webgpu` 加载模型，对**同一份语料**各做一遍余弦排序，并按 HF 文件字节数（HEAD 的 `content-length`，取不到时用 `Range` 读 `content-range`）统计各自 `权重 + tokenizer.json + config.json` 的体积；
4. 把面板表格里的数字抄进 [TECH.md](./TECH.md) §1/§2。

> 面板的 Recall/MRR 是浏览器内纯向量余弦排序的结果，与线上「seekdb 向量+全文 RRF」名次不同，不能当线上检索质量。

**复现耗时**：页面「耗时」区就是口径（`模型加载(首次含下载) / 向量化 / 写入 seekdb / 合计`，检索时的 `查询向量化` 与 `seekdb 混合检索（服务端）/（含网络）`）。24 篇语料导入 + 若干次查询重复 3 遍取中位数，丢弃含着色器编译预热的首轮。

**复现「INT8 在 WebGPU 上崩」**：把 `public/compare.js` 里 `q8` 那组的 `device` 改成 `'webgpu'` 重跑面板，可复现 Recall@5 33.3% 的崩溃（成因是 WebGPU EP 的 INT8 算子支持不完整，**不是量化损失**）；验证后改回 `'wasm'`。

---

## 6. 目录结构

```
edge-kb/
├── package.json        # start / dev；前端依赖全部来自 CDN，无需构建
├── server/
│   ├── index.mjs       # 路由、384 维校验、超时、静态托管
│   └── seekdb.mjs      # 连接与集合生命周期、显式 Schema、hybridSearch 参数
├── public/
│   ├── index.html      # 导入 / 检索 / 量化对比 + 日志
│   ├── app.js          # 模型加载与降级、导入、检索、UI
│   ├── compare.js      # 量化对比（体积 + Recall@K + MRR@10）
│   └── style.css
├── samples/
│   ├── corpus.jsonl    # 24 篇语料
│   └── queries.jsonl   # 12 条查询 + qrels
├── README.md
└── TECH.md
```

早期曾有一个 Node 侧复算脚本 `measure.mjs`，现已移除：量化与质量评测统一由浏览器内「量化对比评测」面板完成，避免在 Node 侧再装一份 transformers。

---

## 7. Troubleshooting

**1）`npm install` 报 `Unsupported URL Type "workspace:": workspace:*`**
来自 seekdb 的 optional peer `@seekdb/default-embed`（上游发布瑕疵），本项目用不到（向量在浏览器算）：

```bash
npm install --omit=dev --legacy-peer-deps
```

**2）嵌入式 seekdb 报 `cannot allocate memory in static TLS block`**
某些 Node/glibc 组合加载原生绑定时触发。本项目默认就是 server 模式，不依赖原生绑定：

```bash
SEEKDB_MODE=server SEEKDB_HOST=127.0.0.1 SEEKDB_PORT=2881 npm start
```

PowerShell 写法：`$env:SEEKDB_MODE='server'; $env:SEEKDB_HOST='127.0.0.1'; npm start`

**3）局域网 IP 打开时没有 WebGPU（降级 WASM）**
WebGPU 与 Cache API 只在安全上下文可用。开发期可临时放行：

```bash
chrome --unsafely-treat-insecure-origin-as-secure=http://192.168.1.23:3000 --user-data-dir=C:\edgekb-chrome
```

生产环境请用 https 反代或只监听 localhost。

**4）建集合报 `Embedding function 'default-embed' is not registered`**
说明用了简写 `configuration: { dimension, distance }`，落回了默认 embedding function。必须用显式 `Schema`（`server/seekdb.mjs` 的 `buildSchema()`）且 `embeddingFunction: null`。已建坏的集合先重置：`curl -X POST http://127.0.0.1:3000/api/collections/<集合名>/reset`

**5）Node 版本不够**
`engines` 要求 `>=20`，`cds` 不适用；用 NodeSource 或官方 tarball 装 20/22，`node -v` 确认后再装依赖。

---

## 8. 许可与模型来源

- 本项目代码：**Apache-2.0**（`package.json` 的 `license`）。
- 模型 [`Xenova/all-MiniLM-L6-v2`](https://huggingface.co/Xenova/all-MiniLM-L6-v2)：**Apache-2.0**，384 维，`sentence-transformers/all-MiniLM-L6-v2` 的 ONNX 导出；HF 仓库 `lastModified` = **2025-07-22**（经 HF API 核对，2026-11 查询）。
- 推理运行时：`@huggingface/transformers` **3.7.5**（jsDelivr，**Apache-2.0**，经 jsDelivr 上的 `package.json` 核对）；前端不打包、不落 npm 依赖。
- seekdb：Node SDK `^1.4.0` + `oceanbase/seekdb` 镜像，许可以其官方发布为准。
