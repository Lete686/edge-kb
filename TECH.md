# TECH · Edge-KB 技术说明（一页）

## 1. 模型与量化

**模型**：`Xenova/all-MiniLM-L6-v2`（HuggingFace `lastModified` 2025-07-22，Apache-2.0），6 层 BERT 蒸馏，输出 **384 维**句向量。浏览器侧调用：

```js
const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { device, dtype });
const out = await extractor(texts, { pooling: 'mean', normalize: true });
```
mean pooling 按 attention mask 屏蔽 padding；`normalize` 后余弦距离与内积等价，因此集合声明 `distance: 'cosine'`。

**体积与压缩比**（实测，含 `tokenizer.json` + `config.json`，按 MiB = 1024² 计）：

| 配置 | 体积（MiB） | 压缩比 | 说明 |
| --- | ---: | ---: | --- |
| fp32（基线） | 86.39 | 1.00× | `onnx/model.onnx` |
| fp16 | 43.39 | 1.99× | `onnx/model_fp16.onnx`，WebGPU 原生算子 |
| q8（INT8） | 22.10 | 3.91× | `onnx/model_quantized.onnx`，仅 WASM 路径使用 |

只比权重文件：`model.onnx` 90,387,606 B vs `model_quantized.onnx` 22,972,370 B = **3.93×**（两个值取自 HF 仓库文件树）。压缩比 = 同口径总体积之比（fp32 ÷ 该配置），共用的 tokenizer/config 使「含公共文件」的比值略低于只比权重。

> 体积来自面板对 HF 文件的字节统计（HEAD 的 `content-length` / LFS `x-linked-size`，取不到时用 `Range` 读总长），MiB = 1024²，原始记录写作 MB。注意：按 HF 文件树逐字节相加（`tokenizer.json` 711,661 B + `config.json` 650 B）会比表中高约 0.5 MiB，差异原因未核实；本表按实测值填写，不改数。

---

## 2. 端侧运行与降级（含 INT8-on-WebGPU 的失败）

| 环节 | 位置 | 说明 |
| --- | --- | --- |
| 模型拉取 | 浏览器 | jsDelivr 上的 Transformers.js **3.7.5**（Apache-2.0）+ HF 权重；失败自动换 `hf-mirror.com` |
| 文档 / 查询向量化 | 浏览器 | batch 默认 8（可调 1–64），逐批回调进度；单条前向单独计时 |
| 后端 | Node/Express | 只校验 384 维与数值有限性，**无任何 embedding 代码** |

**降级链**：`detectWebGPU()`（`navigator.gpu` + `requestAdapter()` 非 null）→ 依次尝试 `webgpu+fp16 → webgpu+fp32 → wasm+q8 → wasm+fp32` → 每个候选都要通过一次探针推理（输出必须真是 384 维）才采纳 → 徽标实时显示生效的 `device/dtype`（绿=GPU，橙=CPU），失败原因逐条进日志。

### 质量实测（24 篇语料 + 12 条查询，qrels 见 `samples/queries.jsonl`）

| 配置 | 设备 | Recall@5 | Recall@10 | MRR@10 |
| --- | --- | ---: | ---: | ---: |
| fp16 | webgpu | 95.8% | 100% | 0.958 |
| q8 | wasm | 95.8% | 100% | 0.958 |
| fp32 | webgpu | 95.8% | 100% | 0.958 |
| q8 | **webgpu ⚠** | **33.3%** | **54.2%** | **0.176** |

**关键发现（必须如实记录）**：**INT8 在 WebGPU 后端上质量崩溃**——Recall@5 从 95.8% 掉到 33.3%，MRR@10 从 0.958 掉到 0.176；同一份 q8 权重换到 **WASM** 后端立刻恢复到 95.8%。因此结论是：`onnxruntime-web` 的 **WebGPU EP 对 INT8 算子支持不完整**，**不是量化损失**（q8 权重本身没问题，在 WASM 上与本基准的 fp16/fp32 完全持平）。
据此**主用方案定为 FP16 + WebGPU**（体积约为 fp32 的一半，本基准上与 fp32 持平），INT8 只在 WASM 回退路径下使用；`buildCandidates()` 也据此把 `webgpu+q8` 从链路中去掉。

> 指标口径：`compare.js` 在同一份文档向量集合上做余弦排序算 Recall/MRR，是"量化/后端差异"的对照，与线上 seekdb 的 RRF 名次不同，不可当线上检索质量。

---

## 3. seekdb 使用

**连接**：server 模式 `127.0.0.1:2881`（docker `oceanbase/seekdb`），库 `test`；嵌入式模式仅 Linux/macOS。

**集合 schema**（显式构造，不能简写 `configuration`，否则回落到默认 embedding function）：

```js
new Schema({
  fulltextIndex: new FulltextIndexConfig('ik', { ik_mode: 'smart' }),
  vectorIndex: new VectorIndexConfig({
    hnsw: { dimension: 384, distance: 'cosine' },
    embeddingFunction: null,      // 向量只可能来自浏览器
  }),
})
```

**检索**（融合在库内完成，客户端不合并两路结果）：

```js
collection.hybridSearch({
  query: { whereDocument: { $contains: keywordQuery } },   // 全文分支：去停用词的关键词串
  knn:   { queryEmbeddings: [qvec] },                      // 向量分支：原始问题
  rank:  { rrf: {} },
  nResults, include: ['documents', 'metadatas', 'distances'],
})
```

两路各取 `candidateWindow = clamp(nResults×3, nResults, 100)`；全文分支若直接塞整句会被当成近似短语而几乎零命中，RRF 就退化成只有向量一路，所以先 `buildKeywordQuery()` 去停用词、取前 8 个词。

**增量与刷新**：文档 id = 内容哈希（FNV-1a + 长度）→ `doc-<hex>-<lenhex>`；服务端用 `collection.upsert`，同一文本重复导入不新增（实测二次导入仍 24 条）。数据落在 seekdb，刷新页面只重建 UI，集合列表与条数从 `GET /api/collections` 重新拉。重置 = 删除并重建集合。

---

## 4. 状态与优化

### 耗时（Windows 11 + Chrome；来宾 seekdb 在 4 vCPU / 8 GiB VM）

| 阶段 | WebGPU（fp16） | WASM（q8） |
| --- | --- | --- |
| 模型加载（首次，含下载） | 10.36 s | 37.17 s |
| 24 篇文档向量化 | 2.50–3.05 s（50 ms/篇） | 1.51 s（63 ms/篇） |
| 写入 seekdb（24 条） | 203–267 ms | 267 ms |
| 查询向量化 | 53–123 ms | 16 ms |
| seekdb 混合检索 | 35–50 ms（服务端） | 183 ms |
| 端到端（提交→结果） | 约 150–190 ms | 约 200 ms |

> 首轮数字含预热（WebGPU 首次要编译着色器）；本表每格只跑 1–2 次，**属初步数据**，正式引用请用重复测量（≥3 次、丢弃首轮）的中位数。

### 其他优化点

| 关注点 | 做法 |
| --- | --- |
| 批处理 | 向量化 batch 8、写库 batch 32，均可在 UI 调整并按 HTTP 分批提交 |
| 冷启动 | 权重进浏览器 Cache API；仅在非安全上下文才失效 |
| 超时与稳定性 | 模型加载 10 min、向量化 3 min、API 60 s；后端 `Promise.race` 包 seekdb 操作 |
| 内存 | 向量写入后不驻留页面，`state` 只留 pipeline 与集合名 |

---

## 5. 局限与未做项

1. **数据为初步测量**：耗时每格 1–2 次、单机型（Windows 11 + Chrome + 虚拟机里的 seekdb），没有跨机型/跨浏览器矩阵。
2. **质量基准很小**：24 篇语料、12 条查询、qrels 人手标注，只能横向比较配置差异，不能当公开榜单分数。
3. **图片/多模态扩展未做**：只接受文本与 `.txt`/`.md`，没有 image embedding、OCR 或图文混检。
4. **MiniLM 以英文为主**：中文语义召回偏弱，语料也是英文；中文场景应换多语言模型并同步改维度（`EMBEDDING_DIMENSION` 与 HNSW `dimension`，属破坏性变更）。
5. **全文分支是 `$contains`**：依赖关键词串的包含匹配，长查询与同义改写能力有限，召回主要由向量分支 + RRF 提供。
6. **单机单集合、无鉴权**：仅赛题演示，生产需补认证、限流、分页与集合隔离。
7. **`webgpu+q8` 未被采纳**：受 WebGPU EP 的 INT8 算子限制，已在降级链中排除；若上游修复应重新实测再改默认。
