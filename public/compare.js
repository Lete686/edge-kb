/**
 * 量化对比评测（q8 / INT8 与 fp32）—— 全部在浏览器内完成，不经过 seekdb。
 *
 * 为什么单独成模块：主流程（public/app.js）只跑一个 dtype，没有对照基线，
 * 而赛题要求报告「量化前后的模型体积、压缩比与检索质量变化」。这里复用同一套
 * Transformers.js 管线、同一批文档与查询，只切换 dtype，把差异归因到量化本身。
 *
 * 质量口径有边界：两种配置都在同一份文档向量集合上做余弦排序，而线上检索是
 * seekdb 的向量 + 全文 RRF，名次必然不同。因此本模块的 Recall / MRR 只用于
 * 「量化影响评估」，不能当作线上检索质量。
 */

const DEFAULT_OPTIONS = {
  modelId: 'Xenova/all-MiniLM-L6-v2',
  /** 与 app.js 的 CFG.modelHosts 同序：官方优先，取不到再走镜像。 */
  modelHosts: ['https://huggingface.co', 'https://hf-mirror.com'],
  transformersUrl: 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5',
  dimension: 384,
  batch: 8,
  /**
   * 三组配置：主推 FP16（WebGPU 原生支持、体积减半、质量几乎无损）；
   * INT8 放在 WASM 后端——onnxruntime-web 的 WebGPU EP 对 INT8 算子支持不完整，
   * 实测 Recall@5 会从 95.8% 崩到 33.3%，所以它只作为对照，不作为默认。
   */
  configs: [
    { key: 'fp16', label: 'fp16（主用 / WebGPU）', dtype: 'fp16', device: 'webgpu' },
    { key: 'q8', label: 'q8（INT8 / WASM）', dtype: 'q8', device: 'wasm' },
    { key: 'fp32', label: 'fp32（基线）', dtype: 'fp32', device: 'webgpu' },
  ],
  corpusUrl: './samples/corpus.jsonl',
  queriesUrl: './samples/queries.jsonl',
  outputId: 'compare-output',
  maxDocChars: 4000,
  fetchTimeoutMs: 15000,
  loadTimeoutMs: 10 * 60 * 1000,
  embedTimeoutMs: 4 * 60 * 1000,
  /** 调用方可直接给已切好的文档（优先复用页面语料），也可给原始文本让本模块切。 */
  docs: null,
  corpusText: '',
  transformers: null,
  log: null,
  setProgress: null,
};

/**
 * 权重文件路径：与 transformers.js 的 DEFAULT_DTYPE_SUFFIX_MAPPING 一致，
 * q8 → onnx/model_quantized.onnx，fp32 → 不带后缀的 onnx/model.onnx。
 * （仓库里的 model_int8.onnx 与 model_quantized.onnx 是同一 blob，但 dtype 名不同不可混用。）
 */
const DTYPE_FILES = {
  fp16: 'onnx/model_fp16.onnx',
  q8: 'onnx/model_quantized.onnx',
  fp32: 'onnx/model.onnx',
};

const SHARED_MODEL_FILES = ['tokenizer.json', 'config.json'];

/** qrels 字段名以 samples/queries.jsonl 的真实字段为准（relevant），其余是容错别名。 */
const RELEVANT_FIELDS = ['relevant', 'relevant_ids', 'qrels', 'gold'];
const QUERY_FIELDS = ['query', 'question', 'text'];

/* ============================================================ 小工具 */

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

function esc(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function fmtMs(ms) {
  if (ms == null || Number.isNaN(ms)) return '—';
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

function fmtMb(bytes) {
  if (bytes == null || !Number.isFinite(bytes)) return '—';
  return (bytes / 1024 / 1024).toFixed(2);
}

function fmtPct(value) {
  if (value == null || Number.isNaN(value)) return '—';
  return `${(value * 100).toFixed(1)}%`;
}

function errText(err) {
  if (!err) return '未知错误';
  if (err.name === 'AbortError') return '请求超时或被中断';
  if (err.name === 'TimeoutError') return '请求超时';
  return err.message || String(err);
}

/** 评测专用错误：kind 用来区分「模型下载失败」与「评测失败」。 */
class CompareError extends Error {
  constructor(kind, message) {
    super(message);
    this.name = 'CompareError';
    this.kind = kind;
  }
}

/** import() 无法中止，只能加个上限，避免按钮永久转圈。 */
function withTimeout(promise, ms, label) {
  let timer;
  const guard = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} 超时（>${Math.round(ms / 1000)}s）`)), ms);
  });
  return Promise.race([promise, guard]).finally(() => clearTimeout(timer));
}

function fetchSignal(ms) {
  try {
    return AbortSignal.timeout(ms);
  } catch {
    return undefined;
  }
}

function canUseBrowserCache() {
  try {
    return Boolean(window.isSecureContext) && typeof caches !== 'undefined';
  } catch {
    return false;
  }
}

const normText = (value) => String(value ?? '').toLowerCase().replace(/\s+/g, ' ').trim();

const average = (list) => (list.length ? list.reduce((a, b) => a + b, 0) / list.length : null);

/* ============================================================ 数据读取 */

async function loadJsonl(url, { label, timeoutMs, required }) {
  let res;
  try {
    res = await fetch(url, { cache: 'no-store', signal: fetchSignal(timeoutMs) });
  } catch (err) {
    if (required) throw new CompareError('eval', `拉取${label}失败（${url}）：${errText(err)}`);
    return null;
  }
  if (!res.ok) {
    if (required) throw new CompareError('eval', `拉取${label}失败（${url}）：HTTP ${res.status}`);
    return null;
  }
  const text = await res.text();
  const rows = [];
  const lines = text.split('\n');
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i].trim();
    if (!line) continue;
    try {
      rows.push(JSON.parse(line));
    } catch (err) {
      throw new CompareError('eval', `${label}第 ${i + 1} 行不是合法 JSON：${errText(err)}`);
    }
  }
  return rows;
}

/** 与 app.js 的 splitIntoDocs 同口径：空行分段，超过 maxDocChars 再硬切。 */
function splitCorpusText(text, maxDocChars) {
  const blocks = String(text)
    .replace(/\r\n/g, '\n')
    .split(/\n\s*\n/)
    .map((s) => s.trim())
    .filter(Boolean);
  const docs = [];
  for (const block of blocks) {
    if (block.length <= maxDocChars) {
      docs.push(block);
    } else {
      for (let i = 0; i < block.length; i += maxDocChars) docs.push(block.slice(i, i + maxDocChars));
    }
  }
  return docs;
}

async function loadQueries(opts) {
  const rows = await loadJsonl(opts.queriesUrl, {
    label: '查询集 queries.jsonl',
    timeoutMs: opts.fetchTimeoutMs,
    required: true,
  });
  if (!rows || !rows.length) throw new CompareError('eval', `${opts.queriesUrl} 没有任何条目`);

  const queries = [];
  rows.forEach((row, index) => {
    if (!row || typeof row !== 'object') return;
    const textField = QUERY_FIELDS.find((name) => typeof row[name] === 'string' && row[name].trim());
    if (!textField) return;
    const relevantField = RELEVANT_FIELDS.find((name) => Array.isArray(row[name]));
    if (!relevantField) return;
    const relevant = row[relevantField].map((value) => String(value));
    if (!relevant.length) return;
    queries.push({
      id: row.id == null ? `q${index + 1}` : String(row.id),
      text: String(row[textField]).trim(),
      relevant,
    });
  });
  if (!queries.length) {
    throw new CompareError('eval', 'queries.jsonl 里没有同时带 query 与 relevant 的条目，无法计算 Recall / MRR');
  }
  return queries;
}

/**
 * 给每篇文档配一个稳定的 key，用来和 qrels 的 relevant 对齐。
 * 页面语料是按「标题 + 正文」导入的，所以先按首行标题匹配，再按全文匹配。
 * 匹配不到就退化成 doc{n}（此时该文档不会贡献任何命中，指标口径不受影响）。
 */
function annotateDocs(docs, corpusRows) {
  const byTitle = new Map();
  const byText = new Map();
  for (const row of corpusRows) {
    if (!row || row.id == null) continue;
    const id = String(row.id);
    if (row.title) byTitle.set(normText(row.title), id);
    if (row.text) byText.set(normText(row.text), id);
  }
  const keys = docs.map((doc, index) => {
    const lines = doc.text.split('\n');
    const firstLine = (lines[0] || '').trim();
    const restText = lines.length > 1 ? lines.slice(1).join('\n') : '';
    const hit =
      byTitle.get(normText(firstLine)) ||
      byText.get(normText(doc.text)) ||
      (restText ? byText.get(normText(restText)) : '');
    return hit || `doc${index + 1}`;
  });
  return { keys };
}

function selectUsableQueries(queries, docKeys) {
  const known = new Set(docKeys);
  const usable = [];
  let skipped = 0;
  for (const query of queries) {
    const relevant = query.relevant.filter((key) => known.has(key));
    if (!relevant.length) {
      skipped += 1;
      continue;
    }
    usable.push({ id: query.id, text: query.text, relevant });
  }
  return { usable, skipped };
}

/**
 * 确定本次评测的文档集合与查询集合。
 * 同一批文档在两种 dtype 下共用（一份数组，两遍只读遍历），这是可比性的前提。
 */
async function prepareDataset(opts, notes) {
  const corpusRows = await loadJsonl(opts.corpusUrl, {
    label: '语料 corpus.jsonl',
    timeoutMs: opts.fetchTimeoutMs,
    required: false,
  });
  const queries = await loadQueries(opts);

  let docs = [];
  let docSource = '';
  if (Array.isArray(opts.docs) && opts.docs.length) {
    for (const doc of opts.docs) {
      const text = String(doc && doc.text != null ? doc.text : '').trim();
      if (text) docs.push({ text });
    }
    docSource = '页面已导入文本';
  } else if (String(opts.corpusText || '').trim()) {
    docs = splitCorpusText(opts.corpusText, opts.maxDocChars).map((text) => ({ text }));
    docSource = '页面粘贴文本';
  }

  let annotated = annotateDocs(docs, corpusRows || []);
  let selected = selectUsableQueries(queries, annotated.keys);

  if (docs.length && !selected.usable.length) {
    // 页面语料是用户自己的文本时，queries.jsonl 的 d01… 无从对应：回退示例语料，保证指标可算
    notes.push(
      '页面语料与 queries.jsonl 的 relevant 文档 id 对不上，已回退到 samples/corpus.jsonl 评测（两者文档集不同，指标不可与其它报告混用）'
    );
    docs = [];
  }

  if (!docs.length) {
    if (!corpusRows || !corpusRows.length) {
      throw new CompareError('eval', `没有可用语料：页面未导入文本，且 ${opts.corpusUrl} 取不到`);
    }
    docs = corpusRows
      .map((row) => ({ text: String((row && (row.text || row.title)) || '').trim() }))
      .filter((doc) => doc.text);
    docSource = 'samples/corpus.jsonl';
    annotated = annotateDocs(docs, corpusRows);
    selected = selectUsableQueries(queries, annotated.keys);
  }

  if (!docs.length) throw new CompareError('eval', '语料为空，无法评测');
  if (!selected.usable.length) {
    throw new CompareError('eval', '没有任何查询能与语料里的文档 id 对应（relevant 全部落空），无法计算 Recall / MRR');
  }

  return {
    docs,
    docKeys: annotated.keys,
    docSource,
    queries: selected.usable,
    skippedQueries: selected.skipped,
    totalQueries: queries.length,
  };
}

/* ============================================================ 模型体积 */

const modelUrl = (host, modelId, file) => `${String(host).replace(/\/+$/, '')}/${modelId}/resolve/main/${file}`;

/**
 * 取单个模型文件的真实体积。
 * 首选 HEAD（HF 对 resolve 路径是 302 → CDN，content-length 属于 CORS 安全响应头，可直接读）；
 * LFS 文件还带 x-linked-size；两者都没有时用 2 字节 Range 读 content-range 的总量。
 */
async function headBytes(url, timeoutMs) {
  const res = await fetch(url, { method: 'HEAD', signal: fetchSignal(timeoutMs) });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const direct = Number(res.headers.get('content-length'));
  if (Number.isFinite(direct) && direct > 0) return direct;
  const linked = Number(res.headers.get('x-linked-size'));
  if (Number.isFinite(linked) && linked > 0) return linked;
  return rangeBytes(url, timeoutMs);
}

async function rangeBytes(url, timeoutMs) {
  const res = await fetch(url, { headers: { Range: 'bytes=0-1' }, signal: fetchSignal(timeoutMs) });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const total = /\/(\d+)\s*$/.exec(res.headers.get('content-range') || '');
  if (total) return Number(total[1]);
  const length = Number(res.headers.get('content-length'));
  if (Number.isFinite(length) && length > 0) return length;
  throw new Error('响应里没有可读的文件大小');
}

async function measureModelSize(opts, dtype, notes) {
  const weightFile = DTYPE_FILES[dtype] || DTYPE_FILES.fp32;
  let host = null;
  let weightBytes = null;

  for (const candidate of opts.modelHosts) {
    try {
      weightBytes = await headBytes(modelUrl(candidate, opts.modelId, weightFile), opts.fetchTimeoutMs);
      host = candidate;
      break;
    } catch (err) {
      notes.push(`体积：${candidate} 取 ${weightFile} 失败（${errText(err)}）`);
    }
  }

  if (host == null) {
    notes.push(`体积：${dtype} 的权重文件在所有 host 都取不到，体积与压缩比记为「—」（取不到就跳过，不估算）`);
    return { dtype, available: false, bytes: null, host: null, weightBytes: null, missing: [weightFile, ...SHARED_MODEL_FILES] };
  }

  let bytes = weightBytes;
  const missing = [];
  for (const file of SHARED_MODEL_FILES) {
    try {
      bytes += await headBytes(modelUrl(host, opts.modelId, file), opts.fetchTimeoutMs);
    } catch {
      missing.push(file);
    }
  }
  if (missing.length) {
    notes.push(`体积：${dtype} 在 ${host} 上取不到 ${missing.join('、')}，合计值只含取到的文件`);
  }
  return { dtype, available: true, bytes, host, weightBytes, missing };
}

/* ============================================================ 推理管线 */

async function loadTransformersModule(opts) {
  if (opts.transformers) return opts.transformers;
  try {
    return await withTimeout(
      import(/* @vite-ignore */ opts.transformersUrl),
      opts.fetchTimeoutMs,
      '拉取 Transformers.js CDN'
    );
  } catch (err) {
    throw new CompareError('model-download', `无法加载 Transformers.js（${opts.transformersUrl}）：${errText(err)}`);
  }
}

/** 探测 WebGPU：有 API 且能拿到 adapter 才算可用（与 app.js 的 detectWebGPU 同口径）。 */
async function detectWebGpu() {
  try {
    if (!('gpu' in navigator) || !navigator.gpu) {
      return { ok: false, reason: 'navigator.gpu 不存在（非安全上下文或浏览器不支持）' };
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { ok: false, reason: 'requestAdapter() 返回 null' };
    return { ok: true, reason: null };
  } catch (err) {
    return { ok: false, reason: errText(err) };
  }
}

async function createExtractor(mod, opts, attempt, ctx) {
  const { pipeline, env } = mod;
  env.allowLocalModels = false;
  env.useBrowserCache = canUseBrowserCache();
  const previousHost = env.remoteHost;
  env.remoteHost = attempt.host;
  try {
    const extractor = await withTimeout(
      pipeline('feature-extraction', opts.modelId, {
        device: attempt.device,
        dtype: attempt.dtype,
        progress_callback: (p) => {
          if (p && p.status === 'progress' && typeof p.progress === 'number') {
            ctx.progress(
              p.progress / 100,
              `下载模型（${attempt.device}/${attempt.dtype} @ ${String(attempt.host).replace(/^https?:\/\//, '')}）`,
              `${Math.round(p.progress)}%`
            );
          }
        },
      }),
      opts.loadTimeoutMs,
      `加载模型（${attempt.device}/${attempt.dtype}）`
    );

    // 探针：device/dtype 声称可用不代表首次前向能跑，维度也要符合预期
    const probe = await extractor(['warmup'], { pooling: 'mean', normalize: true });
    const vector = typeof probe.tolist === 'function' ? probe.tolist()[0] : probe[0];
    const dimension = Array.isArray(vector) ? vector.length : 0;
    if (dimension !== opts.dimension) {
      throw new Error(`探针向量维度 ${dimension}，期望 ${opts.dimension}`);
    }
    return extractor;
  } finally {
    if (previousHost === undefined) delete env.remoteHost;
    else env.remoteHost = previousHost;
  }
}

async function embedTexts(extractor, texts, opts, onProgress) {
  const batch = Math.max(1, Math.trunc(opts.batch) || DEFAULT_OPTIONS.batch);
  const vectors = [];
  for (let i = 0; i < texts.length; i += batch) {
    const slice = texts.slice(i, i + batch);
    const out = await withTimeout(
      extractor(slice, { pooling: 'mean', normalize: true }),
      opts.embedTimeoutMs,
      `向量化 ${slice.length} 条`
    );
    const list = typeof out.tolist === 'function' ? out.tolist() : out;
    if (!Array.isArray(list) || list.length !== slice.length) {
      throw new Error(`向量化返回 ${Array.isArray(list) ? list.length : '非数组'} 条，期望 ${slice.length} 条`);
    }
    for (const vector of list) vectors.push(Array.from(vector));
    if (onProgress) onProgress(Math.min(i + slice.length, texts.length), texts.length);
  }
  return vectors;
}

/* ============================================================ 相似度与指标 */

/** 两种 dtype 共用这一个余弦实现，是「同一相似度口径」的唯一来源。 */
function cosine(a, b) {
  const n = Math.min(a.length, b.length);
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < n; i += 1) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / Math.sqrt(na * nb);
}

/** 返回按相似度降序的文档下标；同分时按文档序，保证两种 dtype 的排序完全确定。 */
function rankDocs(queryVector, docVectors) {
  const scored = docVectors.map((vector, index) => ({ index, score: cosine(queryVector, vector) }));
  scored.sort((a, b) => b.score - a.score || a.index - b.index);
  return scored.map((item) => item.index);
}

function recallAtK(ranking, relevant, k) {
  const top = ranking.slice(0, k);
  let hit = 0;
  for (const key of top) if (relevant.has(key)) hit += 1;
  return hit / relevant.size;
}

function reciprocalRank(ranking, relevant, k) {
  const limit = Math.min(k, ranking.length);
  for (let i = 0; i < limit; i += 1) {
    if (relevant.has(ranking[i])) return 1 / (i + 1);
  }
  return 0;
}

function scoreConfig(docKeys, docVectors, queries, queryVectors) {
  const recall5 = [];
  const recall10 = [];
  const mrr10 = [];
  for (let i = 0; i < queries.length; i += 1) {
    const ranking = rankDocs(queryVectors[i], docVectors).map((index) => docKeys[index]);
    const relevant = new Set(queries[i].relevant);
    recall5.push(recallAtK(ranking, relevant, 5));
    recall10.push(recallAtK(ranking, relevant, 10));
    mrr10.push(reciprocalRank(ranking, relevant, 10));
  }
  return { recall5: average(recall5), recall10: average(recall10), mrr10: average(mrr10) };
}

/* ============================================================ 跑一种配置 */

function makeContext(opts, { base, span, gpu, notes, log }) {
  const report = typeof opts.setProgress === 'function' ? opts.setProgress : () => {};
  return {
    gpu,
    notes,
    progress(local, label, count) {
      const clamped = Math.max(0, Math.min(1, Number(local) || 0));
      report(true, Math.max(0, Math.min(1, base + clamped * span)), label || '量化对比评测…', count || '');
    },
    info(message) {
      try {
        if (typeof log === 'function') log(message);
      } catch {
        // 日志失败不影响评测
      }
    },
  };
}

/**
 * 候选尝试链：**尊重配置里的 device**，只在它失败后才退到另一个后端。
 *
 * 之前实现忽略 cfg.device、总是先试 webgpu，导致「q8 / WASM」那一行实际跑在 WebGPU 上，
 * 表里会把 INT8 算子在 WebGPU 上的已知缺陷算到 WASM 头上。现在 device 由配置决定。
 */
function buildAttempts(cfg, hostChain, gpuOk) {
  const preferred = cfg.device === 'wasm' ? ['wasm', 'webgpu'] : gpuOk ? ['webgpu', 'wasm'] : ['wasm'];
  const devices = preferred.filter((device) => device !== 'webgpu' || gpuOk);
  const attempts = [];
  for (const device of devices) {
    for (const host of hostChain) attempts.push({ device, dtype: cfg.dtype, host });
  }
  return attempts;
}

async function runOneConfig(opts, mod, dataset, cfg, ctx) {
  const attempts = buildAttempts(cfg, ctx.hostChain, ctx.gpu.ok);
  let phase = 'load';
  let lastError = null;

  for (const attempt of attempts) {
    let extractor = null;
    phase = 'load';
    try {
      ctx.info(`配置 ${cfg.label}：尝试 device=${attempt.device} dtype=${attempt.dtype}`);
      extractor = await createExtractor(mod, opts, attempt, ctx);

      phase = 'embed';
      ctx.progress(0, `向量化文档（${cfg.label}）`, `0 / ${dataset.docs.length}`);
      const docStart = now();
      const docVectors = await embedTexts(
        extractor,
        dataset.docs.map((doc) => doc.text),
        opts,
        (done, total) => ctx.progress(done / total, `向量化文档（${cfg.label}）`, `${done} / ${total}`)
      );
      const docEmbedMs = now() - docStart;

      const queryStart = now();
      const queryVectors = await embedTexts(
        extractor,
        dataset.queries.map((query) => query.text),
        opts,
        (done, total) => ctx.progress(done / total, `向量化查询（${cfg.label}）`, `${done} / ${total}`)
      );
      const queryEmbedMs = now() - queryStart;

      phase = 'metrics';
      const metrics = scoreConfig(dataset.docKeys, docVectors, dataset.queries, queryVectors);

      return {
        key: cfg.key,
        label: cfg.label,
        dtype: attempt.dtype,
        device: attempt.device,
        host: attempt.host,
        downgraded: attempt.device !== cfg.device && ctx.gpu.ok,
        docEmbedMs,
        docEmbedPerDocMs: docEmbedMs / Math.max(1, dataset.docs.length),
        queryEmbedMs,
        ...metrics,
      };
    } catch (err) {
      lastError = err;
      ctx.notes.push(`配置 ${cfg.label}：device=${attempt.device} dtype=${attempt.dtype} 失败（${errText(err)}）`);
    } finally {
      // 两种配置串行跑，跑完立刻释放：fp32 与 q8 同时在 GPU 上会叠加显存
      try {
        if (extractor && typeof extractor.dispose === 'function') await extractor.dispose();
      } catch {
        // 释放失败不影响结果
      }
    }
  }

  const kind = phase === 'load' ? 'model-download' : 'eval';
  throw new CompareError(kind, `配置 ${cfg.label} 的所有后端都失败：${errText(lastError)}`);
}

/* ============================================================ 渲染 */

const DISCLAIMER =
  '质量对比在客户端完成，仅用于量化影响评估：排序由本页用余弦相似度完成，' +
  '应用主流程的检索仍走 seekdb 混合检索（向量 + 全文 RRF），名次与线上不完全一致。';

function renderReport(box, report) {
  if (!box) return;
  const baseline = report.rows.find((row) => row.dtype === 'fp32' && row.bytes) || null;
  const rowsHtml = report.rows
    .map((row) => {
      const ratio = row.bytes && baseline && baseline.bytes ? baseline.bytes / row.bytes : null;
      const deviceCell = row.downgraded ? `${row.device} ⚠` : row.device;
      const deviceTitle = row.downgraded ? `目标 device 跑不起来，已降级到 ${row.device}` : '';
      const sizeTitle = row.bytes
        ? `${row.host || ''} · ${row.sizeMissing.length ? `缺 ${row.sizeMissing.join('、')}` : '权重 + tokenizer.json + config.json'}`
        : '取不到文件大小';
      return `<tr class="${row.dtype === 'fp32' ? 'is-baseline' : ''}">
        <td>${esc(row.label)}</td>
        <td title="${esc(deviceTitle)}">${esc(deviceCell)}</td>
        <td title="${esc(sizeTitle)}">${fmtMb(row.bytes)}</td>
        <td>${ratio == null ? '—' : `${ratio.toFixed(2)}×`}</td>
        <td>${fmtPct(row.recall5)}</td>
        <td>${fmtPct(row.recall10)}</td>
        <td>${row.mrr10 == null ? '—' : row.mrr10.toFixed(3)}</td>
        <td>${fmtMs(row.docEmbedMs)}</td>
        <td>${fmtMs(row.docEmbedPerDocMs)}</td>
      </tr>`;
    })
    .join('');

  const notesHtml = [
    `语料：${report.dataset.docSource} · ${report.dataset.docs} 篇；查询：${report.dataset.queries} / ${report.dataset.totalQueries} 条参与评测` +
      (report.dataset.skippedQueries ? `（${report.dataset.skippedQueries} 条因 relevant 无法对应被跳过）` : ''),
    '体积取法：HEAD {host}/{modelId}/resolve/main/onnx/{file}，q8 → model_quantized.onnx，fp32 → model.onnx，另加 tokenizer.json 与 config.json',
    '体积单位：1 MB = 1024 × 1024 字节（即 MiB），与浏览器缓存里落盘的字节数一致',
    `相似度口径：两种配置都由 pipeline(normalize: true) 输出单位向量，再用同一个余弦函数排序，TopK 与 qrels 完全一致`,
    ...report.notes,
  ]
    .map((line) => `<li>${esc(line)}</li>`)
    .join('');

  box.innerHTML = `
    <table class="compare-table">
      <thead>
        <tr>
          <th>配置</th><th>设备</th><th>模型体积(MB)</th><th>压缩比</th>
          <th>Recall@5</th><th>Recall@10</th><th>MRR@10</th>
          <th>向量化耗时</th><th>每文档均耗时</th>
        </tr>
      </thead>
      <tbody>${rowsHtml}</tbody>
    </table>
    <p class="compare-disclaimer">${esc(DISCLAIMER)}</p>
    <ul class="compare-notes">${notesHtml}</ul>
  `;
}

function renderError(box, kind, message) {
  if (!box) return;
  const title = kind === 'model-download' ? '模型下载失败' : '评测失败';
  const hint =
    kind === 'model-download'
      ? '请检查网络或把 modelHosts / transformersUrl 指向可访问的镜像后重试。'
      : '数据或指标计算环节出错，可先点「加载示例语料」再试一次。';
  box.innerHTML = `
    <div class="compare-error">
      <strong>${esc(title)}</strong>
      <p>${esc(message)}</p>
      <p class="muted">${esc(hint)}</p>
    </div>
  `;
}

/* ============================================================ 入口 */

function normalizeOptions(options) {
  const opts = { ...DEFAULT_OPTIONS, configs: DEFAULT_OPTIONS.configs.slice() };
  for (const [key, value] of Object.entries(options || {})) {
    if (value !== undefined && value !== null) opts[key] = value;
  }
  opts.batch = Math.max(1, Math.trunc(Number(opts.batch)) || DEFAULT_OPTIONS.batch);
  return opts;
}

/**
 * 跑一次量化对比评测。
 * 刻意不向外抛异常：按钮回调里的 promise rejection 会变成未捕获异常，
 * 所以这里统一把失败转成 { ok: false, errorKind, message }，并把可读错误渲染进结果区。
 */
export async function runQuantizationComparison(options = {}) {
  const opts = normalizeOptions(options);
  const box = typeof document !== 'undefined' ? document.getElementById(opts.outputId) : null;
  const notes = [];
  const log = typeof opts.log === 'function' ? opts.log : null;
  const started = now();

  try {
    if (typeof opts.setProgress === 'function') opts.setProgress(true, 0.01, '量化对比：读取语料与查询…', '');

    // 1) 同一批文档与查询：只解析一次，两种 dtype 共用同一份（只读遍历）
    const dataset = await prepareDataset(opts, notes);

    // 2) 体积测量同时也是 host 可达性探测，结果决定模型从哪里下载
    const sizes = [];
    for (let i = 0; i < opts.configs.length; i += 1) {
      if (typeof opts.setProgress === 'function') {
        opts.setProgress(true, 0.03 + i * 0.02, `测量模型体积（${opts.configs[i].label}）…`, '');
      }
      sizes.push(await measureModelSize(opts, opts.configs[i].dtype, notes));
    }
    const sizeByDtype = new Map(sizes.map((size) => [size.dtype, size]));
    const hostChain = [];
    for (const size of sizes) {
      if (size.host && !hostChain.includes(size.host)) hostChain.push(size.host);
    }
    for (const host of opts.modelHosts) if (!hostChain.includes(host)) hostChain.push(host);

    // 3) 逐个配置串行跑，避免两份权重同时占显存
    const mod = await loadTransformersModule(opts);
    const gpu = await detectWebGpu();
    if (!gpu.ok) notes.push(`WebGPU 不可用（${gpu.reason}），两种配置都使用 wasm`);

    const rows = [];
    const bandStart = 0.06;
    const bandSpan = 0.9 / opts.configs.length;
    for (let i = 0; i < opts.configs.length; i += 1) {
      const ctx = makeContext(opts, {
        base: bandStart + i * bandSpan,
        span: bandSpan,
        gpu,
        notes,
        log,
      });
      ctx.hostChain = hostChain;
      if (typeof log === 'function') log(`量化对比：开始 ${opts.configs[i].label}（语料 ${dataset.docs.length} 篇 / 查询 ${dataset.queries.length} 条）`);
      const row = await runOneConfig(opts, mod, dataset, opts.configs[i], ctx);
      const size = sizeByDtype.get(row.dtype) || {};
      rows.push({
        ...row,
        bytes: size.bytes == null ? null : size.bytes,
        host: size.host || row.host,
        sizeMissing: size.missing || [],
      });
      if (typeof log === 'function') {
        log(
          `量化对比：${row.label} 完成（${row.device}/${row.dtype}）· Recall@5 ${fmtPct(row.recall5)} · ` +
            `Recall@10 ${fmtPct(row.recall10)} · MRR@10 ${row.mrr10 == null ? '—' : row.mrr10.toFixed(3)}`,
          'ok'
        );
      }
    }

    const report = {
      rows,
      sizes,
      notes,
      dataset: {
        docSource: dataset.docSource,
        docs: dataset.docs.length,
        queries: dataset.queries.length,
        totalQueries: dataset.totalQueries,
        skippedQueries: dataset.skippedQueries,
      },
      elapsedMs: now() - started,
    };
    renderReport(box, report);
    if (typeof opts.setProgress === 'function') opts.setProgress(true, 1, '量化对比完成', '');
    return { ok: true, ...report };
  } catch (err) {
    const kind = err instanceof CompareError ? err.kind : 'eval';
    const message = errText(err);
    renderError(box, kind, message);
    if (typeof log === 'function') {
      log(`${kind === 'model-download' ? '模型下载失败' : '评测失败'}：${message}`, 'err');
    }
    console.error('[compare]', err);
    return { ok: false, errorKind: kind, message, notes };
  } finally {
    if (typeof opts.setProgress === 'function') {
      try {
        opts.setProgress(false);
      } catch {
        // 进度条复位失败不影响结果
      }
    }
  }
}
