/**
 * Edge-KB 前端（原生 ESM，无打包器）。
 *
 * 职责划分（对应题目要求）：
 *   1. 用 Transformers.js 在浏览器内把「文档」和「查询」都编码成 384 维向量 —— 后端不参与任何向量计算；
 *   2. 向量 + 原文 + metadata 通过 HTTP 写入 seekdb；
 *   3. 查询时只把 queryVector / queryText 交给后端，由 seekdb 的 hybridSearch 做 RRF 融合；
 *   4. WebGPU 优先，失败自动降级 WASM，并在顶部徽标明确显示当前后端。
 *   5. 量化对比评测（q8 vs fp32）委托给 compare.js，只在浏览器内做质量对照，不改动 seekdb。
 */

import { runQuantizationComparison } from './compare.js';

const CFG = {
  /** 已在 jsDelivr 上核实 3.7.5 存在（package.json 的 jsdelivr 字段指向 dist/transformers.min.js）。 */
  transformersUrl: 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5',
  modelId: 'Xenova/all-MiniLM-L6-v2',
  /**
   * 模型仓库基址，按序尝试。huggingface.co 在部分网络下不可达，此时回退到镜像；
   * 只影响模型文件从哪里下载，推理始终发生在浏览器里。
   */
  modelHosts: ['https://huggingface.co', 'https://hf-mirror.com'],
  dimension: 384,
  defaultEmbedBatch: 8,
  defaultWriteBatch: 32,
  modelLoadTimeoutMs: 10 * 60 * 1000,
  embedTimeoutMs: 3 * 60 * 1000,
  apiTimeoutMs: 60 * 1000,
  maxDocChars: 4000,
  snippetChars: 260,
};

const STORAGE = {
  collection: 'edgekb.collection',
  embedBatch: 'edgekb.embedBatch',
  writeBatch: 'edgekb.writeBatch',
};

const $ = (id) => document.getElementById(id);

/* ============================================================ 通用工具 */

/** 给任意 Promise 加超时，防止模型加载 / 请求永久挂起。 */
function withTimeout(promise, ms, label) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} 超时（>${Math.round(ms / 1000)}s）`)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

/** 带超时与错误归一化的 JSON 请求。 */
async function fetchJson(url, { method = 'GET', body, timeoutMs = CFG.apiTimeoutMs } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method,
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    });
    const text = await res.text();
    let data = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        // 非 JSON（例如代理返回的 HTML 错误页）
        throw new Error(`响应不是 JSON（HTTP ${res.status}）：${text.slice(0, 160)}`);
      }
    }
    if (!res.ok) {
      const msg = (data && (data.error || data.message)) || `HTTP ${res.status}`;
      throw new Error(msg);
    }
    return data;
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new Error(`请求超时（${Math.round(timeoutMs / 1000)}s）：${url}`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

function fmtMs(ms) {
  if (ms == null || Number.isNaN(ms)) return '—';
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

/** 把模型仓库基址缩成日志里好读的短标签。 */
function hostLabel(host) {
  return String(host || '').replace(/^https?:\/\//, '');
}

/** Cache API（以及 WebGPU）只在安全上下文可用：https、localhost、或 Chrome 的显式放行。 */
function canUseBrowserCache() {
  try {
    return window.isSecureContext && typeof caches !== 'undefined';
  } catch {
    return false;
  }
}

/** 英文功能词：全文分支不应该拿它们去匹配。 */
const KEYWORD_STOPWORDS = new Set([
  'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'than', 'that', 'this', 'these', 'those',
  'is', 'are', 'was', 'were', 'be', 'been', 'being', 'do', 'does', 'did', 'done',
  'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'about', 'from', 'as', 'into', 'over',
  'it', 'its', 'we', 'you', 'they', 'he', 'she', 'i', 'me', 'my', 'our', 'their',
  'how', 'what', 'why', 'when', 'where', 'which', 'who', 'whom',
  'can', 'could', 'should', 'would', 'will', 'shall', 'may', 'might', 'must',
  'not', 'no', 'yes', 'so', 'such', 'also', 'more', 'most', 'much', 'many',
  'there', 'here', 'between', 'during', 'through', 'under', 'above', 'below',
]);

/**
 * 把自然语言问题压成关键词串，供 seekdb 的全文分支使用。
 * 只影响「拿什么文本去匹配」，匹配本身仍由 seekdb 完成。
 */
function buildKeywordQuery(question) {
  const tokens = String(question || '').toLowerCase().match(/[a-z0-9][a-z0-9+#._-]*/g) || [];
  const kept = [];
  for (const token of tokens) {
    if (token.length < 3 || KEYWORD_STOPWORDS.has(token)) continue;
    if (kept.includes(token)) continue;
    kept.push(token);
    if (kept.length === 8) break;
  }
  return kept.length ? kept.join(' ') : String(question || '').trim();
}

function escapeHtml(str) {
  return String(str ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function log(message, level = 'info') {
  const el = $('log');
  if (!el) return;
  const time = new Date().toLocaleTimeString('zh-CN', { hour12: false });
  const line = document.createElement('span');
  line.className = level;
  line.innerHTML = `<span class="t">[${time}]</span> ${escapeHtml(message)}`;
  el.appendChild(line);
  el.appendChild(document.createTextNode('\n'));
  el.scrollTop = el.scrollHeight;
}

function setBusy(busy, ...ids) {
  for (const id of ids) {
    const el = $(id);
    if (el) el.disabled = busy;
  }
}

/* ============================================================ 全局状态 */

const state = {
  /** @type {any} */ transformers: null,
  /** @type {any} */ extractor: null,
  /** 'webgpu' | 'wasm' | null */ device: null,
  /** 'q8' | 'fp16' | 'fp32' | null */ dtype: null,
  attempts: [],
  modelLoadMs: null,
  loadPromise: null,
  collection: localStorage.getItem(STORAGE.collection) || 'kb_default',
};

function readInt(id, fallback, min, max) {
  const v = Number.parseInt($(id).value, 10);
  if (!Number.isFinite(v)) return fallback;
  return Math.min(max, Math.max(min, v));
}

/* ============================================================ 模型加载 */

/** 探测 WebGPU 可用性：navigator.gpu 存在且能拿到 adapter 才算可用。 */
async function detectWebGPU() {
  if (!('gpu' in navigator) || !navigator.gpu) {
    return { ok: false, reason: 'navigator.gpu 不存在（非安全上下文或浏览器不支持）' };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { ok: false, reason: 'requestAdapter() 返回 null（无可用 GPU 适配器）' };
    return { ok: true, reason: null };
  } catch (err) {
    return { ok: false, reason: `requestAdapter() 抛错：${err.message || err}` };
  }
}

/**
 * 生成候选后端链：WebGPU(fp16 → fp32) → WASM(q8 → fp32)。
 *
 * 为什么不把 q8 放在 WebGPU 首位：onnxruntime-web 的 WebGPU EP 对 INT8 算子支持不完整，
 * 实测同一批语料 Recall@5 从 95.8% 掉到 33.3%（见「量化对比评测」面板）。
 * FP16 在 WebGPU 上是原生算子，体积约为 fp32 的一半且质量基本无损。
 */
function buildCandidates(gpu) {
  const list = [];
  if (gpu.ok) {
    list.push({ device: 'webgpu', dtype: 'fp16' });
    list.push({ device: 'webgpu', dtype: 'fp32' });
  }
  list.push({ device: 'wasm', dtype: 'q8' });
  list.push({ device: 'wasm', dtype: 'fp32' });
  return list;
}

function setProgress(visible, ratio = 0, label = '', count = '') {
  const wrap = $('progress-wrap');
  if (!wrap) return;
  wrap.hidden = !visible;
  const pct = Math.max(0, Math.min(100, Math.round(ratio * 100)));
  $('progress-bar').style.width = `${pct}%`;
  if (label) $('progress-label').textContent = label;
  $('progress-count').textContent = count;
}

function paintBackendPill() {
  const pill = $('pill-backend');
  const value = $('backend-value');
  pill.classList.remove('is-webgpu', 'is-wasm', 'is-error');
  if (state.device === 'webgpu') {
    pill.classList.add('is-webgpu');
    value.textContent = 'WebGPU';
  } else if (state.device === 'wasm') {
    pill.classList.add('is-wasm');
    value.textContent = 'WASM';
  } else {
    value.textContent = '未加载';
  }
  $('dtype-value').textContent = state.dtype ? state.dtype.toUpperCase() : '—';
  $('model-value').textContent = CFG.modelId;
  $('model-value').title = `${CFG.modelId} · ${state.device || '-'} · ${state.dtype || '-'}`;
  $('dimension-value').textContent = String(CFG.dimension);
}

/**
 * 加载（并按需降级）推理管线。并发调用会复用同一个 Promise。
 * 返回 { device, dtype }。
 */
async function ensureModel() {
  if (state.extractor) return { device: state.device, dtype: state.dtype };
  if (state.loadPromise) return state.loadPromise;

  state.loadPromise = (async () => {
    const started = performance.now();
    setProgress(true, 0, '加载 Transformers.js…', '');
    // 动态 import：CDN 不可用时能给出可读错误，而不是整页白屏
    let mod;
    try {
      mod = await withTimeout(
        import(/* @vite-ignore */ CFG.transformersUrl),
        CFG.apiTimeoutMs,
        '拉取 Transformers.js CDN'
      );
    } catch (err) {
      throw new Error(
        `无法从 CDN 加载 Transformers.js（${CFG.transformersUrl}）：${err.message || err}。` +
          '请检查网络，或把 transformersUrl 指向可访问的镜像。'
      );
    }

    const { pipeline, env } = mod;
    state.transformers = mod;
    env.allowLocalModels = false; // 只用远端模型仓库，避免依赖本地 /models
    // Cache API 只存在于安全上下文（https / localhost）。用局域网 IP + http 打开时必须关掉，
    // 否则 Transformers.js 会直接抛 "Browser cache is not available" 让整条链路失败。
    const cacheable = canUseBrowserCache();
    env.useBrowserCache = cacheable;
    if (!cacheable) {
      log('当前上下文不支持 Cache API（非 https/localhost），本次不缓存模型文件，功能不受影响', 'warn');
    }

    const gpu = await detectWebGPU();
    log(gpu.ok ? 'WebGPU 可用，优先使用 GPU 推理' : `WebGPU 不可用（${gpu.reason}），将降级 WASM`, gpu.ok ? 'ok' : 'warn');

    const candidates = buildCandidates(gpu);
    // 模型仓库按序尝试：官方优先，之后是常用镜像（国内网络下 huggingface.co 常不可达）。
    const plan = [];
    for (const cand of candidates) {
      for (const host of CFG.modelHosts) plan.push({ ...cand, host });
    }
    let lastError = null;

    for (const cand of plan) {
      const attempt = { ...cand, ok: false, error: null };
      try {
        env.remoteHost = cand.host;
        setProgress(true, 0, `加载模型（${cand.device} / ${cand.dtype} @ ${hostLabel(cand.host)}）…`, '');
        // 题目要求的显式配置：device + dtype 都由这里决定
        const extractor = await withTimeout(
          pipeline('feature-extraction', CFG.modelId, {
            device: cand.device,
            dtype: cand.dtype,
            progress_callback: (p) => {
              if (p && p.status === 'progress' && typeof p.progress === 'number') {
                const file = p.file ? String(p.file).split('/').pop() : 'model';
                setProgress(true, p.progress / 100, `下载 ${file}`, `${Math.round(p.progress)}%`);
              } else if (p && p.status === 'ready') {
                setProgress(true, 1, '模型就绪', '100%');
              }
            },
          }),
          CFG.modelLoadTimeoutMs,
          `加载模型（${cand.device}/${cand.dtype}）`
        );

        // 探针推理：确认管线真的能跑（防止 device 声称成功但首次前向失败）
        const probe = await extractor(['warmup'], { pooling: 'mean', normalize: true });
        const vec = typeof probe.tolist === 'function' ? probe.tolist()[0] : probe[0];
        if (!Array.isArray(vec) || vec.length !== CFG.dimension) {
          throw new Error(`探针向量维度异常：${Array.isArray(vec) ? vec.length : typeof vec}`);
        }

        state.extractor = extractor;
        state.device = cand.device;
        state.dtype = cand.dtype;
        state.modelLoadMs = performance.now() - started;
        attempt.ok = true;
        state.attempts.push(attempt);

        paintBackendPill();
        setProgress(false);
        const downgraded = cand.device !== 'webgpu' && gpu.ok;
        log(
          `模型就绪：${CFG.modelId} · device=${cand.device} · dtype=${cand.dtype} · ${CFG.dimension} 维 · ` +
            `${fmtMs(state.modelLoadMs)}` + (downgraded ? '（WebGPU 已尝试但失败，已降级）' : ''),
          'ok'
        );
        $('t-model').textContent = fmtMs(state.modelLoadMs);
        return { device: state.device, dtype: state.dtype };
      } catch (err) {
        lastError = err;
        attempt.error = err && err.message ? err.message : String(err);
        state.attempts.push(attempt);
        log(`后端 ${cand.device}/${cand.dtype} 不可用：${attempt.error}`, 'warn');
      }
    }

    state.device = null;
    state.dtype = null;
    $('pill-backend').classList.add('is-error');
    $('backend-value').textContent = '不可用';
    setProgress(false);
    throw new Error(
      `所有推理后端均失败：${lastError ? lastError.message || lastError : '未知错误'}`
    );
  })();

  // 用局部变量持住 Promise：避免失败路径里把 state.loadPromise 置空后
  // 这里 await 到 null（并发调用会提前返回 undefined）。
  const pending = state.loadPromise;
  try {
    return await pending;
  } catch (err) {
    state.loadPromise = null;
    throw err;
  }
}

/* ============================================================ 向量化 */

async function embedBatch(texts) {
  if (!state.extractor) throw new Error('模型尚未加载');
  const out = await withTimeout(
    state.extractor(texts, { pooling: 'mean', normalize: true }),
    CFG.embedTimeoutMs,
    `向量化 ${texts.length} 条`
  );
  const list = typeof out.tolist === 'function' ? out.tolist() : out;
  if (!Array.isArray(list) || list.length !== texts.length) {
    throw new Error(`向量化返回数量异常：期望 ${texts.length}，实际 ${Array.isArray(list) ? list.length : '非数组'}`);
  }
  return list.map((v) => Array.from(v));
}

/** 分批向量化，逐批回调进度。 */
async function embedAll(texts, onProgress) {
  const batchSize = readInt('embed-batch', CFG.defaultEmbedBatch, 1, 64);
  const vectors = [];
  for (let i = 0; i < texts.length; i += batchSize) {
    const slice = texts.slice(i, i + batchSize);
    const vecs = await embedBatch(slice);
    vectors.push(...vecs);
    if (onProgress) onProgress(Math.min(i + slice.length, texts.length), texts.length);
  }
  return vectors;
}

/* ============================================================ 集合管理 */

async function refreshCollections({ silent = false } = {}) {
  const select = $('collection-select');
  try {
    const data = await fetchJson('/api/collections', { timeoutMs: 15000 });
    const collections = (data && data.collections) || [];
    const names = collections.map((c) => c.name);

    if (names.length && !names.includes(state.collection)) {
      // 优先挑一个非默认且有数据的集合
      const withData = collections.find((c) => (c.count || 0) > 0);
      state.collection = withData ? withData.name : names[0];
      localStorage.setItem(STORAGE.collection, state.collection);
    }

    select.innerHTML = '';
    if (!names.length) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = '（暂无集合）';
      select.appendChild(opt);
      select.value = '';
    } else {
      for (const c of collections) {
        const opt = document.createElement('option');
        opt.value = c.name;
        opt.textContent = c.count == null ? c.name : `${c.name}（${c.count} 条）`;
        select.appendChild(opt);
      }
      if (names.includes(state.collection)) select.value = state.collection;
    }
    if (!silent) log(`集合列表已刷新：${names.length ? names.join(', ') : '空'}`, 'ok');
    return collections;
  } catch (err) {
    log(`获取集合列表失败：${err.message}`, 'err');
    if (!silent) throw err;
    return [];
  }
}

async function ensureCollectionExists(name) {
  // 集合由后端 getOrCreateCollection 惰性创建：健康检查 + 列表即可确认后端可用
  await fetchJson('/api/collections', { timeoutMs: 15000 });
  return name;
}

async function switchCollection(name, { announce = true } = {}) {
  if (!name) return;
  const previous = state.collection;
  state.collection = name;
  localStorage.setItem(STORAGE.collection, name);
  $('collection-select').value = name;

  // 切换集合必须让结果区同步失效，否则会展示上一个集合的命中结果
  if (previous !== name) {
    $('results').innerHTML = `<p class="empty">已切换到集合「${escapeHtml(name)}」，请重新检索。</p>`;
    $('t-qembed').textContent = '—';
    $('t-query').textContent = '—';
  }
  if (announce) log(`当前集合：${name}`, 'ok');
}

async function onResetCollection() {
  const name = state.collection;
  if (!name) {
    log('没有选中集合', 'warn');
    return;
  }
  if (!window.confirm(`确定重置集合「${name}」？将删除该集合内的全部文档。`)) return;
  setBusy(true, 'btn-reset-collection');
  try {
    const data = await fetchJson(`/api/collections/${encodeURIComponent(name)}/reset`, {
      method: 'POST',
      body: {},
      timeoutMs: 30000,
    });
    $('results').innerHTML = '<p class="empty">集合已重置，请重新导入语料。</p>';
    log(`集合 ${name} 已重置（${fmtMs(data.serverMs)}）`, 'ok');
    await refreshCollections({ silent: true });
  } catch (err) {
    log(`重置失败：${err.message}`, 'err');
  } finally {
    setBusy(false, 'btn-reset-collection');
  }
}

/* ============================================================ 导入 */

/** 把原始文本切成文档：空行分段，超长再按长度硬切。 */
function splitIntoDocs(text, sourceLabel) {
  const blocks = String(text)
    .replace(/\r\n/g, '\n')
    .split(/\n\s*\n/)
    .map((s) => s.trim())
    .filter(Boolean);

  const docs = [];
  for (const block of blocks) {
    if (block.length <= CFG.maxDocChars) {
      docs.push(block);
    } else {
      for (let i = 0; i < block.length; i += CFG.maxDocChars) {
        docs.push(block.slice(i, i + CFG.maxDocChars));
      }
    }
  }
  return docs.map((text, i) => {
    const firstLine = text.split('\n')[0].trim();
    const title = firstLine.length > 0 && firstLine.length <= 80 ? firstLine : `${sourceLabel} #${i + 1}`;
    return { text, title, source: sourceLabel };
  });
}

/**
 * 文档身份 = 内容哈希（FNV-1a + 长度）。
 * 同一段文本重复导入得到同一个 id，配合服务端 upsert 就变成幂等更新，
 * 这是「增量写入 + 文档身份」能成立的前提。
 */
function makeDocId(text) {
  const value = String(text ?? '');
  let hash = 0x811c9dc5;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return `doc-${hash.toString(16).padStart(8, '0')}-${value.length.toString(16)}`;
}

async function collectImportItems() {
  const items = [];

  // 1) 文本框
  const pasted = $('corpus-text').value.trim();
  if (pasted) {
    items.push(...splitIntoDocs(pasted, '粘贴文本'));
  }

  // 2) 上传文件
  const files = Array.from($('corpus-file').files || []);
  for (const file of files) {
    if (!/\.(txt|md|markdown)$/i.test(file.name)) {
      log(`跳过不支持的文件类型：${file.name}`, 'warn');
      continue;
    }
    if (file.size > 5 * 1024 * 1024) {
      log(`跳过过大的文件（>5MB）：${file.name}`, 'warn');
      continue;
    }
    const text = await file.text();
    items.push(...splitIntoDocs(text, file.name));
  }

  return items;
}

async function runImport() {
  setBusy(true, 'btn-import', 'btn-sample');
  const t0 = performance.now();
  try {
    const items = await collectImportItems();
    if (items.length === 0) {
      log('没有可导入的内容：请粘贴文本或选择 .txt/.md 文件', 'warn');
      return;
    }
    log(`解析出 ${items.length} 篇文档，开始向量化…`);

    // ---- 阶段 1：模型（可能触发下载）
    const hadModel = Boolean(state.extractor);
    const loadStart = performance.now();
    await ensureModel();
    const loadMs = hadModel ? 0 : performance.now() - loadStart;

    // ---- 阶段 2：浏览器内向量化
    const embedStart = performance.now();
    setProgress(true, 0, '向量化中…', `0 / ${items.length}`);
    const vectors = await embedAll(
      items.map((it) => it.text),
      (done, total) => {
        setProgress(true, done / total, '向量化中…', `${done} / ${total}`);
        $('progress-label').textContent = `向量化中（后端 ${state.device}）…`;
      }
    );
    const embedMs = performance.now() - embedStart;
    setProgress(true, 1, '向量化完成，写入 seekdb…', `${items.length} / ${items.length}`);
    log(`向量化完成：${items.length} 条 / ${fmtMs(embedMs)}`, 'ok');

    // ---- 阶段 3：分批写入 seekdb
    const writeStart = performance.now();
    const writeBatch = readInt('write-batch', CFG.defaultWriteBatch, 1, 256);
    const collectionName = state.collection;
    let inserted = 0;

    for (let i = 0; i < items.length; i += writeBatch) {
      const slice = items.slice(i, i + writeBatch);
      const payload = {
        ids: slice.map((it) => makeDocId(it.text)),
        documents: slice.map((it) => it.text),
        embeddings: vectors.slice(i, i + slice.length),
        metadatas: slice.map((it) => ({
          doc_id: it.title,
          title: it.title,
          source: it.source,
          imported_at: new Date().toISOString(),
        })),
      };
      const res = await fetchJson(`/api/collections/${encodeURIComponent(collectionName)}/documents`, {
        method: 'POST',
        body: payload,
        timeoutMs: 60000,
      });
      inserted += (res && res.inserted) || slice.length;
      setProgress(true, (i + slice.length) / items.length, '写入 seekdb…', `${inserted} / ${items.length}`);
    }

    const writeMs = performance.now() - writeStart;
    const totalMs = performance.now() - t0;

    $('t-embed').textContent = fmtMs(embedMs);
    $('t-write').textContent = `${fmtMs(writeMs)}（${inserted} 条）`;
    $('t-total').textContent = fmtMs(totalMs);
    $('t-model').textContent = hadModel ? '已缓存' : fmtMs(loadMs);

    log(`写入完成：${inserted} 条 → 集合 ${collectionName}（${fmtMs(writeMs)}）`, 'ok');
    setProgress(false);
    await refreshCollections({ silent: true });
  } catch (err) {
    setProgress(false);
    log(`导入失败：${err.message}`, 'err');
    console.error(err);
  } finally {
    setBusy(false, 'btn-import', 'btn-sample');
  }
}

async function loadSampleCorpus() {
  try {
    log('正在拉取示例语料 samples/corpus.jsonl…');
    const res = await fetch('./samples/corpus.jsonl', { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const text = await res.text();
    const docs = text
      .split('\n')
      .map((l) => l.trim())
      .filter(Boolean)
      .map((l) => JSON.parse(l));

    const select = $('collection-select');
    if (!select.value) {
      // 还没有集合：先用默认名占位，后端会在首次写入时创建
      await switchCollection(state.collection || 'kb_default', { announce: false });
    }

    $('corpus-text').value = docs
      .map((d) => `${d.title}\n${d.text}`)
      .join('\n\n');
    log(`已载入 ${docs.length} 条示例文档到输入框，点击「向量化并导入」写入 seekdb`, 'ok');
  } catch (err) {
    log(`加载示例语料失败：${err.message}`, 'err');
  }
}

/* ============================================================ 检索 */

function highlight(text, query) {
  const safe = escapeHtml(text);
  const terms = Array.from(
    new Set(
      String(query)
        .split(/[\s,，。；;、/]+/)
        .map((t) => t.trim())
        .filter((t) => t.length >= 2)
    )
  ).slice(0, 8);
  if (!terms.length) return safe;
  let out = safe;
  for (const term of terms) {
    const safeTerm = escapeHtml(term).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    out = out.replaceAll(new RegExp(safeTerm, 'gi'), (m) => `<mark>${m}</mark>`);
  }
  return out;
}

function renderResults(data, query) {
  const box = $('results');
  const results = (data && data.results) || [];
  if (!results.length) {
    box.innerHTML = '<p class="empty">没有命中结果。可尝试换关键词，或确认当前集合已导入文档。</p>';
    return;
  }

  box.innerHTML = '';
  results.forEach((r) => {
    const docText = r.document || '';
    const snippet = docText.length > CFG.snippetChars ? `${docText.slice(0, CFG.snippetChars)}…` : docText;
    const metaId = (r.metadata && (r.metadata.doc_id || r.metadata.title)) || '';
    const distance = typeof r.distance === 'number' ? r.distance.toFixed(4) : '—';
    // cosine 距离换算成相似度（仅作展示，排序仍由 seekdb 决定）
    const similarity =
      typeof r.distance === 'number' ? (1 - r.distance).toFixed(4) : '—';

    const card = document.createElement('article');
    card.className = 'result';
    card.innerHTML = `
      <div class="result-head">
        <span class="rank">#${r.rank}</span>
        <span class="doc-id">${escapeHtml(r.id || '')}</span>
        ${metaId ? `<span class="doc-id">· ${escapeHtml(metaId)}</span>` : ''}
        <span class="score">sim ${similarity} · dist ${distance}</span>
      </div>
      <p class="snippet">${highlight(snippet, query)}</p>
      <details>
        <summary>查看原文（${docText.length} 字符）</summary>
        <pre class="fulltext">${escapeHtml(docText)}</pre>
      </details>
    `;
    box.appendChild(card);
  });
}

async function runSearch() {
  const query = $('query-input').value.trim();
  if (!query) {
    log('请输入查询内容', 'warn');
    return;
  }
  const collectionName = state.collection;
  if (!collectionName) {
    log('请先选择或新建集合', 'warn');
    return;
  }

  setBusy(true, 'btn-search');
  try {
    await ensureModel();

    // ---- 查询向量在浏览器内计算
    const qEmbedStart = performance.now();
    const [queryVector] = await embedBatch([query]);
    const qEmbedMs = performance.now() - qEmbedStart;
    $('t-qembed').textContent = fmtMs(qEmbedMs);

    // ---- 交给 seekdb 做混合检索（RRF 在库内完成）
    // 全文分支用去停用词的关键词串：整句会被当成近似短语，几乎匹配不到任何文档，
    // 结果就是 RRF 里只剩向量一路在排序。向量分支仍然用原始问题。
    const nResults = readInt('n-results', 10, 1, 50);
    const keywordQuery = buildKeywordQuery(query);
    const searchStart = performance.now();
    const data = await fetchJson(`/api/collections/${encodeURIComponent(collectionName)}/search`, {
      method: 'POST',
      body: { queryVector, queryText: keywordQuery, nResults },
      timeoutMs: 60000,
    });
    const clientMs = performance.now() - searchStart;
    $('t-query').textContent = `${fmtMs(data.serverMs)}（服务端）/ ${fmtMs(clientMs)}（含网络）`;

    renderResults(data, query);
    log(
      `「${query}」命中 ${(data.results || []).length} 条 · 集合 ${collectionName} · ` +
        `seekdb ${fmtMs(data.serverMs)}（候选窗口 ${data.candidateWindow}）`,
      'ok'
    );
  } catch (err) {
    log(`检索失败：${err.message}`, 'err');
    console.error(err);
  } finally {
    setBusy(false, 'btn-search');
  }
}

/* ============================================================ 后端健康检查 */

async function checkHealth() {
  try {
    const data = await fetchJson('/api/health', { timeoutMs: 15000 });
    const db = data.db || {};
    const cfg = db.config || {};
    // `ok` 是顶层字段（见 server/index.mjs 的 /api/health），db.ok 恒为 undefined。
    const healthy = data.ok === true;
    const where = cfg.mode === 'embedded' ? cfg.path : `${cfg.host}:${cfg.port}`;
    $('db-value').textContent = healthy ? `${db.mode} ✓` : `${db.mode} ✗`;
    $('pill-db').title = healthy
      ? `seekdb ${db.mode} @ ${where} · ${db.latencyMs}ms`
      : `seekdb 不可用：${db.error || '未知错误'}`;
    log(
      healthy
        ? `seekdb 连接正常：mode=${db.mode} @ ${where} · ${db.latencyMs}ms`
        : `seekdb 连接失败：${db.error || '未知错误'}`,
      healthy ? 'ok' : 'err'
    );
  } catch (err) {
    $('db-value').textContent = '后端离线';
    $('pill-db').title = err.message;
    log(`健康检查失败：${err.message}（请确认 npm start 已运行）`, 'err');
  }
}

/* ============================================================ 量化对比评测 */

/**
 * 量化对比评测：真正的工作在 compare.js，这里只负责取当前语料、按钮态与日志。
 * 结果表与错误块都由 compare.js 直接写进 #compare-output；
 * 它对外不抛异常，所以按钮回调不会产生未捕获的 promise rejection。
 */
async function runCompare() {
  setBusy(true, 'btn-compare');
  const output = $('compare-output');
  if (output) {
    output.innerHTML = '<p class="empty">评测进行中：两种配置都要下载模型并向量化全量语料，首次可能持续数分钟…</p>';
  }
  const pasted = $('corpus-text').value.trim();
  log(
    `量化对比评测开始：q8 vs fp32，语料来自${pasted ? '页面「粘贴文本」' : 'samples/corpus.jsonl'}，` +
      '查询来自 samples/queries.jsonl'
  );

  try {
    const result = await runQuantizationComparison({
      modelId: CFG.modelId,
      modelHosts: CFG.modelHosts,
      transformersUrl: CFG.transformersUrl,
      dimension: CFG.dimension,
      // 与导入流程用同一个 batch 输入，保证「向量化耗时」可比
      batch: readInt('embed-batch', CFG.defaultEmbedBatch, 1, 64),
      // 复用页面上已加载的 Transformers.js 模块与已切好的文档，避免重复 import / 重新切分
      transformers: state.transformers,
      docs: pasted ? splitIntoDocs(pasted, '页面文本') : null,
      log,
      setProgress,
    });

    if (result.ok) {
      const summary = result.rows
        .map((row) => {
          const r5 = row.recall5 == null ? '—' : `${(row.recall5 * 100).toFixed(1)}%`;
          const r10 = row.recall10 == null ? '—' : `${(row.recall10 * 100).toFixed(1)}%`;
          const mrr = row.mrr10 == null ? '—' : row.mrr10.toFixed(3);
          return `${row.label} ${row.device}/${row.dtype} R@5 ${r5} · R@10 ${r10} · MRR ${mrr}`;
        })
        .join('；');
      log(`量化对比评测完成（${fmtMs(result.elapsedMs)}）：${summary}`, 'ok');
      log('提示：表中质量指标由浏览器内余弦排序得出，仅用于量化影响评估；主流程检索仍走 seekdb 混合检索。');
    } else {
      log(`${result.errorKind === 'model-download' ? '模型下载失败' : '评测失败'}：${result.message}`, 'err');
    }
  } catch (err) {
    // 兜底：compare.js 已内建错误处理，这里只防止回调本身被打穿
    log(`量化对比评测异常：${err.message}`, 'err');
    console.error(err);
  } finally {
    setBusy(false, 'btn-compare');
    setProgress(false);
  }
}

/* ============================================================ 启动绑定 */

function bindEvents() {
  $('btn-import').addEventListener('click', runImport);
  $('btn-sample').addEventListener('click', loadSampleCorpus);
  $('btn-clear').addEventListener('click', () => {
    $('corpus-text').value = '';
    $('corpus-file').value = '';
  });

  $('btn-search').addEventListener('click', runSearch);
  $('query-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') runSearch();
  });

  $('collection-select').addEventListener('change', (e) => {
    const name = e.target.value;
    if (name) switchCollection(name);
  });
  $('btn-refresh-collections').addEventListener('click', () => refreshCollections());
  $('btn-reset-collection').addEventListener('click', onResetCollection);

  $('btn-compare').addEventListener('click', runCompare);

  $('btn-create-collection').addEventListener('click', async () => {
    const name = $('new-collection').value.trim();
    if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$/.test(name)) {
      log('集合名非法：仅允许字母数字开头，1-63 位 [A-Za-z0-9_-]', 'warn');
      return;
    }
    try {
      await ensureCollectionExists(name);
      await refreshCollections({ silent: true });
      await switchCollection(name);
      $('new-collection').value = '';
      $('results').innerHTML = '<p class="empty">已切换集合，请导入语料或直接检索。</p>';
    } catch (err) {
      log(`切换集合失败：${err.message}`, 'err');
    }
  });

  $('btn-clear-log').addEventListener('click', () => {
    $('log').textContent = '';
  });

  // 记住 batch 设置
  for (const key of ['embed-batch', 'write-batch']) {
    const el = $(key);
    const saved = localStorage.getItem(`edgekb.${key}`);
    if (saved) el.value = saved;
    el.addEventListener('change', () => localStorage.setItem(`edgekb.${key}`, el.value));
  }
}

async function main() {
  $('app-version').textContent = 'edge-kb v1.0.0';
  paintBackendPill();
  bindEvents();
  await checkHealth();
  await refreshCollections({ silent: true });
  if (state.collection) {
    $('collection-select').value = state.collection;
    log(`当前集合：${state.collection}`, 'ok');
  }
  log('提示：首次点击「向量化并导入」会下载模型（约 20–90MB，取决于量化精度），请保持网络畅通。');
}

main();
