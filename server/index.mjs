/**
 * Express 后端：只做「接收浏览器算好的向量 → 写入 seekdb / 用 seekdb 做混合检索」。
 *
 * 本文件刻意不引入任何 embedding / 向量运算库：所有向量都来自请求体，
 * 服务端只做长度校验与转发，满足「后端绝不计算向量」。
 *
 * 端点：
 *   POST /api/collections/:name/documents   写入（增量 add）
 *   POST /api/collections/:name/search      向量 + 全文混合检索（seekdb hybridSearch）
 *   GET  /api/collections                   集合列表
 *   POST /api/collections/:name/reset       重置（删除重建）集合
 *   GET  /api/health                        健康检查 + 当前连接模式
 *   静态托管 public/ 与 samples/
 */
import express from 'express';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  EMBEDDING_DIMENSION,
  assertCollectionName,
  countCollection,
  describeConfig,
  getCollection,
  health,
  listCollections,
  resetCollection,
  withTimeout,
  closeClient,
} from './seekdb.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const APP_ROOT = path.resolve(__dirname, '..');
const PUBLIC_DIR = process.env.PUBLIC_DIR
  ? path.resolve(process.env.PUBLIC_DIR)
  : path.join(APP_ROOT, 'public');
const SAMPLES_DIR = process.env.SAMPLES_DIR
  ? path.resolve(process.env.SAMPLES_DIR)
  : path.join(APP_ROOT, 'samples');

const PORT = Number(process.env.PORT || 3000);
const HOST = process.env.HOST || '0.0.0.0';
const WRITE_TIMEOUT_MS = Number(process.env.WRITE_TIMEOUT_MS || 30000);
const SEARCH_TIMEOUT_MS = Number(process.env.SEARCH_TIMEOUT_MS || 30000);
const MAX_N_RESULTS = 50;
const MAX_DOCS_PER_REQUEST = Number(process.env.MAX_DOCS_PER_REQUEST || 512);

const app = express();
app.disable('x-powered-by');
// 向量是 384 维浮点数组，批量导入时请求体较大，放宽到 64MB。
app.use(express.json({ limit: '64mb' }));

/** 统一错误：带 status 的直接返回，其余 500。 */
function httpError(status, message, extra) {
  const err = new Error(message);
  err.status = status;
  if (extra) Object.assign(err, extra);
  return err;
}

/** 包一层 async 路由，异常统一进 error middleware。 */
const wrap = (fn) => (req, res, next) => Promise.resolve(fn(req, res, next)).catch(next);

/** 把可能是 [[...]] 的字段拍平成一维。seekdb 返回形如 { ids: [[...]] }。 */
function flatten(value) {
  if (value == null) return [];
  if (!Array.isArray(value)) return [value];
  if (value.length === 0) return [];
  if (Array.isArray(value[0])) return flatten(value[0]); // 单查询只取第一行
  return value;
}

/** OpenAI 风格向量做了 float32 传输时可能带 NaN/Infinity，写库前挡掉。 */
function assertVector(vec, index, field) {
  if (!Array.isArray(vec)) {
    throw httpError(400, `${field}[${index}] 不是数组`);
  }
  if (vec.length !== EMBEDDING_DIMENSION) {
    throw httpError(
      400,
      `${field}[${index}] 维度为 ${vec.length}，期望 ${EMBEDDING_DIMENSION}（请确认浏览器端模型为 all-MiniLM-L6-v2）`
    );
  }
  for (const v of vec) {
    if (typeof v !== 'number' || !Number.isFinite(v)) {
      throw httpError(400, `${field}[${index}] 含非有限数值，请检查向量化结果`);
    }
  }
  return vec;
}

/** metadata 只允许标量，避免 seekdb 侧序列化失败。 */
function sanitizeMetadata(meta, index) {
  if (meta == null) return {};
  if (typeof meta !== 'object' || Array.isArray(meta)) {
    throw httpError(400, `metadatas[${index}] 必须是对象`);
  }
  const out = {};
  for (const [k, v] of Object.entries(meta)) {
    if (v == null) continue;
    if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') {
      out[k] = v;
    } else if (Array.isArray(v) && v.every((x) => ['string', 'number', 'boolean'].includes(typeof x))) {
      out[k] = v.map((x) => String(x));
    } else {
      out[k] = JSON.stringify(v);
    }
  }
  return out;
}

/* ---------------------------------------------------------------- 健康检查 */

app.get(
  '/api/health',
  wrap(async (_req, res) => {
    const info = await health();
    res.status(info.ok ? 200 : 503).json({
      ok: info.ok,
      db: {
        mode: info.mode,
        config: info.config,
        latencyMs: info.latencyMs,
        collections: info.collections,
        error: info.error,
      },
      embedding: { dimension: EMBEDDING_DIMENSION, computedIn: 'browser' },
      node: process.version,
      uptimeSec: Math.round(process.uptime()),
    });
  })
);

/* ---------------------------------------------------------------- 集合列表 */

app.get(
  '/api/collections',
  wrap(async (_req, res) => {
    const names = await listCollections();
    const collections = await Promise.all(
      names.map(async (name) => ({ name, count: await countCollection(name) }))
    );
    res.json({ collections, mode: describeConfig().mode });
  })
);

/* ------------------------------------------------------------------ 写入 */

app.post(
  '/api/collections/:name/documents',
  wrap(async (req, res) => {
    const name = assertCollectionName(req.params.name);
    const { ids, documents, embeddings, metadatas } = req.body || {};

    if (!Array.isArray(ids) || ids.length === 0) {
      throw httpError(400, 'ids 必须是非空数组');
    }
    if (!Array.isArray(embeddings)) {
      throw httpError(400, 'embeddings 必须是数组（由浏览器端计算，后端不代算）');
    }
    if (ids.length !== embeddings.length) {
      throw httpError(400, `ids=${ids.length} 与 embeddings=${embeddings.length} 数量不一致`);
    }
    if (ids.length > MAX_DOCS_PER_REQUEST) {
      throw httpError(413, `单次最多写入 ${MAX_DOCS_PER_REQUEST} 条，请在前端分批`);
    }
    const docs = Array.isArray(documents) ? documents : [];
    const metas = Array.isArray(metadatas) ? metadatas : [];
    if (docs.length && docs.length !== ids.length) {
      throw httpError(400, `documents=${docs.length} 与 ids=${ids.length} 数量不一致`);
    }
    if (metas.length && metas.length !== ids.length) {
      throw httpError(400, `metadatas=${metas.length} 与 ids=${ids.length} 数量不一致`);
    }
    embeddings.forEach((v, i) => assertVector(v, i, 'embeddings'));

    const payload = {
      ids: ids.map((id) => String(id)),
      embeddings: embeddings.map((v) => Array.from(v)),
    };
    if (docs.length) payload.documents = docs.map((d) => String(d));
    if (metas.length) payload.metadatas = metas.map((m, i) => sanitizeMetadata(m, i));

    const started = Date.now();
    const collection = await getCollection(name);
    // 用 upsert 而不是 add：文档 id 由前端按内容哈希生成，重复导入应更新同一行而不是新增副本。
    const write =
      typeof collection.upsert === 'function'
        ? collection.upsert(payload)
        : collection.add(payload);
    await withTimeout(write, WRITE_TIMEOUT_MS, `write(${name})`);
    const took = Date.now() - started;

    res.json({
      ok: true,
      collection: name,
      inserted: payload.ids.length,
      serverMs: took,
      count: await countCollection(name),
    });
  })
);

/* -------------------------------------------------------------- 混合检索 */

app.post(
  '/api/collections/:name/search',
  wrap(async (req, res) => {
    const name = assertCollectionName(req.params.name);
    const body = req.body || {};
    const { queryVector, queryText } = body;
    const nResultsRaw = body.nResults;

    if (!Array.isArray(queryVector)) {
      throw httpError(400, 'queryVector 必须是数组（由浏览器端计算，后端不代算）');
    }
    assertVector(queryVector, 0, 'queryVector');
    if (typeof queryText !== 'string' || queryText.trim() === '') {
      throw httpError(400, 'queryText 必须是非空字符串（全文检索分支需要关键词）');
    }
    const nResults = Number.isFinite(Number(nResultsRaw))
      ? Math.min(MAX_N_RESULTS, Math.max(1, Math.trunc(Number(nResultsRaw))))
      : 10;

    // RRF 需要两路各召回足够多的候选，窗口比 nResults 宽一些。
    const candidateWindow = Math.min(100, Math.max(nResults * 3, nResults));

    const collection = await getCollection(name);

    const started = Date.now();
    // ⚠️ 混合检索完全在 seekdb 内完成：向量分支 + 全文章节由 RRF 融合。
    //    客户端不会分别取两路结果再自行合并。
    const raw = await withTimeout(
      collection.hybridSearch({
        query: {
          whereDocument: { $contains: queryText.trim() },
          nResults: candidateWindow,
        },
        knn: {
          queryEmbeddings: [Array.from(queryVector)],
          nResults: candidateWindow,
        },
        rank: { rrf: {} },
        nResults,
        include: ['documents', 'metadatas', 'distances'],
      }),
      SEARCH_TIMEOUT_MS,
      `hybridSearch(${name})`
    );
    const serverMs = Date.now() - started;

    const ids = flatten(raw && raw.ids);
    const docs = flatten(raw && raw.documents);
    const metas = flatten((raw && raw.metadatas) || []);
    const dists = flatten((raw && raw.distances) || []);

    const results = ids.map((id, i) => ({
      id: id == null ? null : String(id),
      document: docs[i] == null ? '' : String(docs[i]),
      metadata: (metas[i] && typeof metas[i] === 'object') ? metas[i] : {},
      distance: typeof dists[i] === 'number' ? dists[i] : null,
      rank: i + 1,
    }));

    res.json({
      ok: true,
      collection: name,
      queryText: queryText.trim(),
      nResults,
      candidateWindow,
      serverMs,
      results,
    });
  })
);

/* ------------------------------------------------------------------ 重置 */

app.post(
  '/api/collections/:name/reset',
  wrap(async (req, res) => {
    const name = assertCollectionName(req.params.name);
    const started = Date.now();
    const info = await resetCollection(name);
    res.json({
      ok: true,
      collection: name,
      recreated: Boolean(info.collection),
      existed: info.deleted,
      serverMs: Date.now() - started,
    });
  })
);

/* ------------------------------------------------------------ 静态资源与兜底 */

// samples/ 只读挂载，方便前端「加载示例语料」按钮直接拉取。
app.use('/samples', express.static(SAMPLES_DIR, { fallthrough: true }));
app.use(express.static(PUBLIC_DIR, { extensions: ['html'] }));

app.use('/api', (_req, res) => {
  res.status(404).json({ ok: false, error: '接口不存在' });
});

// 非 API 的未知路径回到单页
app.get('*', (_req, res, next) => {
  res.sendFile(path.join(PUBLIC_DIR, 'index.html'), (err) => (err ? next(err) : undefined));
});

// eslint-disable-next-line no-unused-vars
app.use((err, _req, res, _next) => {
  const status = err.status || (err.type === 'entity.too.large' ? 413 : 500);
  if (status >= 500) console.error('[edge-kb]', err);
  res.status(status).json({
    ok: false,
    error: err.message || '服务器内部错误',
    code: err.code || null,
  });
});

const server = app.listen(PORT, HOST, () => {
  const cfg = describeConfig();
  console.log(`[edge-kb] listening on http://127.0.0.1:${PORT}`);
  console.log(
    `[edge-kb] seekdb mode=${cfg.mode} ` +
      (cfg.mode === 'embedded' ? `path=${cfg.path}` : `host=${cfg.host}:${cfg.port}`) +
      ` database=${cfg.database}`
  );
  console.log(`[edge-kb] embeddings: browser-side only, dim=${EMBEDDING_DIMENSION}`);
});

/** 优雅关闭：断开 seekdb 连接，避免嵌入式模式留下锁文件。 */
async function shutdown(signal) {
  console.log(`[edge-kb] ${signal} received, shutting down...`);
  server.close();
  await closeClient();
  process.exit(0);
}
process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));

export { app, server };
