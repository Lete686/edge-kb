// Minimal seekdb smoke test: server/embedded client, explicit schema, add, hybridSearch.
// Run: SEEKDB_MODE=server SEEKDB_HOST=127.0.0.1 SEEKDB_PORT=2881 node scripts/smoke-seekdb.mjs
//
// Why an explicit Schema: the short `configuration` form falls back to the default embedding
// function, which needs `@seekdb/default-embed`. This app computes every vector in the browser,
// so the collection must declare `embeddingFunction: null` and add a full-text index itself.

import { FulltextIndexConfig, Schema, SeekdbClient, VectorIndexConfig } from "seekdb";

const DIMENSION = 384;
const COLLECTION = "smoke_hybrid";

function vector(seed) {
  let state = seed;
  return Array.from({ length: DIMENSION }, () => {
    state = (state * 1103515245 + 12345) % 2147483648;
    return state / 2147483648 - 0.5;
  });
}

function log(step, detail) {
  console.log(`[${step}] ${detail}`);
}

function buildSchema() {
  return new Schema({
    fulltextIndex: new FulltextIndexConfig("ik", { ik_mode: "smart" }),
    vectorIndex: new VectorIndexConfig({
      hnsw: { dimension: DIMENSION, distance: "cosine" },
      embeddingFunction: null,
    }),
  });
}

async function main() {
  const mode = process.env.SEEKDB_MODE ?? "server";
  const client =
    mode === "server"
      ? new SeekdbClient({
          host: process.env.SEEKDB_HOST ?? "127.0.0.1",
          port: Number(process.env.SEEKDB_PORT ?? 2881),
          database: process.env.SEEKDB_DATABASE ?? "test",
          user: process.env.SEEKDB_USER ?? "root",
          password: process.env.SEEKDB_PASSWORD ?? "",
        })
      : new SeekdbClient({
          path: process.env.SEEKDB_PATH ?? "./data/smoke",
          database: process.env.SEEKDB_DATABASE ?? "test",
        });

  log("connect", `mode=${mode}`);
  await client.listCollections();
  log("connect", "ok");

  await client.deleteCollection(COLLECTION).catch(() => {});
  const collection = await client.createCollection({ name: COLLECTION, schema: buildSchema() });
  log("collection", `created name=${collection.name} (schema: fulltext ik + hnsw, embeddingFunction=null)`);

  const ids = ["doc-apple", "doc-banana", "doc-cherry"];
  const documents = [
    "apple orchard harvest notes for the autumn season",
    "banana shipment arrived with a delayed schedule",
    "cherry pie recipe with a buttery crust",
  ];
  const metadatas = [{ doc: "apple.txt" }, { doc: "banana.txt" }, { doc: "cherry.txt" }];

  await collection.add({
    ids,
    documents,
    metadatas,
    embeddings: ids.map((_id, index) => vector(index + 1)),
  });
  log("add", `rows=${ids.length} count=${await collection.count()}`);

  const result = await collection.hybridSearch({
    query: { whereDocument: { $contains: "apple" }, nResults: 5 },
    knn: { queryEmbeddings: [vector(1)], nResults: 5 },
    rank: { rrf: {} },
    nResults: 3,
    include: ["documents", "metadatas", "distances"],
  });

  const hitIds = result.ids?.[0] ?? [];
  log("hybridSearch", `returned=${hitIds.length} ids=${hitIds.join(",")}`);
  if (hitIds.length === 0) {
    throw new Error("hybridSearch returned no rows; the full-text branch or RRF ranker is not working");
  }
  log("documents", JSON.stringify(result.documents?.[0] ?? []));

  await collection.delete({ ids }).catch(() => {});
  await client.close();
  log("done", "SMOKE_OK");
}

main().catch((error) => {
  console.error("SMOKE_FAILED:", error?.stack ?? error);
  const message = String(error?.message ?? error);
  if (/fulltext|full text|FULLTEXT|no such index|unknown index/i.test(message)) {
    console.error("HINT: the full-text index is missing; check FulltextIndexConfig support on this server version.");
  }
  if (/embedding function/i.test(message)) {
    console.error("HINT: a collection still falls back to the default embedder; pass embeddingFunction: null in VectorIndexConfig.");
  }
  process.exit(1);
});
