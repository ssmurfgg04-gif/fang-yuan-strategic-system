-- ============================================================================
-- Fang Yuan Strategic System — Master Database Schema (SQLite 3.44+, FTS5)
-- Layered knowledge: provenance-gated corpus, structured canon, decision
-- records, quotes, classics, policy lessons, research findings, benchmarks,
-- simulator runs, engineering test log.
-- Every ingestible row carries provenance via sources.id.
-- ============================================================================

PRAGMA journal_mode = WAL;

-- ---------------------------------------------------------------- provenance
CREATE TABLE IF NOT EXISTS sources (
  id                 INTEGER PRIMARY KEY,
  source_key         TEXT UNIQUE,              -- stable key e.g. 'fandom:fang_yuan'
  kind               TEXT NOT NULL,            -- fandom_wiki | public_domain_classic | uploaded_notes | user_provided_corpus | research_search | analyst_generated | web_article
  title              TEXT,
  url                TEXT,
  license            TEXT,                     -- 'CC BY-SA 3.0' | 'Public domain' | 'User responsibility' | 'n/a'
  provenance_status  TEXT NOT NULL DEFAULT 'lawful_verified'
                     -- lawful_verified | user_responsibility | quarantined | analyst_generated
                     -- quarantined = file present but provenance unresolved; MUST NOT be used for training until resolved
  ,fetched_at        TEXT,
  meta_json          TEXT
);

-- ------------------------------------------------------------ raw text layer
CREATE TABLE IF NOT EXISTS corpus_items (
  id          INTEGER PRIMARY KEY,
  source_id   INTEGER NOT NULL REFERENCES sources(id),
  item_type   TEXT NOT NULL,       -- wiki_article | classic_text | chapter | note_chunk | quote_passage
  title       TEXT,
  lang        TEXT DEFAULT 'en',   -- zh | en
  text        TEXT NOT NULL,
  meta_json   TEXT
);
CREATE INDEX IF NOT EXISTS idx_corpus_type ON corpus_items(item_type);

-- ----------------------------------------------------- Layer A: canon graph
CREATE TABLE IF NOT EXISTS characters (
  id           INTEGER PRIMARY KEY,
  source_id    INTEGER REFERENCES sources(id),
  name         TEXT NOT NULL,
  aliases_json TEXT,
  faction      TEXT,
  region       TEXT,
  role         TEXT,               -- protagonist | rival | ally | antagonist | minor
  arc_span     TEXT,               -- e.g. 'V1-V5'
  summary      TEXT,
  confidence   TEXT DEFAULT 'medium',  -- high | medium | low (low = needs verification)
  meta_json    TEXT
);

CREATE TABLE IF NOT EXISTS factions (
  id         INTEGER PRIMARY KEY,
  source_id  INTEGER REFERENCES sources(id),
  name       TEXT NOT NULL,
  region     TEXT,
  kind       TEXT,                 -- clan | sect | super_force | tribe | merchant
  summary    TEXT,
  confidence TEXT DEFAULT 'medium'
);

CREATE TABLE IF NOT EXISTS gu_worms (
  id         INTEGER PRIMARY KEY,
  source_id  INTEGER REFERENCES sources(id),
  name       TEXT NOT NULL,
  rank       TEXT,                 -- Rank 1-9 / Immortal Gu
  path       TEXT,                -- refinement | blood | wisdom | strength | transformation | rule | luck | info | ...
  owner      TEXT,
  summary    TEXT,
  confidence TEXT DEFAULT 'medium'
);

CREATE TABLE IF NOT EXISTS locations (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  region TEXT,
  summary TEXT,
  confidence TEXT DEFAULT 'medium'
);

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY,
  source_id INTEGER REFERENCES sources(id),
  title TEXT NOT NULL,
  arc TEXT,                        -- qing_mao_mountain | caravan | shang_clan | northern_plains | ...
  volume INTEGER,
  chapter_range TEXT,
  summary TEXT,
  participants_json TEXT,
  confidence TEXT DEFAULT 'medium'
);

-- ------------------------------------------------- Layer B: decision records
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY,
  event_id INTEGER REFERENCES events(id),
  source_id INTEGER REFERENCES sources(id),
  record_class TEXT NOT NULL DEFAULT 'major'
      -- major | tactical | transaction | failure | mask   (mask = benevolence/weakness as strategy)
  ,situation TEXT NOT NULL,
  objective TEXT,
  known_facts_json TEXT,
  unknowns_json TEXT,
  resources_json TEXT,
  options_json TEXT,
  selected_action TEXT,
  rejected_options_json TEXT,      -- [{action, why_rejected}]
  cost TEXT,
  result_json TEXT,
  lesson_json TEXT,
  canon_anchor TEXT,
  volume TEXT,
  confidence TEXT DEFAULT 'medium'
);
CREATE INDEX IF NOT EXISTS idx_decisions_class ON decisions(record_class);

-- ----------------------------------------------------- Layer C: quote bank
CREATE TABLE IF NOT EXISTS quotes (
  id INTEGER PRIMARY KEY,
  source_id INTEGER REFERENCES sources(id),
  text_en TEXT,
  text_zh TEXT,
  pinyin TEXT,
  translation_kind TEXT,           -- fan_translation | literal | polished | paraphrase
  speaker TEXT,
  chapter_ref TEXT,
  scene TEXT,
  theme TEXT,
  verification_status TEXT NOT NULL DEFAULT 'unverified'
      -- verified | unverified | fandom_paraphrase | needs_chinese_source
      -- RULE: model must never assert unverified Chinese wording as canon.
  ,variants_json TEXT,
  confidence TEXT DEFAULT 'low'
);
CREATE INDEX IF NOT EXISTS idx_quotes_status ON quotes(verification_status);

-- --------------------------- Layer D: Chinese strategic tradition (separate!)
CREATE TABLE IF NOT EXISTS classics (
  id INTEGER PRIMARY KEY,
  source_id INTEGER REFERENCES sources(id),
  work TEXT NOT NULL,              -- daodejing | sunzi | hanfeizi | guiguzi | zhanguoce | shiji | zizhitongjian | shangjunshu
  unit_ref TEXT,                   -- chapter/pian reference
  text_zh TEXT,
  text_en TEXT,
  translation_source TEXT,         -- translation attribution or NULL
  analyst_summary TEXT,            -- modern analytical summary (original analysis, no copyright issue)
  principles_json TEXT,            -- strategic principles extracted
  fang_yuan_anchor TEXT,           -- novel decision this maps to (or NULL)
  passage_tag TEXT NOT NULL DEFAULT 'CHINESE_STRATEGIC_TRADITION'
      -- CANON_FANG_YUAN | AUTHOR_CONTEXT | CHINESE_STRATEGIC_TRADITION | HISTORICAL_ANALOGY | MODERN_ANALOGY
  ,lang TEXT DEFAULT 'zh'
);

CREATE TABLE IF NOT EXISTS analogies (
  id INTEGER PRIMARY KEY,
  classic_ref TEXT,                -- classics unit ref
  novel_ref TEXT,                  -- canon anchor
  modern_ref TEXT,
  mapping_json TEXT
);

-- --------------------------------------- lessons / plan / policy evolution
CREATE TABLE IF NOT EXISTS lessons (
  id INTEGER PRIMARY KEY,
  phase TEXT,                      -- corpus | policy | training | benchmark | ops
  category TEXT,                   -- understood | underweighted | engineering | legal | failure_mode
  statement TEXT NOT NULL,
  detail TEXT,
  origin TEXT,                     -- uploaded_notes | manus_run | super_z | test_run
  ts TEXT
);

CREATE TABLE IF NOT EXISTS plan_steps (
  id INTEGER PRIMARY KEY,
  phase TEXT,                      -- month1..month6 or milestone name
  objective TEXT NOT NULL,
  deliverables_json TEXT,
  status TEXT DEFAULT 'pending',   -- pending | in_progress | done | blocked
  notes TEXT
);

-- ------------------------------------------------------- research findings
CREATE TABLE IF NOT EXISTS findings (
  id INTEGER PRIMARY KEY,
  topic TEXT,
  query TEXT,
  source_url TEXT,
  source_title TEXT,
  finding TEXT,
  confidence TEXT DEFAULT 'medium',
  created_at TEXT
);

-- -------------------------------------------------- benchmark & evaluation
CREATE TABLE IF NOT EXISTS benchmark_items (
  id INTEGER PRIMARY KEY,
  item_key TEXT UNIQUE,
  layer TEXT NOT NULL,             -- canon | counterfactual | dynamic | quote_verification | style_concealment
  dimension TEXT NOT NULL,         -- objective_fixation | attachmentlessness | risk_calibration | reality_recognition | method_flexibility | path_optimization | concealment_discipline
  scenario_id TEXT,
  variant TEXT,
  prompt TEXT NOT NULL,
  rubric_json TEXT NOT NULL,       -- graded expectations
  holdout INTEGER DEFAULT 0        -- 1 = secret holdout, never used in training
);

CREATE TABLE IF NOT EXISTS benchmark_runs (
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL,
  item_id INTEGER REFERENCES benchmark_items(id),
  model TEXT,
  adapter TEXT,
  output_json TEXT,
  scores_json TEXT,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS simulator_runs (
  id INTEGER PRIMARY KEY,
  env TEXT NOT NULL,
  seed INTEGER,
  horizon INTEGER,
  agent TEXT,
  trajectory_json TEXT,
  final_metrics_json TEXT,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS test_runs (
  id INTEGER PRIMARY KEY,
  test_name TEXT NOT NULL,
  outcome TEXT NOT NULL,           -- pass | fail | xfail | skip
  duration_ms REAL,
  details_json TEXT,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS worklog (
  id INTEGER PRIMARY KEY,
  ts TEXT,
  task_id TEXT,
  agent TEXT,
  summary TEXT
);

-- ------------------------------------------------------------ FTS indexes
CREATE VIRTUAL TABLE IF NOT EXISTS fts_corpus    USING fts5(title, text, tags, content='');
CREATE VIRTUAL TABLE IF NOT EXISTS fts_decisions USING fts5(situation, selected_action, lesson, canon_anchor, content='');
CREATE VIRTUAL TABLE IF NOT EXISTS fts_quotes    USING fts5(text_en, text_zh, theme, speaker, content='');
CREATE VIRTUAL TABLE IF NOT EXISTS fts_classics  USING fts5(work, text_zh, text_en, analyst_summary, content='');
CREATE VIRTUAL TABLE IF NOT EXISTS fts_findings  USING fts5(topic, finding, content='');

-- ------------------------------------------------------------------- views
CREATE VIEW IF NOT EXISTS v_decision_stats AS
SELECT record_class, COUNT(*) AS n FROM decisions GROUP BY record_class;

CREATE VIEW IF NOT EXISTS v_quote_integrity AS
SELECT verification_status, COUNT(*) AS n FROM quotes GROUP BY verification_status;

CREATE VIEW IF NOT EXISTS v_db_stats AS
SELECT
  (SELECT COUNT(*) FROM sources)               AS sources,
  (SELECT COUNT(*) FROM corpus_items)          AS corpus_items,
  (SELECT COUNT(*) FROM characters)            AS characters,
  (SELECT COUNT(*) FROM factions)              AS factions,
  (SELECT COUNT(*) FROM gu_worms)              AS gu_worms,
  (SELECT COUNT(*) FROM events)                AS events,
  (SELECT COUNT(*) FROM decisions)             AS decisions,
  (SELECT COUNT(*) FROM quotes)                AS quotes,
  (SELECT COUNT(*) FROM classics)              AS classics,
  (SELECT COUNT(*) FROM analogies)             AS analogies,
  (SELECT COUNT(*) FROM lessons)               AS lessons,
  (SELECT COUNT(*) FROM findings)              AS findings,
  (SELECT COUNT(*) FROM benchmark_items)       AS benchmark_items,
  (SELECT COUNT(*) FROM benchmark_runs)        AS benchmark_runs,
  (SELECT COUNT(*) FROM simulator_runs)        AS simulator_runs,
  (SELECT COUNT(*) FROM test_runs)             AS test_runs;
