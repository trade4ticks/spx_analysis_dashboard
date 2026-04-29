-- Research knowledge library — run once against the main (spx_interpolated) database
-- Auto-created on first API access, but included here for reference

CREATE TABLE IF NOT EXISTS research_knowledge (
    id         SERIAL PRIMARY KEY,
    category   TEXT NOT NULL DEFAULT 'policy',   -- terminology, assumption, caveat, policy
    rule       TEXT NOT NULL,
    active     BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Seed the initial rule
INSERT INTO research_knowledge (category, rule) VALUES
('terminology', 'Do not refer to OI-derived gamma as actual dealer gamma exposure. OI does not reveal dealer inventory. Use "OI-implied gamma proxy" or "model-estimated gamma exposure" unless actual dealer positioning data is available.');
