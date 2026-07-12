//! Top-k tool retrieval using Needle's contrastive head.
//!
//! Why: the decoder was finetuned on prompts with 1-6 tools. Stuffing all 16
//! termux tools (~850 tokens) into the prompt both slows prefill ~3x and
//! degrades accuracy. Needle ships a CLIP-style retrieval head for exactly
//! this: embed the query and each tool's schema into a shared 128-d space,
//! shortlist by cosine similarity, and prompt the decoder with only those.
//!
//! Tool embeddings are cached on disk beside the tools config; the cache is
//! keyed on each tool's schema text and a model-file fingerprint, so editing
//! a tool or swapping the model invalidates exactly the right entries.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use anyhow::Result;
use candle_core::Device;

use crate::model::Model;
use crate::tokenizer::Tokenizer;
use crate::tools::{single_tool_model_json, ToolSet};

const MAX_EMBED_TOKENS: usize = 256; // matches needle's retrieval default

fn hash_str(s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

fn model_fingerprint(model_dir: &Path) -> String {
    let p = model_dir.join("model.safetensors");
    match std::fs::metadata(&p) {
        Ok(m) => {
            let mtime = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            format!("{}:{}", m.len(), mtime)
        }
        Err(_) => "unknown".into(),
    }
}

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct Cache {
    model_fp: String,
    /// tool name -> (schema text hash, embedding)
    entries: std::collections::HashMap<String, (u64, Vec<f32>)>,
}

pub struct Retriever {
    cache_path: PathBuf,
    cache: Cache,
    dirty: bool,
}

impl Retriever {
    pub fn new(tools_path: &Path, model_dir: &Path) -> Self {
        let cache_path = PathBuf::from(format!("{}.embcache", tools_path.display()));
        let fp = model_fingerprint(model_dir);
        let mut cache: Cache = std::fs::read_to_string(&cache_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        if cache.model_fp != fp {
            // model changed -> all embeddings stale
            cache = Cache { model_fp: fp, entries: Default::default() };
        }
        Self { cache_path, cache, dirty: false }
    }

    fn embed_text(model: &Model, tok: &Tokenizer, text: &str, dev: &Device) -> Result<Vec<f32>> {
        let mut ids = tok.encode(text);
        ids.truncate(MAX_EMBED_TOKENS);
        model.embed_for_retrieval(&ids, dev)
    }

    /// Return indices of the top-k tools for this query (config order preserved).
    pub fn shortlist(
        &mut self,
        model: &Model,
        tok: &Tokenizer,
        toolset: &ToolSet,
        query: &str,
        k: usize,
        dev: &Device,
    ) -> Result<Vec<usize>> {
        // ensure every tool has a cached embedding
        for tool in &toolset.tools {
            let text = single_tool_model_json(tool);
            let h = hash_str(&text);
            let hit = self
                .cache
                .entries
                .get(&tool.name)
                .map(|(cached_h, _)| *cached_h == h)
                .unwrap_or(false);
            if !hit {
                let emb = Self::embed_text(model, tok, &text, dev)?;
                self.cache.entries.insert(tool.name.clone(), (h, emb));
                self.dirty = true;
            }
        }
        self.save_if_dirty();

        let q_emb = Self::embed_text(model, tok, query, dev)?;

        // cosine similarity == dot product (both sides are L2-normalized)
        let mut scored: Vec<(usize, f32)> = toolset
            .tools
            .iter()
            .enumerate()
            .map(|(i, tool)| {
                let (_h, emb) = &self.cache.entries[&tool.name];
                // [Rust Book Ch. 13] zip + map + sum: the iterator version of
                // a dot-product loop, compiled to the same tight code.
                let dot: f32 = q_emb.iter().zip(emb.iter()).map(|(a, b)| a * b).sum();
                (i, dot)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut keep: Vec<usize> = scored.into_iter().take(k).map(|(i, _)| i).collect();
        keep.sort_unstable(); // present tools in config order (training shuffles anyway)
        Ok(keep)
    }

    fn save_if_dirty(&mut self) {
        if self.dirty {
            if let Ok(json) = serde_json::to_string(&self.cache) {
                // best-effort: a read-only FS just means we re-embed next run
                let _ = std::fs::write(&self.cache_path, json);
            }
            self.dirty = false;
        }
    }
}
