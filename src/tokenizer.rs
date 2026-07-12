//! Pure-Rust reimplementation of Needle's SentencePiece BPE tokenizer.
//!
//! Needle's tokenizer spec (exported from the .model proto):
//!   - identity normalizer, no charsmap
//!   - remove_extra_whitespaces = true  (trim + collapse runs of spaces)
//!   - escape_whitespaces = true        (' ' becomes '\u{2581}' aka ▁)
//!   - add_dummy_prefix = true          (a leading ▁ is prepended)
//!   - byte_fallback = true             (unknown chars become <0xXX> byte pieces)
//!   - user-defined symbols <tool_call> and <tools> match atomically
//!
//! Verified token-for-token against the Python SentencePiece implementation
//! over the full finetuning corpus (see `sting verify-tokenizer`).

use std::collections::HashMap;

use anyhow::{Context, Result};

// [Rust Book Ch. 5] A plain struct with named fields. `pub` exposes fields to
// other modules; everything else stays private to this file.
pub struct Tokenizer {
    /// Piece text -> id. For byte pieces the *piece text* is "<0xXX>".
    piece_to_id: HashMap<String, u32>,
    /// id -> piece text (the raw SentencePiece piece, ▁ and all).
    pieces: Vec<String>,
    /// id -> piece type (1=NORMAL 2=UNK 3=CONTROL 4=USER_DEFINED 6=BYTE).
    types: Vec<u8>,
    /// id -> merge score (higher = merge earlier). NaN-free f32s from the proto.
    scores: Vec<f32>,
    /// byte value -> id of its <0xXX> piece.
    byte_ids: [u32; 256],
    /// user-defined symbols, longest first (greedy atomic matching).
    user_defined: Vec<String>,
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
}

pub const PAD_ID: u32 = 0;
pub const EOS_ID: u32 = 1;
#[allow(dead_code)]
pub const BOS_ID: u32 = 2;
pub const UNK_ID: u32 = 3;
pub const TOOL_CALL_ID: u32 = 4;
pub const TOOLS_ID: u32 = 5;

impl Tokenizer {
    /// Load from the JSON spec exported by `export_tokenizer_spec.py`.
    pub fn from_spec_file(path: &std::path::Path) -> Result<Self> {
        // [Rust Book Ch. 9] `?` propagates errors up to the caller instead of
        // panicking — the whole loader is one chain of fallible steps.
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading tokenizer spec {}", path.display()))?;
        let spec: serde_json::Value = serde_json::from_str(&raw)?;

        let piece_arr = spec["pieces"]
            .as_array()
            .context("tokenizer spec: missing pieces")?;

        let mut pieces = Vec::with_capacity(piece_arr.len());
        let mut types = Vec::with_capacity(piece_arr.len());
        let mut scores = Vec::with_capacity(piece_arr.len());
        let mut piece_to_id = HashMap::with_capacity(piece_arr.len());
        let mut byte_ids = [UNK_ID; 256];
        let mut user_defined = Vec::new();

        for (id, entry) in piece_arr.iter().enumerate() {
            let text = entry[0].as_str().context("piece text")?.to_string();
            let score = entry[1].as_f64().unwrap_or(0.0) as f32;
            let ptype = entry[2].as_u64().unwrap_or(1) as u8;

            if ptype == 6 {
                // byte piece "<0xXX>"
                let hex = &text[3..5];
                let b = u8::from_str_radix(hex, 16)?;
                byte_ids[b as usize] = id as u32;
            }
            if ptype == 4 {
                user_defined.push(text.clone());
            }
            piece_to_id.insert(text.clone(), id as u32);
            pieces.push(text);
            types.push(ptype);
            scores.push(score);
        }
        // Greedy longest-first matching for user-defined symbols.
        user_defined.sort_by_key(|s| std::cmp::Reverse(s.len()));

        Ok(Self {
            piece_to_id,
            pieces,
            types,
            scores,
            byte_ids,
            user_defined,
            add_dummy_prefix: spec["add_dummy_prefix"].as_bool().unwrap_or(true),
            remove_extra_whitespaces: spec["remove_extra_whitespaces"].as_bool().unwrap_or(true),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    /// Normalize + escape whitespace, mirroring SentencePiece's normalizer
    /// with the identity rule.
    fn normalize(&self, text: &str) -> String {
        let mut s = text.to_string();
        if self.remove_extra_whitespaces {
            // trim leading/trailing spaces, collapse internal runs
            let mut out = String::with_capacity(s.len());
            let mut prev_space = true; // leading spaces dropped
            for ch in s.chars() {
                if ch == ' ' {
                    if !prev_space {
                        out.push(' ');
                    }
                    prev_space = true;
                } else {
                    out.push(ch);
                    prev_space = false;
                }
            }
            while out.ends_with(' ') {
                out.pop();
            }
            s = out;
        }
        let mut escaped = String::with_capacity(s.len() + 4);
        if self.add_dummy_prefix && !s.is_empty() {
            escaped.push('\u{2581}');
        }
        for ch in s.chars() {
            escaped.push(if ch == ' ' { '\u{2581}' } else { ch });
        }
        escaped
    }

    /// Encode text to token ids (no BOS/EOS added — same as needle's encode).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        let normalized = self.normalize(text);

        // Split on user-defined symbols first (they are atomic).
        // [Rust Book Ch. 6] An enum models the two segment kinds; `match`
        // below forces us to handle both — no forgotten cases.
        enum Seg {
            Text(String),
            Symbol(u32),
        }
        let mut segments: Vec<Seg> = Vec::new();
        let mut rest = normalized.as_str();
        'outer: while !rest.is_empty() {
            // find the earliest user-defined match
            let mut best: Option<(usize, &str)> = None;
            for sym in &self.user_defined {
                if let Some(pos) = rest.find(sym.as_str()) {
                    let better = match best {
                        None => true,
                        Some((bpos, bsym)) => pos < bpos || (pos == bpos && sym.len() > bsym.len()),
                    };
                    if better {
                        best = Some((pos, sym));
                    }
                }
            }
            match best {
                Some((pos, sym)) => {
                    if pos > 0 {
                        segments.push(Seg::Text(rest[..pos].to_string()));
                    }
                    segments.push(Seg::Symbol(self.piece_to_id[sym]));
                    rest = &rest[pos + sym.len()..];
                }
                None => {
                    segments.push(Seg::Text(rest.to_string()));
                    break 'outer;
                }
            }
        }

        let mut ids = Vec::new();
        for seg in segments {
            match seg {
                Seg::Symbol(id) => ids.push(id),
                Seg::Text(t) => self.bpe_segment(&t, &mut ids),
            }
        }
        ids
    }

    /// Score-greedy BPE over one text segment (SentencePiece BPE semantics:
    /// always merge the adjacent pair with the highest score; ties -> leftmost).
    fn bpe_segment(&self, text: &str, out: &mut Vec<u32>) {
        // Working symbols as owned strings. A linked-list-ish Vec with tombstones
        // keeps merging O(n * merges) — plenty fast for <1k-token inputs.
        let mut syms: Vec<Option<String>> = text.chars().map(|c| Some(c.to_string())).collect();

        loop {
            // find best-scoring adjacent pair
            let mut best_score = f32::NEG_INFINITY;
            let mut best_i: Option<usize> = None;
            let mut i = 0;
            while i < syms.len() {
                if syms[i].is_none() {
                    i += 1;
                    continue;
                }
                // find next live symbol after i
                let mut j = i + 1;
                while j < syms.len() && syms[j].is_none() {
                    j += 1;
                }
                if j >= syms.len() {
                    break;
                }
                let merged = format!("{}{}", syms[i].as_ref().unwrap(), syms[j].as_ref().unwrap());
                if let Some(&id) = self.piece_to_id.get(&merged) {
                    // only NORMAL pieces participate in merges
                    if self.types[id as usize] == 1 {
                        let score = self.scores[id as usize];
                        if score > best_score {
                            best_score = score;
                            best_i = Some(i);
                        }
                    }
                }
                i = j;
            }

            // [Rust Book Ch. 6] `if let` handles just the case we care about.
            if let Some(i) = best_i {
                let mut j = i + 1;
                while syms[j].is_none() {
                    j += 1;
                }
                // [Rust Book Ch. 4] `take()` MOVES the String out of the slot,
                // leaving None behind — ownership transferred, no clone needed.
                let right = syms[j].take().unwrap();
                if let Some(left) = syms[i].as_mut() {
                    left.push_str(&right);
                }
            } else {
                break; // no mergeable pair left
            }
        }

        for sym in syms.into_iter().flatten() {
            match self.piece_to_id.get(&sym) {
                Some(&id) if self.types[id as usize] != 2 => out.push(id),
                _ => {
                    // byte fallback: emit one <0xXX> piece per UTF-8 byte
                    for b in sym.bytes() {
                        let id = self.byte_ids[b as usize];
                        out.push(if id == UNK_ID { UNK_ID } else { id });
                    }
                }
            }
        }
    }

    /// Decode ids to text (mirrors SentencePiece Decode: control pieces are
    /// skipped, byte pieces are accumulated into UTF-8, ▁ becomes ' ', and the
    /// dummy-prefix leading space is stripped).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut result = String::new();
        let mut byte_buf: Vec<u8> = Vec::new();

        // [Rust Book Ch. 13] a closure capturing nothing, used like a local fn
        fn flush(buf: &mut Vec<u8>, result: &mut String) {
            if !buf.is_empty() {
                result.push_str(&String::from_utf8_lossy(buf));
                buf.clear();
            }
        }

        for &id in ids {
            let idx = id as usize;
            if idx >= self.pieces.len() {
                continue;
            }
            match self.types[idx] {
                3 => continue,          // control (<pad>, </s>, <s>)
                2 => continue,          // <unk>
                6 => {
                    let hex = &self.pieces[idx][3..5];
                    if let Ok(b) = u8::from_str_radix(hex, 16) {
                        byte_buf.push(b);
                    }
                }
                _ => {
                    flush(&mut byte_buf, &mut result);
                    for ch in self.pieces[idx].chars() {
                        result.push(if ch == '\u{2581}' { ' ' } else { ch });
                    }
                }
            }
        }
        flush(&mut byte_buf, &mut result);
        // SentencePiece strips ALL leading spaces produced by ▁ pieces
        // (normally just the dummy prefix's one).
        result.trim_start_matches(' ').to_string()
    }

    /// Visible text a token contributes to decoded output — used by the
    /// constrained decoder's trie matcher (▁ -> ' ', bytes -> their char,
    /// control/unk -> "").
    pub fn token_visible_text(&self, id: u32) -> String {
        let idx = id as usize;
        if idx >= self.pieces.len() {
            return String::new();
        }
        match self.types[idx] {
            2 | 3 => String::new(),
            6 => {
                let hex = &self.pieces[idx][3..5];
                match u8::from_str_radix(hex, 16) {
                    Ok(b) => (b as char).to_string(),
                    Err(_) => String::new(),
                }
            }
            _ => self.pieces[idx].replace('\u{2581}', " "),
        }
    }
}
