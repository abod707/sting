//! Generation: encoder-input assembly, greedy decode loop, and a port of
//! needle's grammar-constrained decoding (needle/model/constrained.py).
//!
//! The constrainer tracks where we are inside the model's compact JSON output
//!   [{"name":"tool","arguments":{"key":value,...}}]
//! and masks logits so tool names / argument keys can only be valid ones.
//! Argument *values* are unconstrained.

use std::collections::HashMap;

use anyhow::Result;
use candle_core::Device;

use crate::model::Model;
use crate::tokenizer::{Tokenizer, EOS_ID, TOOLS_ID};

pub const MAX_ENC_LEN: usize = 1024;
pub const MAX_GEN_LEN: usize = 192;

/// Build the encoder input exactly like needle's _build_encoder_input:
/// [query_tokens (capped), <tools>, tools_tokens (capped to what remains)]
pub fn build_encoder_input(tok: &Tokenizer, query: &str, tools_json: &str) -> Vec<u32> {
    let mut q = tok.encode(query);
    let t = tok.encode(tools_json);
    let max_query = MAX_ENC_LEN - 2;
    if q.len() > max_query {
        q.truncate(max_query);
    }
    let remaining = MAX_ENC_LEN - q.len() - 1;
    let mut ids = q;
    ids.push(TOOLS_ID);
    ids.extend(t.into_iter().take(remaining));
    ids
}

// ---------------------------------------------------------------------------
// Character trie over valid names/keys
// ---------------------------------------------------------------------------

// [Rust Book Ch. 15 preview] A tree of owned children. Box isn't needed here
// because HashMap already heap-allocates its entries.
#[derive(Default)]
struct TrieNode {
    children: HashMap<char, TrieNode>,
    terminal: bool,
}

impl TrieNode {
    fn insert(&mut self, word: &str) {
        // [Rust Book Ch. 4] `node` is a mutable BORROW that we re-assign as we
        // walk down the tree — each step narrows the borrow to a child.
        let mut node = self;
        for ch in word.chars() {
            node = node.children.entry(ch).or_default();
        }
        node.terminal = true;
    }

    fn walk(&self, prefix: &str) -> Option<&TrieNode> {
        let mut node = self;
        for ch in prefix.chars() {
            node = node.children.get(&ch)?;
        }
        Some(node)
    }

    /// Is `text` a valid continuation from this node? A '"' ends the
    /// constrained span and requires the node to be terminal.
    fn valid_continuation(&self, text: &str) -> bool {
        let mut node = self;
        for ch in text.chars() {
            if ch == '"' {
                return node.terminal;
            }
            match node.children.get(&ch) {
                Some(n) => node = n,
                None => return false,
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// JSON state machine (port of constrained.py::JsonStateMachine)
// ---------------------------------------------------------------------------

#[derive(PartialEq, Clone, Copy)]
enum JsonState {
    Free,
    InName,
    InArgKey,
}

struct StateMachine {
    state: JsonState,
    buffer: String,
    constrained_buf: String,
    current_function: String,
    in_arguments: bool,
    arguments_depth: i32,
    nesting_depth: i32,
    in_string: bool,
    prev_escape: bool,
}

impl StateMachine {
    fn new() -> Self {
        Self {
            state: JsonState::Free,
            buffer: String::new(),
            constrained_buf: String::new(),
            current_function: String::new(),
            in_arguments: false,
            arguments_depth: 0,
            nesting_depth: 0,
            in_string: false,
            prev_escape: false,
        }
    }

    fn feed(&mut self, text: &str) {
        for ch in text.chars() {
            self.feed_char(ch);
        }
    }

    fn feed_char(&mut self, ch: char) {
        if self.state != JsonState::Free {
            if ch == '"' {
                if self.state == JsonState::InName {
                    self.current_function = self.constrained_buf.clone();
                }
                self.constrained_buf.clear();
                self.state = JsonState::Free;
            } else {
                self.constrained_buf.push(ch);
            }
            self.buffer.push(ch);
            return;
        }

        self.buffer.push(ch);

        if self.in_string {
            if self.prev_escape {
                self.prev_escape = false;
                return;
            }
            match ch {
                '\\' => self.prev_escape = true,
                '"' => self.in_string = false,
                _ => {}
            }
            return;
        }

        match ch {
            '{' | '[' => self.nesting_depth += 1,
            '}' | ']' => {
                self.nesting_depth = (self.nesting_depth - 1).max(0);
                if ch == '}' && self.in_arguments && self.nesting_depth < self.arguments_depth {
                    self.in_arguments = false;
                }
                return;
            }
            _ => {}
        }

        if self.buffer.ends_with("\"name\":\"") && !self.in_arguments {
            self.state = JsonState::InName;
            self.constrained_buf.clear();
            return;
        }
        if self.buffer.ends_with("\"arguments\":{") {
            self.in_arguments = true;
            self.arguments_depth = self.nesting_depth;
            return;
        }
        if self.in_arguments
            && self.nesting_depth == self.arguments_depth
            && (self.buffer.ends_with("{\"") || self.buffer.ends_with(",\""))
        {
            self.state = JsonState::InArgKey;
            self.constrained_buf.clear();
            return;
        }
        if ch == '"' && self.value_quote() {
            self.in_string = true;
        }
    }

    fn value_quote(&self) -> bool {
        // the current '"' opens a string VALUE iff the previous non-space char is ':'
        for c in self.buffer[..self.buffer.len() - 1].chars().rev() {
            if c.is_whitespace() {
                continue;
            }
            return c == ':';
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Constrained decoder
// ---------------------------------------------------------------------------

pub struct Constrainer {
    name_trie: TrieNode,
    param_tries: HashMap<String, TrieNode>,
    machine: StateMachine,
    token_texts: Vec<String>,
    /// first visible char -> candidate token ids (speeds up masking)
    first_char_index: HashMap<char, Vec<u32>>,
}

impl Constrainer {
    pub fn new(tools_json: &str, tok: &Tokenizer) -> Self {
        let mut name_trie = TrieNode::default();
        let mut param_tries = HashMap::new();

        if let Ok(serde_json::Value::Array(tools)) = serde_json::from_str(tools_json) {
            for tool in &tools {
                let name = tool["name"].as_str().unwrap_or("");
                if name.is_empty() {
                    continue;
                }
                name_trie.insert(name);
                let mut ptrie = TrieNode::default();
                if let Some(params) = tool["parameters"].as_object() {
                    for (key, val) in params {
                        if val.is_object() {
                            ptrie.insert(key);
                        }
                    }
                }
                param_tries.insert(name.to_string(), ptrie);
            }
        }

        let vocab = tok.vocab_size();
        let mut token_texts = Vec::with_capacity(vocab);
        let mut first_char_index: HashMap<char, Vec<u32>> = HashMap::new();
        for id in 0..vocab as u32 {
            let text = tok.token_visible_text(id);
            if let Some(first) = text.chars().next() {
                first_char_index.entry(first).or_default().push(id);
            }
            token_texts.push(text);
        }

        Self {
            name_trie,
            param_tries,
            machine: StateMachine::new(),
            token_texts,
            first_char_index,
        }
    }

    pub fn is_active(&self) -> bool {
        self.machine.state != JsonState::Free
    }

    /// Mask logits in-place: only tokens that continue a valid name/key survive.
    pub fn constrain(&self, logits: &mut [f32]) {
        let trie = match self.machine.state {
            JsonState::Free => return,
            JsonState::InName => &self.name_trie,
            JsonState::InArgKey => match self.param_tries.get(&self.machine.current_function) {
                Some(t) => t,
                None => return,
            },
        };
        let node = match trie.walk(&self.machine.constrained_buf) {
            Some(n) => n,
            None => return, // off-trie: fall back to unconstrained
        };

        let mut allowed: Vec<u32> = Vec::new();
        let mut first_chars: Vec<char> = node.children.keys().copied().collect();
        if node.terminal {
            first_chars.push('"');
        }
        for fc in first_chars {
            if let Some(cands) = self.first_char_index.get(&fc) {
                for &tid in cands {
                    if node.valid_continuation(&self.token_texts[tid as usize]) {
                        allowed.push(tid);
                    }
                }
            }
        }
        if allowed.is_empty() {
            return; // match python: warn-and-fallback (we just fall back)
        }
        // [Rust Book Ch. 8] a Vec<bool> as a cheap mask keyed by token id
        let mut keep = vec![false; logits.len()];
        for tid in allowed {
            keep[tid as usize] = true;
        }
        for (i, l) in logits.iter_mut().enumerate() {
            if !keep[i] {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    pub fn update(&mut self, token_id: u32) {
        // feed the token's visible text into the state machine
        let text = self.token_texts[token_id as usize].clone();
        self.machine.feed(&text);
    }
}

// ---------------------------------------------------------------------------
// Greedy generation loop
// ---------------------------------------------------------------------------

pub struct GenStats {
    pub prefill_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_ms: u128,
    pub decode_ms: u128,
}

pub fn generate(
    model: &Model,
    tok: &Tokenizer,
    query: &str,
    tools_json: &str,
    constrained: bool,
    dev: &Device,
) -> Result<(String, GenStats)> {
    let enc_ids = build_encoder_input(tok, query, tools_json);

    let t0 = std::time::Instant::now();
    let enc_out = model.encode(&enc_ids, dev)?;
    // Cross-attention K/V depend only on the encoder output, so compute them
    // once here (prefill-side work) rather than re-projecting the whole encoder
    // output on every decoded token.
    let mut cache = model.init_decode(&enc_out)?;
    let prefill_ms = t0.elapsed().as_millis();

    let mut constrainer = constrained.then(|| Constrainer::new(tools_json, tok));

    // Decoder starts from [EOS]; the model emits <tool_call> then the JSON. Each
    // step feeds a single token; the KV cache holds all prior positions.
    let mut token: u32 = EOS_ID;
    let mut generated: Vec<u32> = Vec::new();

    let t1 = std::time::Instant::now();
    for _ in 0..MAX_GEN_LEN - 1 {
        let logits = model.decode_step(&mut cache, token, dev)?;
        let mut logits: Vec<f32> = logits.to_vec1()?;

        // [Rust Book Ch. 6] Option<&mut T> + if let: borrow the constrainer
        // only when it exists (unconstrained mode passes straight through).
        if let Some(c) = constrainer.as_mut() {
            if c.is_active() {
                c.constrain(&mut logits);
            }
        }

        // greedy argmax
        let mut best = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best = i;
            }
        }
        let next = best as u32;

        if let Some(c) = constrainer.as_mut() {
            c.update(next);
        }
        if next == EOS_ID {
            break;
        }
        generated.push(next);
        token = next; // feed the freshly generated token on the next step
    }
    let decode_ms = t1.elapsed().as_millis();

    let mut text = tok.decode(&generated);
    if let Some(stripped) = text.strip_prefix("<tool_call>") {
        text = stripped.to_string();
    }

    Ok((
        text,
        GenStats {
            prefill_tokens: enc_ids.len(),
            generated_tokens: generated.len(),
            prefill_ms,
            decode_ms,
        },
    ))
}
