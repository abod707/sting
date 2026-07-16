# sting × The Rust Book — a chapter map

This project was built alongside a read-through of the official Rust Book
(2024 edition). Comments tagged `[Rust Book Ch. N]` mark the spots where a
concept from chapter N is doing real work. This file is the index.

If you've read **up to chapter 4**, everything below your line is a preview —
the annotations are written so you can still follow what the code does.

## Chapter 3 — Common concepts
Everywhere: `let` bindings, shadowing (`src/tokenizer.rs` `normalize` re-binds
`s` after each transformation), integer types chosen deliberately
(`u32` token ids, `usize` indices, `f32` weights).

## Chapter 4 — Ownership ★ (you are here)
- `src/tokenizer.rs::bpe_segment` — the BPE merge loop is an ownership
  showcase: `syms[j].take()` **moves** the right-hand String out of its slot
  (leaving `None`), then `push_str` borrows it just long enough to append.
  No clones in the hot loop.
- `src/main.rs::parse_args` — each `arg` String is **moved** into `opts.query`;
  after the move, the loop variable is gone. Try adding a `println!("{arg}")`
  after the move and watch the borrow checker object.
- `src/retrieval.rs::shortlist` — `&toolset.tools` is borrowed immutably while
  `self.cache` is mutated: two disjoint borrows, which is exactly why the
  borrow checker allows it.
- `src/generate.rs::TrieNode::insert` — a mutable borrow (`node`) that is
  re-assigned deeper into the tree each iteration. This is the "borrows can
  move through a structure" pattern.
- `src/model.rs::Attention::self_step` — the KV cache is a `&mut Option<Tensor>`
  per layer. `cache_k.take()` **moves** the old cached tensor out (leaving
  `None`), we `Tensor::cat` the new column onto it, then write the grown tensor
  back with `*cache_k = Some(...)`. Same `Option::take` move trick as the BPE
  loop, one level up: you can't append to something you only borrowed, so you
  take ownership, transform, and put it back.
- `src/tokenizer.rs::bpe_segment` (rewrite) — instead of a `Vec<&Symbol>` linked
  list (which would tangle the borrow checker with self-references), the live
  symbols are threaded by **index** through `next`/`prev: Vec<usize>` with a
  `usize::MAX` sentinel. Owning the strings in one `Vec` and linking by index is
  the idiomatic Rust way around "linked lists fight the borrow checker."

## Chapter 5 — Structs
`Opts` (main.rs), `Tool`/`ExecSpec` (tools.rs), `Tokenizer` (tokenizer.rs),
`Model`/`Attention`/`Config` (model.rs). Note which fields are `pub` and which
stay private — that boundary is the API.

## Chapter 6 — Enums & pattern matching
- `ArgTemplate` (tools.rs) — a union of "literal token" vs "argument slot",
  deserialized straight from JSON via serde's untagged enums.
- `JsonState` + the big `match` in `generate.rs::feed_char` — a state machine
  where the compiler checks you handled every state.
- `Outcome` (dispatch.rs) — four ways a dispatch can end, each rendered
  differently by main. Adding a fifth variant would force main to handle it.
- `Option` everywhere: `Option<ExecSpec>` = "this tool may not be executable",
  `Option<Rope>` = "cross-attention has no rotary embeddings".

## Chapter 8 — Collections
`Vec` as token buffers, `HashMap` for vocab and embedding caches,
`String` vs `&str` decisions in the tokenizer (owned symbols because merges
create new strings).

## Chapter 9 — Error handling
`anyhow::Result` + `?` end to end: `main` returns `Result`, so every fallible
layer (file IO → JSON parse → tensor lookup → process spawn) propagates with
`.with_context(...)` breadcrumbs. `bail!` for unrecoverable config mismatches.

## Chapter 10 — Generics & traits
serde's `Deserialize` derive on `Config`/`ExecSpec`/`ArgTemplate` — trait-driven
JSON. candle's `Tensor` methods are generic over dtypes; we pin f32.

## Chapter 13 — Closures & iterators
- `run_one` in main.rs — a closure capturing the model by shared borrow and the
  retriever by mutable borrow (hence `let mut run_one`).
- Dot-product via `zip + map + sum` in retrieval.rs; argmax scan in generate.rs.
- `softmax_last` in model.rs — the same per-row closure is fed to either
  `chunks_mut` (serial) or rayon's `par_chunks_mut` (one row per core). Because
  each row owns a disjoint `&mut [f32]` slice, the borrow checker proves the
  parallel version is data-race-free at compile time — "fearless concurrency"
  with no locks.

## Chapter 15 — Smart pointers (preview)
candle `Tensor`s are `Arc`-backed: `.clone()` in `Weights::take` copies a
pointer, not 52MB of weights. The trie's `HashMap<char, TrieNode>` heap-allocates
children without explicit `Box`.

## Suggested drills (15-30 min each)
1. **Ch. 4:** In `bpe_segment`, replace `take()` with `clone()` and measure
   tokenizer parity runtime on `parity.jsonl`. Then explain why the original
   compiles without cloning.
2. **Ch. 6:** Add a `--json` output mode to `Outcome` handling in main.rs —
   the compiler will walk you through every variant.
3. **Ch. 9:** Make `ToolSet::load` report *which* tool entry failed to parse
   (index + name) using `.with_context`.
4. **Ch. 13:** Rewrite the argmax loop in `generate.rs` as an iterator chain
   (`iter().enumerate().max_by(...)`) and compare readability.
