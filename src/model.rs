//! Needle's "Simple Attention Network" (26M params) in candle.
//!
//! Encoder-decoder transformer with NO feed-forward layers anywhere:
//! every block is just (ZCRMSNorm -> attention -> gated residual).
//! Faithful port of needle/model/architecture.py (JAX/Flax reference).
//!
//! Inference-only, batch size 1, f32 throughout. The JAX reference runs
//! bfloat16; f32 here is a superset in precision, and greedy outputs were
//! verified to match the reference on the finetuning corpus.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_encoder_layers: usize,
    pub num_decoder_layers: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_max_seq")]
    pub max_seq_len: usize,
}

fn default_rope_theta() -> f64 {
    10000.0
}
fn default_max_seq() -> usize {
    1024
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }
}

/// Zero-centred RMSNorm: (1 + scale) * x / rms(x), rms computed in f32.
/// `scale` is stored zero-centred (trained around 0), hence the 1 + scale.
fn zcrms_norm(x: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let eps = 1e-6f64;
    let x2 = x.sqr()?.mean_keepdim(D::Minus1)?;
    let rms = (x2 + eps)?.sqrt()?;
    let one_plus = (scale + 1.0)?;
    // broadcast: x / rms * (1 + scale)
    let normed = x.broadcast_div(&rms)?;
    Ok(normed.broadcast_mul(&one_plus)?)
}

/// Rotary embeddings, split-half convention (NOT interleaved):
///   x1 = x[..., :half], x2 = x[..., half:]
///   out = concat(x1*cos - x2*sin, x2*cos + x1*sin)
struct Rope {
    cos: Tensor, // (max_seq, half)
    sin: Tensor,
}

impl Rope {
    fn new(head_dim: usize, max_seq: usize, theta: f64, dev: &Device) -> Result<Self> {
        let half = head_dim / 2;
        let freqs: Vec<f32> = (0..half)
            .map(|i| 1.0f32 / (theta as f32).powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        let mut cos = Vec::with_capacity(max_seq * half);
        let mut sin = Vec::with_capacity(max_seq * half);
        for t in 0..max_seq {
            for f in &freqs {
                let angle = t as f32 * f;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        Ok(Self {
            cos: Tensor::from_vec(cos, (max_seq, half), dev)?,
            sin: Tensor::from_vec(sin, (max_seq, half), dev)?,
        })
    }

    /// x: (B, H, T, head_dim) -> same shape with rotary applied.
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, _h, t, hd) = x.dims4()?;
        let half = hd / 2;
        let cos = self.cos.i(..t)?.reshape((1, 1, t, half))?;
        let sin = self.sin.i(..t)?.reshape((1, 1, t, half))?;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let out1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let out2 = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;
        Ok(Tensor::cat(&[out1, out2], D::Minus1)?)
    }

    /// Apply rotary embeddings to a single-position tensor x: (B, H, 1, head_dim)
    /// at absolute position `pos`. Numerically identical to `apply` restricted to
    /// row `pos` — used by the KV-cached incremental decoder.
    fn apply_at(&self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let hd = x.dim(D::Minus1)?;
        let half = hd / 2;
        let cos = self.cos.i(pos..pos + 1)?.reshape((1, 1, 1, half))?;
        let sin = self.sin.i(pos..pos + 1)?.reshape((1, 1, 1, half))?;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let out1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let out2 = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;
        Ok(Tensor::cat(&[out1, out2], D::Minus1)?)
    }
}

/// One attention module (self- or cross-). Projection weights are stored
/// PRE-TRANSPOSED to (in, out) at load (see `Weights::take_t`), so `linear`
/// computes y = x @ W directly with no per-call transpose/copy.
struct Attention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    out_proj: Tensor,
    q_norm: Tensor, // (head_dim,) zero-centred scales
    k_norm: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(w: &Weights, prefix: &str, cfg: &Config) -> Result<Self> {
        Ok(Self {
            q_proj: w.take_t(&format!("{prefix}.q_proj.weight"))?,
            k_proj: w.take_t(&format!("{prefix}.k_proj.weight"))?,
            v_proj: w.take_t(&format!("{prefix}.v_proj.weight"))?,
            out_proj: w.take_t(&format!("{prefix}.out_proj.weight"))?,
            q_norm: w.take(&format!("{prefix}.q_norm.scale"))?,
            k_norm: w.take(&format!("{prefix}.k_norm.scale"))?,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim(),
        })
    }

    /// q_input: (B, Tq, d), kv_input: (B, Tk, d)
    /// rope: Some(rope) for self-attention, None for cross-attention.
    /// causal: apply lower-triangular mask (decoder self-attention).
    fn forward(
        &self,
        q_input: &Tensor,
        kv_input: &Tensor,
        rope: Option<&Rope>,
        causal: bool,
    ) -> Result<Tensor> {
        let (b, tq, _d) = q_input.dims3()?;
        let tk = kv_input.dim(1)?;
        let hd = self.head_dim;

        let q = linear(q_input, &self.q_proj)?; // (B, Tq, H*hd)
        let k = linear(kv_input, &self.k_proj)?; // (B, Tk, KV*hd)
        let v = linear(kv_input, &self.v_proj)?;

        let q = q.reshape((b, tq, self.num_heads, hd))?.transpose(1, 2)?; // (B,H,Tq,hd)
        let k = k.reshape((b, tk, self.num_kv_heads, hd))?.transpose(1, 2)?;
        let v = v.reshape((b, tk, self.num_kv_heads, hd))?.transpose(1, 2)?;

        // per-head-dim RMS norms on q and k (before GQA repeat + rope)
        let q = zcrms_norm(&q, &self.q_norm)?;
        let k = zcrms_norm(&k, &self.k_norm)?;

        // GQA: repeat each kv head `repeats` times consecutively
        let repeats = self.num_heads / self.num_kv_heads;
        let (k, v) = if repeats > 1 {
            (repeat_kv(&k, repeats)?, repeat_kv(&v, repeats)?)
        } else {
            (k, v)
        };

        let (q, k) = match rope {
            Some(r) => (r.apply(&q)?, r.apply(&k)?),
            None => (q, k),
        };

        let scale = (hd as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? / scale)?; // (B,H,Tq,Tk)

        if causal {
            scores = apply_causal_mask(&scores)?;
        }

        let probs = softmax_last(&scores)?;
        let out = probs.matmul(&v.contiguous()?)?; // (B,H,Tq,hd)
        let out = out.transpose(1, 2)?.reshape((b, tq, self.num_heads * hd))?;
        linear(&out, &self.out_proj)
    }

    /// Reshape a projected (1, T, H*hd) tensor to per-head (1, H, T, hd).
    fn split_heads(&self, x: &Tensor, heads: usize) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        Ok(x.reshape((b, t, heads, self.head_dim))?.transpose(1, 2)?)
    }

    /// Precompute cross-attention K/V from the (fixed) encoder output — done
    /// ONCE per query in `init_decode`, not per decoded token. Returns
    /// `(k_t, v)` where k_t is pre-transposed to (1, H, hd, Tenc) for the score
    /// matmul and v is (1, H, Tenc, hd). No RoPE on cross-attention.
    fn cross_kv(&self, kv_input: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = self.split_heads(&linear(kv_input, &self.k_proj)?, self.num_kv_heads)?;
        let v = self.split_heads(&linear(kv_input, &self.v_proj)?, self.num_kv_heads)?;
        let k = zcrms_norm(&k, &self.k_norm)?;
        let repeats = self.num_heads / self.num_kv_heads;
        let (k, v) = if repeats > 1 {
            (repeat_kv(&k, repeats)?, repeat_kv(&v, repeats)?)
        } else {
            (k, v)
        };
        Ok((k.transpose(2, 3)?.contiguous()?, v.contiguous()?))
    }

    /// Single-token cross-attention against precomputed encoder K/V.
    /// q_input: (1, 1, d) -> (1, 1, d).
    fn cross_step(&self, q_input: &Tensor, k_t: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (b, tq, _d) = q_input.dims3()?; // tq == 1
        let q = self.split_heads(&linear(q_input, &self.q_proj)?, self.num_heads)?;
        let q = zcrms_norm(&q, &self.q_norm)?.contiguous()?; // (1,H,1,hd)
        let scale = (self.head_dim as f64).sqrt();
        let scores = (q.matmul(k_t)? / scale)?; // (1,H,1,Tenc)
        let probs = softmax_last(&scores)?;
        let out = probs.matmul(v)?; // (1,H,1,hd)
        let out = out.transpose(1, 2)?.reshape((b, tq, self.num_heads * self.head_dim))?;
        linear(&out, &self.out_proj)
    }

    /// Single-token self-attention with an incremental KV cache. Computes Q/K/V
    /// for just the new token at absolute position `pos`, appends its K/V to the
    /// per-layer cache, and attends over the whole cache. No causal mask is
    /// needed: a lone query at the end legitimately sees every cached key.
    /// `cache_k` holds K pre-transposed as (1, H, hd, T); `cache_v` holds V as
    /// (1, H, T, hd). q_input: (1, 1, d) -> (1, 1, d).
    fn self_step(
        &self,
        q_input: &Tensor,
        rope: &Rope,
        pos: usize,
        cache_k: &mut Option<Tensor>,
        cache_v: &mut Option<Tensor>,
    ) -> Result<Tensor> {
        let (b, tq, _d) = q_input.dims3()?; // tq == 1
        let q = self.split_heads(&linear(q_input, &self.q_proj)?, self.num_heads)?;
        let k = self.split_heads(&linear(q_input, &self.k_proj)?, self.num_kv_heads)?;
        let v = self.split_heads(&linear(q_input, &self.v_proj)?, self.num_kv_heads)?;
        let q = zcrms_norm(&q, &self.q_norm)?;
        let k = zcrms_norm(&k, &self.k_norm)?;
        let repeats = self.num_heads / self.num_kv_heads;
        let (k, v) = if repeats > 1 {
            (repeat_kv(&k, repeats)?, repeat_kv(&v, repeats)?)
        } else {
            (k, v)
        };
        // RoPE at the token's absolute position (matches the full-sequence path).
        let q = rope.apply_at(&q, pos)?.contiguous()?; // (1,H,1,hd)
        let k = rope.apply_at(&k, pos)?;
        let k_t = k.transpose(2, 3)?.contiguous()?; // (1,H,hd,1)
        let v = v.contiguous()?; // (1,H,1,hd)

        // Append the new token's K/V to the running cache.
        let new_k = match cache_k.take() {
            Some(prev) => Tensor::cat(&[&prev, &k_t], 3)?, // grow along hd's time axis
            None => k_t,
        };
        let new_v = match cache_v.take() {
            Some(prev) => Tensor::cat(&[&prev, &v], 2)?,
            None => v,
        };

        let scale = (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&new_k)? / scale)?; // (1,H,1,T)
        let probs = softmax_last(&scores)?;
        let out = probs.matmul(&new_v)?; // (1,H,1,hd)
        let out = out.transpose(1, 2)?.reshape((b, tq, self.num_heads * self.head_dim))?;
        let y = linear(&out, &self.out_proj)?;

        *cache_k = Some(new_k);
        *cache_v = Some(new_v);
        Ok(y)
    }
}

/// Numerically stable softmax over the last dim (max-subtracted).
fn softmax_last(x: &Tensor) -> Result<Tensor> {
    let m = x.max_keepdim(D::Minus1)?;
    let e = x.broadcast_sub(&m)?.exp()?;
    let s = e.sum_keepdim(D::Minus1)?;
    Ok(e.broadcast_div(&s)?)
}

/// y = x @ W for W PRE-TRANSPOSED to (in, out) at load. candle matmul over the
/// last 2 dims. No per-call transpose or contiguous copy of the weight.
fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let (b, t, d_in) = x.dims3()?;
    let d_out = w.dim(1)?;
    let y = x
        .reshape((b * t, d_in))?
        .matmul(w)?
        .reshape((b, t, d_out))?;
    Ok(y)
}

fn repeat_kv(x: &Tensor, repeats: usize) -> Result<Tensor> {
    // (B, KV, T, hd) -> (B, KV*repeats, T, hd), each head repeated consecutively
    let (b, kv, t, hd) = x.dims4()?;
    let x = x
        .unsqueeze(2)?
        .expand((b, kv, repeats, t, hd))?
        .reshape((b, kv * repeats, t, hd))?;
    Ok(x)
}

fn apply_causal_mask(scores: &Tensor) -> Result<Tensor> {
    let (_b, _h, tq, tk) = scores.dims4()?;
    let dev = scores.device();
    // mask[i, j] = 0 where j <= i, -inf where j > i  (tq == tk here)
    let mut data = vec![0f32; tq * tk];
    for i in 0..tq {
        for j in 0..tk {
            if j > i {
                data[i * tk + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask = Tensor::from_vec(data, (1, 1, tq, tk), dev)?;
    Ok(scores.broadcast_add(&mask)?)
}

struct EncoderLayer {
    norm1: Tensor,
    self_attn: Attention,
    attn_gate: f32, // sigmoid already applied at load
}

struct DecoderLayer {
    norm1: Tensor,
    self_attn: Attention,
    self_gate: f32,
    norm2: Tensor,
    cross_attn: Attention,
    cross_gate: f32,
}

/// Temporary weight map that hands tensors out by name.
struct Weights {
    map: HashMap<String, Tensor>,
}

impl Weights {
    fn take(&self, name: &str) -> Result<Tensor> {
        self.map
            .get(name)
            .cloned() // candle Tensors are cheap Arc clones (see Rust Book ch. 15)
            .with_context(|| format!("missing tensor {name}"))
    }
    /// Take a projection matrix stored as (out, in) and return it PRE-TRANSPOSED
    /// to (in, out), contiguous. `linear` then multiplies by it directly, so the
    /// per-call `w.t()?.contiguous()?` (a transpose + full copy on every matmul,
    /// every layer, every token) is paid once here at load instead.
    fn take_t(&self, name: &str) -> Result<Tensor> {
        Ok(self.take(name)?.t()?.contiguous()?)
    }
    fn scalar(&self, name: &str) -> Result<f32> {
        let t = self.take(name)?;
        Ok(t.reshape(())?.to_scalar::<f32>()?)
    }
}

pub struct Model {
    pub cfg: Config,
    embedding: Tensor, // (vocab, d_model)
    embed_scale: f64,
    enc_layers: Vec<EncoderLayer>,
    enc_final_norm: Tensor,
    dec_layers: Vec<DecoderLayer>,
    dec_final_norm: Tensor,
    rope: Rope,
    /// Contrastive retrieval head (query/tool embeddings for top-k tool
    /// shortlisting). Optional: older conversions may not include it.
    contrastive: Option<ContrastiveHead>,
}

struct ContrastiveHead {
    hidden_w: Tensor, // PRE-TRANSPOSED to (d_model, d_model/4)
    hidden_b: Tensor, // (d_model/4,)
    proj_w: Tensor,   // PRE-TRANSPOSED to (d_model/4, contrastive_dim)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Per-query decoder state for KV-cached generation. Cross-attention K/V are
/// computed once from the encoder output; self-attention K/V grow by one column
/// per decoded token. Create with `Model::init_decode`, advance with
/// `Model::decode_step`.
pub struct KvCache {
    self_k: Vec<Option<Tensor>>, // per layer, (1, H, hd, T) — pre-transposed
    self_v: Vec<Option<Tensor>>, // per layer, (1, H, T, hd)
    cross_k_t: Vec<Tensor>,      // per layer, (1, H, hd, Tenc) — pre-transposed
    cross_v: Vec<Tensor>,        // per layer, (1, H, Tenc, hd)
    pos: usize,                  // absolute position of the next token
}

impl Model {
    pub fn load(dir: &Path, dev: &Device) -> Result<Self> {
        let cfg_raw = std::fs::read_to_string(dir.join("config.json"))
            .with_context(|| format!("reading {}/config.json", dir.display()))?;
        let cfg: Config = serde_json::from_str(&cfg_raw)?;

        let tensors = candle_core::safetensors::load(dir.join("model.safetensors"), dev)?;
        // ensure f32
        let mut map = HashMap::new();
        for (k, v) in tensors {
            map.insert(k, v.to_dtype(DType::F32)?);
        }
        let w = Weights { map };

        let mut enc_layers = Vec::with_capacity(cfg.num_encoder_layers);
        for i in 0..cfg.num_encoder_layers {
            let p = format!("encoder.layers.{i}");
            enc_layers.push(EncoderLayer {
                norm1: w.take(&format!("{p}.norm1.scale"))?,
                self_attn: Attention::load(&w, &format!("{p}.self_attn"), &cfg)?,
                attn_gate: sigmoid(w.scalar(&format!("{p}.attn_gate"))?),
            });
        }

        let mut dec_layers = Vec::with_capacity(cfg.num_decoder_layers);
        for i in 0..cfg.num_decoder_layers {
            let p = format!("decoder.layers.{i}");
            dec_layers.push(DecoderLayer {
                norm1: w.take(&format!("{p}.norm1.scale"))?,
                self_attn: Attention::load(&w, &format!("{p}.self_attn"), &cfg)?,
                self_gate: sigmoid(w.scalar(&format!("{p}.self_attn_gate"))?),
                norm2: w.take(&format!("{p}.norm2.scale"))?,
                cross_attn: Attention::load(&w, &format!("{p}.cross_attn"), &cfg)?,
                cross_gate: sigmoid(w.scalar(&format!("{p}.cross_attn_gate"))?),
            });
        }

        // [Rust Book Ch. 6] Option models "the head may not be in the file":
        // `.ok()` turns each Result into an Option, and the ? inside a
        // closure-less chain keeps all three together.
        let contrastive = match (
            w.take("contrastive_hidden.weight"),
            w.take("contrastive_hidden.bias"),
            w.take("contrastive_proj.weight"),
        ) {
            (Ok(hidden_w), Ok(hidden_b), Ok(proj_w)) => Some(ContrastiveHead {
                hidden_w: hidden_w.t()?.contiguous()?,
                hidden_b,
                proj_w: proj_w.t()?.contiguous()?,
            }),
            _ => None,
        };

        Ok(Self {
            embedding: w.take("embedding.weight")?,
            embed_scale: (cfg.d_model as f64).sqrt(),
            enc_final_norm: w.take("encoder.final_norm.scale")?,
            dec_final_norm: w.take("decoder.final_norm.scale")?,
            rope: Rope::new(cfg.head_dim(), cfg.max_seq_len, cfg.rope_theta, dev)?,
            cfg,
            enc_layers,
            dec_layers,
            contrastive,
        })
    }

    pub fn has_retrieval_head(&self) -> bool {
        self.contrastive.is_some()
    }

    /// Embed text into the L2-normalized contrastive space (retrieval).
    /// Port of SimpleAttentionNetwork.encode_contrastive: encoder output ->
    /// mean-pool over positions -> relu(hidden) -> proj -> L2 normalize.
    pub fn embed_for_retrieval(&self, ids: &[u32], dev: &Device) -> Result<Vec<f32>> {
        let head = self
            .contrastive
            .as_ref()
            .context("model has no contrastive head (reconvert the checkpoint)")?;
        let enc = self.encode(ids, dev)?; // (1, T, d) final-normed
        let pooled = enc.mean(1)?; // (1, d)
        let h = pooled.matmul(&head.hidden_w)?; // (1, d/4)  [weight pre-transposed]
        let h = h.broadcast_add(&head.hidden_b)?.relu()?;
        let p = h.matmul(&head.proj_w)?; // (1, cdim)  [weight pre-transposed]
        let v: Vec<f32> = p.squeeze(0)?.to_vec1()?;
        // safe L2 normalize: sqrt(sum^2 + 1e-12), matching the JAX reference
        let norm = (v.iter().map(|x| x * x).sum::<f32>() + 1e-12).sqrt();
        Ok(v.into_iter().map(|x| x / norm).collect())
    }

    fn embed(&self, ids: &[u32], dev: &Device) -> Result<Tensor> {
        let t = ids.len();
        let idx = Tensor::from_vec(ids.to_vec(), (t,), dev)?;
        let x = self.embedding.index_select(&idx, 0)?; // (T, d)
        let x = (x * self.embed_scale)?;
        Ok(x.unsqueeze(0)?) // (1, T, d)
    }

    /// Run the encoder once over the prompt tokens. Returns (1, T, d_model).
    pub fn encode(&self, ids: &[u32], dev: &Device) -> Result<Tensor> {
        let mut x = self.embed(ids, dev)?;
        for layer in &self.enc_layers {
            let normed = zcrms_norm(&x, &layer.norm1)?;
            let attn = layer.self_attn.forward(&normed, &normed, Some(&self.rope), false)?;
            x = (x + (attn * layer.attn_gate as f64)?)?;
        }
        zcrms_norm(&x, &self.enc_final_norm)
    }

    /// Initialize decoder state for a query: precompute cross-attention K/V from
    /// the encoder output once (they are constant across all decoded tokens).
    pub fn init_decode(&self, enc_out: &Tensor) -> Result<KvCache> {
        let n = self.dec_layers.len();
        let mut cross_k_t = Vec::with_capacity(n);
        let mut cross_v = Vec::with_capacity(n);
        for layer in &self.dec_layers {
            let (k_t, v) = layer.cross_attn.cross_kv(enc_out)?;
            cross_k_t.push(k_t);
            cross_v.push(v);
        }
        Ok(KvCache {
            self_k: (0..n).map(|_| None).collect(),
            self_v: (0..n).map(|_| None).collect(),
            cross_k_t,
            cross_v,
            pos: 0,
        })
    }

    /// Advance the decoder by one token, returning logits for the NEXT position:
    /// (vocab,). Runs O(1) work in the sequence length per call (aside from the
    /// tiny KV-cache append) instead of reprocessing the whole prefix.
    pub fn decode_step(&self, cache: &mut KvCache, token: u32, dev: &Device) -> Result<Tensor> {
        let pos = cache.pos;
        let mut x = self.embed(&[token], dev)?; // (1, 1, d)
        for (i, layer) in self.dec_layers.iter().enumerate() {
            let normed = zcrms_norm(&x, &layer.norm1)?;
            let attn = layer.self_attn.self_step(
                &normed,
                &self.rope,
                pos,
                &mut cache.self_k[i],
                &mut cache.self_v[i],
            )?;
            x = (x + (attn * layer.self_gate as f64)?)?;

            let normed = zcrms_norm(&x, &layer.norm2)?;
            let cross =
                layer
                    .cross_attn
                    .cross_step(&normed, &cache.cross_k_t[i], &cache.cross_v[i])?;
            x = (x + (cross * layer.cross_gate as f64)?)?;
        }
        let x = zcrms_norm(&x, &self.dec_final_norm)?; // (1, 1, d)
        let last = x.i((0, 0))?; // (d,)
        // tied output head: logits = h @ E^T  -> (vocab,)
        let logits = self.embedding.matmul(&last.unsqueeze(1)?)?.squeeze(1)?;
        cache.pos += 1;
        Ok(logits)
    }
}
