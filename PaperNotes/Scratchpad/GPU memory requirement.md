Weight VRAM = para x bytes per para x 1.15 (runtime overhead)

KV cache = 2 x L (layer cut) x N_kv (no. Of kv cache heads) x d_head (head dim) x ctx_len  x bytes per element

---

### Example: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)

VRAM = 27B x 1 byte (FP8) x 1.15 = 31.05B bytes = 31.05 GBs

KV = 2 x 64 x [16 x 3 ( 16 x 128 ) x 1 x ( 4 x 256 )] x 1 = 12884901888 = 12.88 GBs
- mixed attention structure
- mathematical formula: $$\text{KV}_{\text{total}} = \sum_{i=1}^{k} \left( 2 \times L_i \times N_{\text{kv}, i} \times d_{\text{head}, i} \times S_i \times \text{Bytes}_i \right)$$

Varying Context Length
$$\text{KV}_{\text{Total}}(S) = \underbrace{2 \times L_{\text{local}} \times N_{\text{kv, local}} \times d_{\text{head, local}} \times \min(S, W) \times P_{\text{bytes}}}_{\text{Sliding Window Component}}$$
* The standard formula for KV cache calculation only works for Global and Local attention distribution, and breaks down for qwen 3.8 27b because it uses Gated DeltaNext and Gated Delta