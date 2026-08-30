### Core regimes of LLM Inference
```
			   ┌──────────────────────────────────────────────┐
			   │           LLM INFERENCE WORKFLOW             │
			   └──────────────────────┬───────────────────────┘
									  │
			┌─────────────────────────┴─────────────────────────┐
			▼                                                   ▼
┌─────────────────────────┐                         ┌─────────────────────────┐
│      PREFILL PHASE      │                         │      DECODE PHASE       │
├─────────────────────────┤                         ├─────────────────────────┤
│ • Input tokens parallel │                         │ • Autoregressive (1-by-1│
│ • High Arithmetic Int.  │                         │ • Low Arithmetic Int.   │
│ • COMPUTE-BOUND (FLOPs) │                         │ • MEMORY BANDWIDTH-BOUND│
│ • Determines TTFT       │                         │ • Determines TPOT / ITL │
└─────────────────────────┘                         └─────────────────────────┘
```

1. **Prefill (Compute-Bound / GEMM):** All prompt tokens ($S_{in}$) are ingested in parallel. Matrix multiplications achieve high arithmetic intensity ($\text{FLOPs} / \text{Byte}$). Performance is limited by **Tensor Core TFLOPS**.
    
2. **Decode (Memory-Bandwidth-Bound / GEMV):** Tokens are generated one-by-one. For each token, the GPU must stream every weight from High Bandwidth Memory (HBM) into registers to compute a single forward pass. Performance is limited by **HBM Bandwidth (GB/s)**.

### Calculation Options:
```
					  CALCULATOR CONFIGURATION MATRIX
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 1. Model & GPU       ──► Base Model Weights & Hardware Silicon Floor            │
│ 2. Interconnect      ──► Communication Latency & All-Reduce Bottlenecks         │
│ 3. Quantization      ──► Precision Scaling (VRAM Footprint & HBM Read Speeds)   │
│ 4. Workload          ──► Concurrency, Context Windows & KV Cache Sizing         │
│ 5. Parallelism       ──► Multi-GPU Sharding Topology (TP · EP · PP · DP)        │
│ 6. Speculative Dec.  ──► Draft-Verify Pipelining for Sub-Linear Decode Latency  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Interconnects (Intra & inter-nodes)
![[Pasted image 20260830195709.png]]

- **NVLink (3.0/4.0/5.0: 600–1800 GB/s) vs. PCIe (Gen 4/5: 64–128 GB/s):**
    - **Tensor Parallelism (TP)** requires **two $All\text{-}Reduce$ operations per transformer layer** (one for Multi-Head Attention, one for MLP).
    - On PCIe, communication latency often exceeds layer computation time, causing GPU stalls. On NVLink/NVSwitch, communication completes in microseconds, enabling linear scaling

### Parallelism Topology Solver ($TP \times EP \times PP \times DP$)
```
				  CLUSTER TOPOLOGY HIERARCHY
				 
			  ┌───────────────────────────────┐
			  │    Data Parallelism (DP)      │  Independent Replicas
			  └──────────────┬────────────────┘  (Global Throughput)
							 │
			  ┌──────────────▼────────────────┐
			  │   Pipeline Parallelism (PP)   │  Inter-Node Layer Splitting
			  └──────────────┬────────────────┘  (Low Comm Bandwidth / P2P)
							 │
			  ┌──────────────▼────────────────┐
			  │    Tensor Parallelism (TP)    │  Intra-Node Matrix Sharding
			  │    Expert Parallelism (EP)    │  (NVLink All-Reduce / All-to-All)
			  └───────────────────────────────┘
```

- **Expert Parallelism (EP):** For MoE models, assigns individual expert networks to specific GPUs. Replaces $All\text{-}Reduce$ with an $All\text{-}to\text{-}All$ token routing step.
    
- **Pipeline Parallelism (PP):** Cuts model layers across separate nodes when a single model exceeds node VRAM. Incurs a pipeline bubble latency penalty:
    
    $$\text{Bubble Fraction} = \frac{PP - 1}{PP - 1 + M_{microbatches}}$$

| **Configuration Lever**       | **Primary Benefit**                            | **Hardware Sensitivity**                                | **Key Tradeoff / Risk**                                 |
| ----------------------------- | ---------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------- |
| **Tensor Parallelism (TP)**   | Slashes single-user latency (TPOT)             | Requires high intra-node bandwidth (NVLink)             | Stalls and degrades on PCIe interconnects               |
| **Pipeline Parallelism (PP)** | Scales to large model sizes across nodes       | Tolerant of lower bandwidth (PCIe/InfiniBand)           | Introduces idle pipeline bubble latency                 |
#### Tensor Parallelism (TP)
```
       ┌──────────────────────────────────────────────────────────────┐
       │                 TRANSFORMER BLOCK (1 LAYER)                  │
       │                                                              │
       │   [Input Activations]                                        │
       │           │                                                  │
       │   [Column-Parallel Attention]                                │
       │           │                                                  │
       │   [Row-Parallel Attention]                                   │
       │           │                                                  │
       │    ──► ALL-REDUCE #1 ◄──  (Synchronous barrier: hidden_size) │
       │           │                                                  │
       │   [Column-Parallel MLP / Feed-Forward]                       │
       │           │                                                  │
       │   [Row-Parallel MLP / Feed-Forward]                          │
       │           │                                                  │
       │    ──► ALL-REDUCE #2 ◄──  (Synchronous barrier: hidden_size) │
       └──────────────────────────────────────────────────────────────┘
```

##### Communication Characteristics
* Primitives: $2\times\text{ }All\text{-}Reduce$ (or $Reduce\text{-}Scatter + All\text{-}Gather$) per transformer layer.
- Frequency: $2 \times L$ per step (where $L$ is total layer count).
- Communication Volume (Decode Phase per step):

$$\text{Volume}_{TP} = 2 \times \left( \frac{TP - 1}{TP} \right) \cdot 2 \cdot B \cdot d_{model} \text{ bytes per layer}$$

#### Expert Parallelism (EP)

```
						 EXPERT DISPATCH & COMBINE
						 
GPU 0 (Attention)     GPU 1 (Attention)     GPU 2 (Attention)     GPU 3 (Attention)
	│                     │                     │                     │
	└─────────────────────┼─────────────────────┼─────────────────────┘
						  │
		   ──► ALL-TO-ALL DISPATCH (Tokens -► Experts) ◄──
						  │
	┌─────────────────────┼─────────────────────┼─────────────────────┐
	▼                     ▼                     ▼                     ▼
[Expert 0, 1]         [Expert 2, 3]         [Expert 4, 5]         [Expert 6, 7]
(GPU 0 Compute)       (GPU 1 Compute)       (GPU 2 Compute)       (GPU 3 Compute)
	│                     │                     │                     │
	└─────────────────────┼─────────────────────┼─────────────────────┘
						  │
		   ──► ALL-TO-ALL COMBINE (Weights -► Tokens) ◄──
						  │
	▼                     ▼                     ▼                     ▼
GPU 0 Output          GPU 1 Output          GPU 2 Output          GPU 3 Output
```

##### Communication Characteristics

- Primitives: $2\times\text{ }All\text{-}to\text{-}All$ operations per MoE layer (1 token dispatch, 1 token combine).
- Frequency: $2 \times L_{MoE}$ per step (once per MoE routing block).
- Communication Volume:
    
    $$\text{Volume}_{EP} = 2 \times \left( \frac{EP - 1}{EP} \right) \cdot B \cdot k \cdot d_{model} \text{ bytes per MoE layer}$$
    
    _(where $k$ is top-$k$ routed experts, e.g., $k=2$ or $k=8$)._

#### Pipeline Parallelism (PP)
* Pipeline Parallelism partitions the model by layers across consecutive GPUs ($0 \dots PP-1$), streaming activations forward and gradients/latents backward across microbatches.

```
					  PIPELINE STAGE SEQUENCING
					  
Stage 0 (Layers 1-8)        Stage 1 (Layers 9-16)       Stage 2 (Layers 17-24)
 [  GPU 0  ]                 [  GPU 1  ]                 [  GPU 2  ]
	  │                           │                           │
	  │  P2P Send/Recv (Acts)     │  P2P Send/Recv (Acts)     │
	  └──────────────────────────►│──────────────────────────►│
	  │  (Stage Boundary ONLY)    │  (Stage Boundary ONLY)    │
```

##### Communication Characteristics
- Primitives: Point-to-Point ($P2P$) $\text{Send} / \text{Recv}$ between adjacent pipeline stages.
- Frequency: $1 \times \text{per microbatch}$ per stage boundary (only between GPU $i$ and GPU $i+1$).
- Communication Volume:
    $$\text{Volume}_{PP} = B_{micro} \cdot S \cdot d_{model} \text{ bytes per stage boundary}$$

### Speculative Decoding
![[Pasted image 20260830200249.png]]

### Chunked Prefil + PD Disaggregation


---

## Glossary
### All Reduce, All-to-All, P2P
> These algorithms are not network algorithms, rather they are communication contracts which repair or reshuffle tensor layout before next computation

```
Parallelism decides - WHAT must move --> Collective primitive defines - WHO gets what --> comm contract decides HOW it moves --> NVLink carries the bytes
```

#### All Reduce
![[Pasted image 20260831004704.png]]

**Why TP naturally needs "All-Reduce"**
* TP splits matrix across two GPUs and each GPU computes only a partial contribution. Hence, the missing operation is obviously "SUM" 
```
GPU 0: Y₀ ──┐
            ├── SUM ──→ GPU 0: Y
GPU 1: Y₁ ──┘           GPU 1: Y
```

#### All-to-All
![[Pasted image 20260831004958.png]]

Why EP naturally needs "All-to-All"
* EP is a problem which doesn't need summing, rather it requires rearrangement over reduction.

#### Point-to-Point (P2P)
![[Pasted image 20260831005136.png]]

Why PP needs only P2P
* Instead of splitting one math ops among GPUs, it splits execution graph. There is nothing to sum, gather, or globally rearrange. $GPU \\ i+1$ simply needs the output tensor produced by $GPU \\ i$

#### Others
![[Pasted image 20260831005151.png]]
<div align="center"><sub><em>Reduce - Scatter</em></sub></div>

![[Pasted image 20260831005209.png]]
<div align="center"><sub><em>All-Gather</em></sub></div>
