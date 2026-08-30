### Codex Pricing (Credit System)
> Last Updated: 30/08/2026

| Model         | Plus messages / 5 h | Input tokens per credit | Cached input / credit | API Key                                                 |
| ------------- | ------------------- | ----------------------- | --------------------- | ------------------------------------------------------- |
| GPT-5.6 Sol   | 10–100              | 10K                     | 100K                  | [Usage-based](https://platform.openai.com/docs/pricing) |
| GPT-5.6 Terra | 25–200              | 20K                     | 200K                  | [Usage-based](https://platform.openai.com/docs/pricing) |
| GPT-5.6 Luna  | 250–2,000           | 200K                    | 2M                    | [Usage-based](https://platform.openai.com/docs/pricing) |

For a much more realistic coding workload such as 20K fresh input + 100K cached + 5K output per turn, a ~500-credit equivalent corresponds to roughly

| Model | Approx turns | Fresh input processed | Cached context processed | Output generated |
| ----- | -----------: | --------------------: | -----------------------: | ---------------: |
| Sol   |          ~91 |                ~1.82M |                    ~9.1M |            ~455K |
| Terra |         ~167 |                ~3.34M |                   ~16.7M |            ~835K |
| Luna  |       ~1,667 |                ~33.3M |                  ~166.7M |            ~8.3M |

#### Example Scenario
- Consider a query with 100k new input token, 400k cached token and 10k output/ reasoning tokens

| Same workload | Credits consumed |
| ------------- | ---------------: |
| **Luna**      |            **1** |
| **Terra**     |           **10** |
| **Sol**       |           **19** |

#### References
- https://learn.chatgpt.com/docs/pricing#credits-overview