# Data Science, ML & GenAI: Concepts, Senior-Level Mechanics, and Landmark Datasets

A working reference mapping concepts to the datasets and repos the industry actually uses — now merged with senior-level algorithmic depth (split criteria, production architectures, and the specific mechanics behind each technique) so it works both as a learning map and an interview/design-review reference.

---

## 1. Foundational Data Science

| Concept | Why it matters | Famous dataset(s) | Source |
|---|---|---|---|
| Exploratory Data Analysis (EDA) | First step of any project; distributions, correlations, missingness | Titanic, Iris | Kaggle / UCI |
| Data cleaning & missing data handling | Real data is messy; imputation strategy affects everything downstream | Ames Housing, NYC Taxi Trips | Kaggle / NYC OpenData |
| Feature engineering | Often bigger lever than model choice | Ames Housing, Porto Seguro Safe Driver | Kaggle |
| Statistical inference (hypothesis testing, A/B testing) | Used constantly in product/growth analytics | Udacity A/B Test dataset, Marketing A/B Testing | Kaggle |
| Time series decomposition & forecasting | Retail, finance, ops all need this | M5 Forecasting (Walmart), Air Passengers, Electricity Load Diagrams | Kaggle / UCI |
| Survival analysis | Churn, reliability engineering, medicine | Telco Customer Churn, WHAS500 | Kaggle / lifelines examples |
| Causal inference | Increasingly demanded in industry roles | Lalonde dataset, IHDP | causal-inference GitHub repos |
| Data visualization | Communicating findings | Gapminder, Superstore Sales | Kaggle / Tableau public data |

---

## 2. Classical Machine Learning & Tabular Engineering at Scale

Tabular data remains the high-margin core of enterprise decision-making (credit scoring, algo-trading risk, customer lifetime value). Senior-level work here means optimizing for high cardinality, data leakage, and mathematical transparency — not just calling `.fit()`.

| Concept | Why it matters | Senior-level mechanics | Famous dataset(s) | Source |
|---|---|---|---|---|
| Linear/Logistic Regression | Interpretable baseline, still widely deployed | — | Boston Housing (deprecated, use Ames), Pima Indians Diabetes | UCI / Kaggle |
| Decision Trees, Random Forests | Robust tabular baseline | — | Titanic, Adult Census Income | UCI / Kaggle |
| Gradient Boosting (XGBoost, LightGBM, CatBoost) | Dominant on tabular data in production and competitions | **XGBoost** uses second-order Taylor expansion for gradients; **LightGBM** uses Gradient-based One-Side Sampling (GOSS) + Exclusive Feature Bundling for speed on sparse/high-dim data; **CatBoost** uses Symmetric Trees + Ordered Boosting to mitigate target leakage on categorical features natively | Porto Seguro, **IEEE-CIS Fraud Detection**, Home Credit Default Risk | Kaggle |
| SVMs | Still relevant for small/medium high-dim data | — | Breast Cancer Wisconsin, MNIST (small subset) | UCI / Kaggle |
| k-NN & clustering (k-means, DBSCAN, hierarchical) | Unsupervised segmentation | — | Mall Customer Segmentation, Iris, Wholesale Customers | Kaggle / UCI |
| Dimensionality reduction (PCA, t-SNE, UMAP) | Visualization, noise reduction, preprocessing | — | MNIST, Fashion-MNIST, Olivetti Faces | Kaggle / sklearn built-in |
| Anomaly/outlier detection | Fraud, intrusion detection, monitoring | — | Credit Card Fraud Detection (ULB), KDD Cup 1999, NSL-KDD | Kaggle / UCI |
| Recommender systems (collaborative filtering, matrix factorization) | Core to e-commerce/media | — | MovieLens, Amazon Reviews, Netflix Prize | GroupLens / Kaggle |
| Imbalanced classification | Fraud, churn, rare-disease detection | Handling extreme class imbalance (~1.5% positive rate in fraud data) with resampling, class weighting, focal loss | Credit Card Fraud (ULB), Give Me Some Credit, IEEE-CIS Fraud Detection | Kaggle |
| Target leakage & high-cardinality encoding | Silent killer of production model performance | Target Encoding with empirical Bayes smoothing; Weight of Evidence (WoE) for credit-risk scoring; automated feature-interaction extraction without combinatorial explosion | Home Credit Default Risk, IEEE-CIS Fraud Detection | Kaggle |
| Regime shifts & non-stationary time series | Data whose statistical properties change over time | **Temporal Fusion Transformers (TFT)** for multi-horizon forecasting; **DeepAR** for probabilistic autoregressive forecasting; **Purged Group TimeSeries Split** to eliminate structural leakage in backtests | M5 Forecasting (Walmart), Electricity Load Diagrams | Kaggle / UCI |
| Explainable AI (XAI) | Model transparency for stakeholders/regulators | **TreeSHAP** (exact local explanations via conditional expectations) vs. **KernelSHAP** (weighted linear regression approximation); **Integrated Gradients** for differentiable neural architectures | Applied to Adult Census Income, Titanic, credit datasets above | — |
| Model evaluation (cross-validation, ROC-AUC, calibration) | Ensures generalization, avoids leakage | — | Any of the above, used as teaching sets | — |
| Hyperparameter optimization (grid/random/Bayesian, Optuna) | Squeezes performance, standard in pipelines | — | Same tabular sets above used for benchmarking tuners | — |

**Featured competitions/benchmarks:**
- **[M5 Forecasting Competition (Walmart)](https://www.kaggle.com/c/m5-forecasting-accuracy)** — the global benchmark for hierarchical time-series forecasting under intermittent demand.
- **[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)** — massive, highly imbalanced, noisy tabular data with complex cross-feature interactions and real-world masking challenges.
- **Ames Housing Dataset** — the default benchmark for advanced regression, high-cardinality nominal features, and target transformations.

---

## 3. Deep Learning Fundamentals

| Concept | Why it matters | Senior-level mechanics | Famous dataset(s) | Source |
|---|---|---|---|---|
| Feedforward NNs / backpropagation | Foundation of all deep learning | — | MNIST | Yann LeCun / Kaggle |
| CNNs (convolution, pooling) | Backbone of vision models | — | MNIST, Fashion-MNIST, CIFAR-10/100, ImageNet | Kaggle / image-net.org |
| RNNs, LSTM, GRU | Sequence modeling pre-transformer era | — | IMDB Reviews, Penn Treebank | Kaggle / LDC |
| Transformers & attention | Current backbone of NLP, vision, and multimodal models | Multi-head self-attention has O(N²) compute over long sequences | WikiText-103, The Pile | HuggingFace |
| State Space Models (SSMs) vs. self-attention | Long-sequence efficiency is now a first-class design axis | **Mamba** and other SSMs use selective state spaces for sub-quadratic (linear-scaling) memory footprint vs. quadratic attention | Long-sequence benchmarks (LongBench, PG-19) | GitHub / HuggingFace |
| Transfer learning / fine-tuning | Standard practice — rarely train from scratch now | — | ImageNet-pretrained backbones + task datasets | torchvision / timm |
| Self-supervised contrastive learning | Builds cross-modal/robust representations without manual labels | **CLIP** (Contrastive Language-Image Pre-training), **SimCLR**, **MoCo** — map different views/modalities of the same data close together in a shared embedding space | ImageNet, LAION subsets, WIT | GitHub / HuggingFace |
| Regularization (dropout, batch norm, weight decay) | Prevents overfitting in large nets | — | Any of the above used to demo | — |
| Optimization (SGD, Adam, learning rate schedules) | Training stability and convergence speed | — | Same benchmark sets | — |
| Generative models — GANs, VAEs, diffusion (pre-LLM) | Image synthesis foundation | — | CelebA, LSUN Bedrooms, CIFAR-10 | Kaggle / official releases |
| Graph Neural Networks (GNNs) | Fraud rings, social networks, molecules | **GCN** (Graph Convolutional Networks) and **GAT** (Graph Attention Networks) process non-Euclidean data — key for anti-money-laundering (AML) transaction graphs, recsys, molecular bond prediction | Cora, Citeseer, PubMed, **OGB (Open Graph Benchmark)** | GitHub (snap-stanford/ogb) |
| Reinforcement Learning | Robotics, games, recommendation, RLHF foundations | — | OpenAI Gym / Gymnasium environments, Atari 2600, CartPole | GitHub (openai/gym, Farama-Foundation) |

**Featured benchmark:** **[OGB (Open Graph Benchmark)](https://github.com/snap-stanford/ogb)** — curated suite for testing scalable ML on graph-structured data.

---

## 4. Natural Language Processing

| Concept | Why it matters | Famous dataset(s) | Source |
|---|---|---|---|
| Text classification / sentiment analysis | Core enterprise NLP task | IMDB Reviews, Sentiment140, Amazon Reviews | Kaggle / Stanford |
| Named Entity Recognition | Information extraction | CoNLL-2003, OntoNotes 5.0 | GitHub / LDC |
| Question answering | Search, chatbots | **SQuAD 1.1 / 2.0**, Natural Questions, TriviaQA | HuggingFace / Google Research |
| Machine translation | Multilingual products | WMT (various years), Europarl, OPUS | statmt.org / GitHub |
| Language modeling / pretraining corpora | GPT-style pretraining | WikiText-103, BookCorpus, C4, The Pile, RedPajama, FineWeb | HuggingFace |
| Text summarization | Enterprise document workflows | CNN/DailyMail, XSum | HuggingFace |
| Natural language inference | Reasoning benchmarks | SNLI, MultiNLI | NYU / Stanford |
| Word embeddings | Pre-transformer representation learning | Word2Vec (Google News), GloVe (Common Crawl/Wikipedia) | Stanford NLP / Google |
| General NLP benchmarking | Standardized model comparison | GLUE, SuperGLUE | GitHub (nyu-mll) |
| Toxicity / content moderation | Trust & safety | Jigsaw Toxic Comment Classification | Kaggle |
| Spam/email filtering | Classic applied NLP | SMS Spam Collection, Enron Email Dataset | UCI / Kaggle |

**Featured benchmark:** **[SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)** — combines 100k+ questions with 50k+ unanswerable ones, rigorously testing whether a model can express uncertainty instead of hallucinating an answer.

---

## 5. Computer Vision

| Concept | Why it matters | Senior-level mechanics | Famous dataset(s) | Source |
|---|---|---|---|---|
| Image classification | Base CV task | — | ImageNet, CIFAR-10/100, Fashion-MNIST | Kaggle / image-net.org |
| Object detection | Autonomous driving, retail, security | **YOLOv8/v10** (anchor-free, real-time) vs. **Mask R-CNN** (two-stage, higher precision); downstream optimization via pruning, layer fusion, TensorRT compilation for edge deployment | **MS COCO**, Pascal VOC, KITTI | cocodataset.org / GitHub |
| Semantic/instance segmentation | Medical imaging, autonomous driving | Mask R-CNN pixel-level masks | COCO-Stuff, Cityscapes, ADE20K | GitHub |
| Face recognition/verification | Security, personalization | — | CelebA, LFW (Labeled Faces in the Wild) | Kaggle / official sites |
| OCR / document AI | Enterprise document automation | — | IAM Handwriting, SROIE, FUNSD | Kaggle / GitHub |
| Medical imaging | Healthcare AI, high industry demand | — | ChestX-ray14 (NIH), RSNA Pneumonia, ISIC Skin Cancer | Kaggle / NIH |
| Satellite/remote sensing | Agriculture, climate, defense | — | EuroSAT, SpaceNet, BigEarthNet | GitHub / Kaggle |
| Video understanding/action recognition | Media, surveillance | — | Kinetics, UCF101 | DeepMind / GitHub |
| Pose estimation | AR/fitness/sports analytics | — | COCO Keypoints, MPII Human Pose | cocodataset.org |
| Image generation (pre-diffusion) | Historical GAN benchmarks | — | CelebA-HQ, LSUN | GitHub |

**Featured benchmark:** **[MS COCO](https://cocodataset.org/)** — the definitive multi-task vision standard: detection, segmentation, and keypoint tracking across 330k+ real-world images.

---

## 6. Generative AI, LLMs & Agentic Orchestration (Current Industry Frontier)

The paradigm has pivoted from training specialized networks from scratch to architecting runtime systems, state machines, and fine-tuning adapters on top of foundation models.

| Concept | Why it matters | Senior-level mechanics | Famous dataset(s) | Source |
|---|---|---|---|---|
| Large-scale pretraining corpora | What foundation models are trained on | — | Common Crawl, C4, The Pile, RedPajama, **FineWeb / FineWeb-Edu**, Dolma | HuggingFace / GitHub (allenai/dolma) |
| Instruction tuning / SFT | Turns base LLMs into assistants | — | Alpaca, Dolly 15k, OpenAssistant (OASST1/2), FLAN, **OpenHermes 2.5** | HuggingFace |
| RLHF / preference optimization | Aligns model outputs to human preference | **DPO (Direct Preference Optimization)** and **KTO** achieve alignment without the training instability of multi-network PPO; **QLoRA** enables this on quantized 4-bit NormalFloat weights at low cost | Anthropic HH-RLHF, OpenAI Summarize-from-Feedback, **UltraFeedback** | HuggingFace / GitHub (OpenBMB/UltraFeedback) |
| Hardware-aware attention optimization | Determines real-world inference cost and latency | **FlashAttention-2/3** tiles computation to cut memory read/write between GPU SRAM and HBM; **PagedAttention** eliminates KV-cache fragmentation | Benchmarked via inference throughput on standard LLM eval suites | — |
| Retrieval-Augmented Generation (RAG) | Grounding LLMs in external knowledge | Production RAG goes beyond basic vector search: **Hybrid Search** (sparse BM25 + dense bi-encoder fused via Reciprocal Rank Fusion), **Hierarchical/Parent-Child Chunking**, **Cross-Encoder Re-rankers** to filter context windows | Natural Questions, MS MARCO, BEIR benchmark suite | HuggingFace / GitHub (beir-cellar) |
| Embeddings / vector search | RAG backbone, semantic search | — | MTEB (Massive Text Embedding Benchmark) | GitHub (embeddings-benchmark) |
| Code generation & agentic software engineering | Copilot-style tools, autonomous bug-fixing agents | — | HumanEval, MBPP, The Stack, CodeSearchNet, **SWE-bench Pro & Live** | GitHub (openai/human-eval, bigcode-project, princeton-nlp/SWE-bench) |
| Agentic state management & cyclical loops | Moves agents from simple chains to robust cyclical state machines | Frameworks like **LangGraph** and **Dify** manage complex DAG execution, persistent long-term/episodic memory, and robust **Tool Calling** (dynamic parameter parsing + error-recovery loops in sandboxed runtimes) | ToolBench, AgentBench, WebArena, GAIA | GitHub / HuggingFace |
| Multimodal (vision-language) models | GPT-4V/Claude-vision-style systems | Built on CLIP-style contrastive pretraining (see Section 3) | LAION-5B, COCO Captions, Visual Genome, ChartQA | LAION / GitHub |
| Text-to-image diffusion | Stable Diffusion / Midjourney-class models | — | LAION-Aesthetics, LAION-5B | LAION / HuggingFace |
| LLM reasoning benchmarks | Model capability tracking | — | GSM8K, MATH, MMLU, BIG-Bench, ARC (AI2 Reasoning Challenge), HellaSwag | HuggingFace / GitHub |
| Hallucination / factuality evaluation | Reliability of GenAI outputs | — | TruthfulQA, HaluEval | HuggingFace |
| Long-context evaluation | Needle-in-haystack, doc QA at scale | — | LongBench, RULER | GitHub |
| Safety, red-teaming, jailbreak robustness | Responsible deployment | — | AdvBench, JailbreakBench, Anthropic red-team dataset | GitHub |
| Synthetic data generation | Data augmentation when real data is scarce/sensitive | — | Self-Instruct, Evol-Instruct-generated sets | GitHub |
| Fine-tuning efficiency (LoRA, QLoRA, PEFT) | Cheap adaptation of foundation models | QLoRA = quantized LoRA using 4-bit NormalFloat weights, dramatically cutting fine-tuning memory footprint | Alpaca, Dolly, OpenHermes 2.5 | HuggingFace |
| Model quantization/distillation | Deployment on constrained hardware | See AWQ/GPTQ in Section 7 | Benchmarked on MMLU, GSM8K, perplexity suites | — |

**Featured datasets/benchmarks:**
- **[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)** — industry gold standard for large-scale web data, classifier-filtered for educational/reasoning value.
- **[OpenHermes 2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5)** — massive high-quality synthetic instruction data (math, code, logic), widely used to train top open-weight models.
- **[SWE-bench Pro & Live](https://github.com/princeton-nlp/SWE-bench)** — the ultimate agentic eval: autonomously resolving full-scale, multi-file bugs in real open-source codebases.
- **[UltraFeedback](https://github.com/OpenBMB/UltraFeedback)** — comprehensive preference data for reward-model fine-tuning across safety, instruction-following, and truthfulness axes.

---

## 7. MLOps, LLMOps, Responsible AI & Distributed Systems

A model is an unvalidated liability until it's consistently deployed, monitored, and scaled. Senior-level MLOps means owning the full lifecycle, not just the notebook.

| Concept | Why it matters | Senior-level mechanics | Famous dataset(s) / benchmark | Source |
|---|---|---|---|---|
| Fairness & bias auditing | Legal/ethical requirement in regulated industries | — | COMPAS Recidivism, Adult Census Income, German Credit | ProPublica GitHub / UCI |
| Explainability (SHAP, LIME) | Model transparency for stakeholders/regulators | See TreeSHAP vs. KernelSHAP in Section 2 | Applied to tabular sets above (Adult, Titanic) | — |
| Data/feature drift monitoring | Keeps production models reliable | **Population Stability Index (PSI)** for static numerical features; semantic drift tracked via high-dimensional embedding cluster tracking (e.g., **Arize Phoenix**, **W&B Weave**) | Any time-indexed dataset (M5, NYC Taxi) used to simulate drift | Kaggle |
| Distributed training at scale | Enables training beyond single-GPU/single-node limits | **DeepSpeed ZeRO** (Stages 1/2/3) and PyTorch **FSDP** (Fully Sharded Data Parallel) shard model parameters, gradients, and optimizer states across GPU clusters | Benchmarked on large pretraining corpora (Section 6) | GitHub (microsoft/DeepSpeed, pytorch/pytorch) |
| Model compression & quantization | Lowers operational footprint for deployment | **AWQ (Activation-aware Weight Quantization)** and **GPTQ** post-training quantization allow massive models to run on commodity hardware with minimal perplexity loss | Benchmarked on MMLU, GSM8K, perplexity suites | GitHub |
| Automated gated testing / LLM eval in CI/CD | Moves evaluation from manual to deterministic, blocks regressions before merge | Frameworks like **DeepEval** and **Deepchecks** score **Context Recall**, **Faithfulness**, **Answer Relevancy** automatically in pipelines | RAG eval sets (BEIR, Natural Questions) | GitHub |
| Data versioning & lineage | Reproducibility | — | DVC/LakeFS example projects | GitHub |
| A/B testing & experimentation platforms | Product decisions | — | Marketing A/B Testing, Udacity A/B Test | Kaggle |
| Privacy-preserving ML (differential privacy, federated learning) | Regulated data (health, finance) | — | Federated EMNIST, LEAF benchmark | GitHub (TalwalkarLab/leaf) |
| Adversarial robustness | Security-critical deployments | — | Adversarial CIFAR-10/ImageNet-C, RobustBench | GitHub |
| Feature stores | Consistent, low-latency features across offline training and online serving | — | — | **[Feast](https://github.com/feast-dev/feast)** |
| LLM observability / tracing | Debugging step-by-step LLM app execution, cost/latency/quality tracking | — | — | **[Langfuse](https://github.com/langfuse/langfuse)** |
| High-throughput LLM serving | Production inference at scale | Built around PagedAttention (Section 6); outperforms naive transformer inference servers by orders of magnitude | — | **[vLLM](https://github.com/vllm-project/vllm)** |

---

## 8. Key Frameworks & Production Repositories (Quick Reference)

| Repo | Purpose |
|---|---|
| [feast-dev/feast](https://github.com/feast-dev/feast) | Open-source feature store for offline/online feature parity |
| [langfuse/langfuse](https://github.com/langfuse/langfuse) | LLM tracing, cost/latency/quality observability |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | High-throughput LLM serving via PagedAttention |
| [princeton-nlp/SWE-bench](https://github.com/princeton-nlp/SWE-bench) | Agentic software-engineering evaluation |
| [snap-stanford/ogb](https://github.com/snap-stanford/ogb) | Open Graph Benchmark for GNNs |
| [OpenBMB/UltraFeedback](https://github.com/OpenBMB/UltraFeedback) | Preference data for reward-model / DPO training |
| [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) | ZeRO-based distributed training |
| LangGraph / Dify | Agentic state-machine orchestration frameworks |
| `huggingface/datasets`, `openai/human-eval`, `google-research/bert`, `allenai/dolma`, `EleutherAI/lm-evaluation-harness`, `kaggle/kaggle-api` | General-purpose bookmarks for datasets, benchmarks, and evaluation harnesses |

---

### How to use this
- **Learning path**: move top to bottom — classical ML tabular skills transfer directly into deep learning, then NLP/CV, then GenAI/agentic systems, then MLOps.
- **Senior-level signal**: it's not enough to know *that* XGBoost/LightGBM/CatBoost exist — know *why* each wins on a given data shape (dense vs. sparse vs. high-cardinality categorical), and the same goes for FlashAttention vs. vanilla attention, or DPO vs. PPO.
- **Portfolio building**: pick 1–2 datasets per row you haven't touched; recruiters recognize these names instantly (Titanic, ImageNet, SQuAD, MMLU, SWE-bench, etc.), which signals fluency fast.
- **Most repos to bookmark**: `huggingface/datasets`, `openai/human-eval`, `google-research/bert`, `allenai/dolma`, `EleutherAI/lm-evaluation-harness`, `kaggle/kaggle-api`, `feast-dev/feast`, `langfuse/langfuse`, `vllm-project/vllm`, `princeton-nlp/SWE-bench`.

*Note: the dataset/tooling landscape — especially GenAI benchmarks, agentic frameworks, and serving infrastructure — shifts every few months. For the very latest leaderboard-topping benchmarks, check the HuggingFace Open LLM Leaderboard and Papers With Code directly.*
