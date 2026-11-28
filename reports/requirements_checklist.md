# Requirement checklist (assignment -> where it is covered)

Status: [x] code/notebook written | [R] needs real run results | [W] needs written report text

| Requirement | Where | Status |
|---|---|---|
| Legal topic, jurisdiction, scope | README; 01-eda (US state residential landlord-tenant/eviction law, 48 jurisdictions incl. DC & PR, 2021 statutes) | [x][W] |
| Dataset, corpus, pretrained model + justification | README resources (reglab/housing_qa, statutes corpus, flan-t5-small, bm25s) | [W] justify choices |
| Seq2seq: question -> short English answer | `modeling/train.py` (generation, greedy, max 4 new tokens) | [x] |
| M1 scratch T5 | notebook 04 | [R] |
| M2 fine-tuned, same split/arch/tokenizer | notebook 04 (shared `train()`, only `init` differs) | [R] |
| M3 RAG: top-k passages + question -> fine-tuned M2 | notebooks 05, 06 | [R] |
| Dataset description, example QA pairs, preprocessing, tokenization, splits | notebooks 01-02; `reports/tables/qa_examples.csv` | [x][W] |
| Document split/index for M3; no test Q/A in index | notebook 03 (chunk 100w/overlap 20, per-state BM25, leak check = 0) | [x][W] |
| Architecture, loss, optimizer, LR, batch, epochs/steps, checkpoint criterion | `reports/tables/training_setup_m1_m2.csv` (notebook 04) | [R] |
| Train/val loss plots for M1 and M2 with selected checkpoint marked + discussion | notebook 04 -> `figures/loss_curves_m1_m2.png` | [R][W] |
| Metrics chosen, how computed, limitations | `evaluation/metrics.py`; write-up | [x][W] |
| M1-M3 on same test set | notebook 06 -> `tables/test_results.csv` | [R] |
| >= 5 analysed questions incl. improvements and remaining errors; M3 error attributed to retrieval/generation/both | notebook 06 | [R][W] |
| Final report assembled from results | notebook 07 | [W] |
| Optional: citations + correctness, reported separately | notebook 06 -> `citation_results.csv`, `citation_manual_annotation.csv` | [R] (manual labelling) |

## Things to state explicitly in the report (known characteristics of the data)
1. **Answers are yes/no.** Token-F1 == EM here; report accuracy, macro-F1, MCC, invalid-output rate, and the majority baseline (test: always "no" = 62.7%).
2. **Test has only 23 distinct question texts** (each asked in ~30 states) -> use cluster-bootstrap CIs; per-group table provided.
3. **Possible near-paraphrase overlap:** 2 of 23 test question texts have TF-IDF cosine > 0.9 with a train question text (6 > 0.8). Exact-match leakage check passes; mention this.
4. **Teammate's `split_summary.csv` / `split_meta.json` "share" column** is computed including `train_aux`, giving 63.1/7.5/7.9%. The real split is 80.4/9.5/10.1% (notebook 02 output). Quote the latter.
5. `docs/` is git-ignored, so README links to `docs/Brief Idea.md` etc. are dead for anyone cloning.
6. M3 uses M2 weights unchanged (as the assignment says), although M2 never saw a context field. The `m3_ft_ablation` variant is optional and must be labelled as an ablation.
7. Seed 42 split: val Yes-rate 29.4% vs train 35.7% vs test 37.3% -> val loss/accuracy and test numbers are not directly comparable.
