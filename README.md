# NLP Final Project 2026 – Legal QA over Housing Law

Hệ thống hỏi–đáp pháp luật tiếng Anh (sequence-to-sequence) về **luật thuê nhà ở các bang Mỹ (2021)**, so sánh 3 hệ thống trên cùng một test set:

| ID | Hệ thống | Mô tả |
| --- | --- | --- |
| M1 | Transformer train từ đầu | Kiến trúc Flan-T5-small, trọng số khởi tạo ngẫu nhiên |
| M2 | Fine-tune pretrained | `google/flan-t5-small` fine-tune trên cùng tập train với M1 |
| M3 | RAG | BM25 (theo từng bang) → top-k đoạn luật → M2 |
| (Mở rộng) | Trích dẫn | M3 xuất kèm `citation` điều luật căn cứ |

Chi tiết ý tưởng và lộ trình: [`docs/Brief Idea.md`](docs/Brief%20Idea.md), [`docs/Implementation Roadmap.md`](docs/Implementation%20Roadmap.md).

## Tài nguyên

- Dataset: [reglab/housing_qa](https://huggingface.co/datasets/reglab/housing_qa) – subsets `questions`, `questions_aux`, `statutes`
- Pretrained model: [google/flan-t5-small](https://huggingface.co/google/flan-t5-small)
- Retriever: [bm25s](https://github.com/xhluca/bm25s)

## Cấu trúc thư mục

```
NLP-final-2026/
├── README.md
├── requirements.txt
├── pyproject.toml          # cho phép `pip install -e .`
├── docs/                   # đề bài, Brief Idea, Roadmap
├── data/
│   ├── raw/                # HousingQA tải về (không commit)
│   ├── processed/          # train / val / test .jsonl (commit, cả nhóm dùng chung)
│   └── index/              # BM25 index theo bang (không commit, build lại được)
├── notebooks/              # đánh số theo thứ tự chạy
│   ├── 01-eda.ipynb
│   ├── 02-preprocess-split.ipynb
│   ├── 03-build-corpus-index.ipynb
│   └── 04-...              # M1, M2, M3, evaluation
├── src/legal_qa/
│   ├── paths.py            # đường dẫn dùng chung
│   ├── data/               # load, preprocessing, split, chunking, indexing
│   ├── retrieval/          # retriever (BM25, dense...) cho M3
│   ├── modeling/           # M1 / M2 train, RAG generation
│   └── evaluation/         # metric, phân tích lỗi, citation
├── models/                 # checkpoint (không commit, lưu Google Drive)
└── reports/
    ├── figures/            # đường loss, biểu đồ
    └── tables/             # bảng kết quả .csv
```

## Cài đặt

### Local

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Google Colab

```python
!git clone <repo-url> NLP-final-2026
%cd NLP-final-2026
!pip install -q -r requirements.txt
!pip install -q -e .
```

## Thứ tự chạy

1. `01-eda.ipynb` – EDA dataset
2. `02-preprocess-split.ipynb` – định dạng `state: ... question: ...`, chia train/val/test theo `question_group` → `data/processed/`
3. `03-build-corpus-index.ipynb` – chunking kho điều luật, build BM25 index theo bang → `data/index/`
4. `04-train-m1-m2.ipynb` – train M1 (scratch) và M2 (fine-tune), vẽ loss curve
5. `05-retrieval-m3.ipynb` – retrieval, chọn k trên val, RAG input
6. `06-evaluation.ipynb` – so sánh M1–M3 trên test, phân tích lỗi, trích dẫn (mở rộng)
7. `07-report.ipynb` – lắp ráp báo cáo từ các bảng/hình đã lưu

Checklist đối chiếu đề bài: [`reports/requirements_checklist.md`](reports/requirements_checklist.md).

## Dùng dữ liệu cho M1 / M2 / M3

```python
from legal_qa.data import HousingQA
from legal_qa.data.indexing import load_state_index, search

splits = HousingQA().splits()              # {"train", "val", "test"} từ data/processed/*.jsonl
retriever = load_state_index("Alabama")    # BM25 index của bang, từ data/index/
hits = search(retriever, question, k=3)    # top-k chunk kèm statute_idx, citation, text, score
```

- M1 / M2: `input_text` → `target_text` (`yes` / `no`). `max_source_len`, `max_target_len` và thông tin chia tập nằm trong `data/processed/split_meta.json`.
- `statutes` / `gold_statute_idx` chỉ dùng để đánh giá retrieval, phân tích lỗi và trích dẫn, không đưa vào đầu vào mô hình.
- `splits(include_aux=True)` thêm `train_aux.jsonl` vào train (tuỳ chọn); nếu dùng thì M1 và M2 phải dùng như nhau.

## Quy ước

- Không commit dữ liệu lớn, file nén hay checkpoint (xem `.gitignore`).
- Code dùng lại ở nhiều notebook đưa vào `src/legal_qa/`.
- Tên file / thư mục không dùng dấu cách.
- `random_state=42` cho mọi bước có ngẫu nhiên.

## Tham khảo

- Zheng, Guha et al. (2025). *A Reasoning-Focused Legal Retrieval Benchmark*. CS&Law 2025. [arXiv:2505.03970](https://arxiv.org/abs/2505.03970)
- Chung et al. (2022). *Scaling Instruction-Finetuned Language Models*. [arXiv:2210.11416](https://arxiv.org/abs/2210.11416)
- Lù (2024). *BM25S: Orders of magnitude faster lexical search via eager sparsity*. [github.com/xhluca/bm25s](https://github.com/xhluca/bm25s)
