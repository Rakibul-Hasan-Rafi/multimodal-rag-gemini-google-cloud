# Multimodal RAG with Gemini and Google Cloud

**Guided learning project | Generative AI · Multimodal Retrieval · Document Intelligence**

This repository documents my hands-on exploration of **multimodal retrieval-augmented generation (RAG)** using the Google Gen AI SDK and a shortened set of Alphabet 10-K financial documents. The notebook demonstrates how to extract text and figures from PDFs, represent them with embeddings, retrieve relevant passages or images, and use the retrieved material as context for Gemini-powered answers.

> **Project status:** Educational implementation adapted from a Google Cloud Skills lab, **not original scientific research**. Outputs are excluded from the public notebook; model access, quota and financial data licenses must be respected. For the proposed independent research extension, see the separate EvidenceTrace-MM project (not included in this repository).

## What this project explores

- **Text and visual document processing:** extract PDF text chunks and embedded images, then produce image descriptions.
- **Semantic retrieval:** compare query embeddings with stored text-chunk and image-description embeddings.
- **Image-to-image retrieval:** retrieve related figures from a query table using multimodal embeddings.
- **Comparative reasoning:** supply two stock-performance figures to Gemini for an evidence-oriented comparison.
- **Multimodal RAG:** combine retrieved text and actual figure images in a model request and inspect document/page citations.
- **Cloud API troubleshooting:** handle SDK transitions, invalid message roles, overlong prompt/path confusion, and quota-related failures in an educational setting.

## Pipeline

```text
PDF reports
    ├── extracted text → text chunks → embeddings ─────────┐
    └── extracted figures → image descriptions + embeddings ┤
                                                            ↓
User question / query image → similarity retrieval → selected evidence
                                                            ↓
                                       Gemini multimodal generation
                                                            ↓
                                   answer + retrieved source references
```

**Important:** retrieved page citations alone do not guarantee that every generated claim is supported. This project does not report formal citation correctness, QA accuracy, or generalization benchmarks.

## Repository structure

```text
multimodal-rag-gemini-google-cloud/
├── README.md
├── multimodal_rag_financial_documents.ipynb
├── lab_rag_compat.py
├── requirements.txt
├── .gitignore
├── LICENSE
├── NOTICE.md
└── docs/
    ├── LEARNING_NOTES.md
    └── GITHUB_PORTFOLIO_TEXT.md
```

The original sample PDFs, extracted images, downloaded cloud assets, cached embeddings and `.env` files are intentionally **not committed**.

## Reproducing the workflow (Windows + VS Code / Jupyter)

1. Clone the repository and open the folder in VS Code.
2. Create a Python virtual environment and install dependencies:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
3. Authenticate with your **own authorized** Google Cloud project and configure your API access. This may incur cloud charges; do **not** reuse temporary Google Skills lab credentials after the lab expires.
4. In the VS Code PowerShell terminal, configure the project before launching Jupyter:

   ```powershell
   $env:GOOGLE_CLOUD_PROJECT = "YOUR_GOOGLE_CLOUD_PROJECT_ID"
   $env:GOOGLE_CLOUD_LOCATION = "global"
   $env:GENAI_CLIENT_MODE = "enterprise"
   # Change these only to model IDs available to your project:
   $env:GENERATION_MODEL = "gemini-3.5-flash-lite"
   $env:EMBEDDING_MODEL = "gemini-embedding-2"
   ```

   This notebook defaults to `enterprise=True` to reflect the provided lab adaptation. If you use the standard Vertex AI endpoint and your selected model supports it, set `GENAI_CLIENT_MODE` to `vertexai`. The two modes are not guaranteed to be interchangeable.
5. Obtain the Google sample data in your project directory (requires `gcloud` CLI and access to the source bucket):

   ```powershell
   gcloud storage rsync -x ".*_utils\.py$" -r gs://github-repo/rag/intro_multimodal_rag/intro_multimodal_rag_old_version .
   ```
6. From the repository root, launch Jupyter and open `multimodal_rag_financial_documents.ipynb` (the notebook sits beside `lab_rag_compat.py` and the downloaded `data/` folder):

   ```powershell
   .\.venv\Scripts\python.exe -m jupyter lab
   ```

   In VS Code, select `.venv` as the kernel. If importing `lab_rag_compat` fails, verify that VS Code opened the repository root, not a nested directory.
7. Run the cells in order. Review API quota limits and available generation/embedding models before performing the metadata-extraction cell.

**Reproducibility boundary:** The notebook has been cleaned of execution outputs and cloud-specific identifiers. Its code was statically checked, but this packaged version has **not** been rerun against a live Google Cloud project. Models, API mode, and permissions may require adjustment for your account.

## Questions explored

- What do the documents report about basic and diluted net income per share for Alphabet's stock classes?
- Which financial tables are retrieved when searching by related text or by an example image?
- How do the provided Class A stock-performance charts compare with selected benchmarks?
- Can text and image evidence be combined to answer several financial questions with page references?

## Selected observations from one lab session

- The session logged **23 text chunks** and **11 image records** in the extracted metadata.
- The notebook retrieved financial information and relevant stock-performance images from the sample PDFs.
- The model comparison and retrieval output were inspected interactively; these are **learning observations, not benchmark results**.
- Some API requests encountered `429 RESOURCE_EXHAUSTED`, highlighting limitations of cloud quota-dependent experiments.

## My learning contributions and scope

I worked through and adapted the guided workflow, organized the text/image retrieval and citation examples, used a compatibility helper for the newer Gen AI SDK, and debugged encountered errors. The underlying multimodal RAG tutorial and sample data **originated with Google Cloud**, not with this repository's author.

## Attribution and responsible use

This learning project adapts the [GoogleCloudPlatform/generative-ai](https://github.com/GoogleCloudPlatform/generative-ai) multimodal RAG sample (Apache-2.0). See [NOTICE.md](NOTICE.md) for source links and attribution. Only the local code, documentation, and notebook are shared here; source sample assets must be obtained separately under their applicable terms.

## Limitations and next steps

- No task-level QA benchmark, confidence intervals, or demonstrated improvement over competing RAG systems.
- Accuracy of extraction, image captions, and citations has not been independently audited.
- Model names and endpoints may vary by project access and change over time.
- Future extension: replace the financial tutorial dataset with a license-reviewed scientific-document benchmark and experimentally compare retrieval and evidence-grounding strategies.

**Intended portfolio category:** *Technical Projects / Generative AI Learning*, not *Publications* or *Independent Research*.
