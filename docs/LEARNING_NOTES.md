# Learning notes — multimodal RAG

## What was implemented

A guided financial-document RAG workflow using sample Alphabet 10-K PDFs:
1. Parse text and embedded images with PyMuPDF.
2. Produce text and image embeddings and short visual descriptions with Gemini.
3. Store extracted page/chunk metadata in Python DataFrames.
4. Retrieve text by cosine similarity and retrieve figures by caption and image embeddings.
5. Provide selected text and actual figures to Gemini to compare stock graphs and answer finance questions.
6. Inspect source document/page references printed by the retrieval helper.

## Observed troubleshooting and adaptations

- `404 NotFound` during legacy `TextEmbeddingModel.from_pretrained`: investigated incompatible SDK/model availability and used a modern Gen AI SDK helper.
- `400 INVALID_ARGUMENT` mentioning roles: explicitly passed `role='user'` in model content.
- `OSError [Errno 36] File name too long`: changed the compatibility helper to distinguish long prompts from image-file paths.
- `429 RESOURCE_EXHAUSTED`: recorded quota-limited generation as a blocker rather than claiming success or retrying indefinitely.
- Excluded model outputs from the public notebook to avoid embedding stale results and cloud-session details.

## What the project does NOT show

- A novel multimodal RAG model or research contribution.
- Standardized held-out QA evaluation or valid accuracy scores.
- Guaranteed reproducibility without authorized Google Cloud access.
- Independently verified correctness of every generated financial statement.

## Potential follow-up learning goals

- Add a small, hand-checked set of QA examples with page-level ground truth.
- Compare text-only, caption-based and image-based retrieval quantitatively.
- Add provenance validation beyond printing citations.
- Measure failure modes and API latency transparently.

Keep these follow-ups separate from claims about the current guided exercise until actually implemented.
