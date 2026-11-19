# Source attribution and changes

This repository is an **educational adaptation**, not the official Google Cloud lab.

Original teaching materials and helper reference:
- [GoogleCloudPlatform/generative-ai](https://github.com/GoogleCloudPlatform/generative-ai) — Apache License 2.0.
- [Original multimodal RAG helper](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/retrieval-augmented-generation/utils/intro_multimodal_rag_utils.py).
- Google Cloud Skills Boost lab **Multimodal Retrieval Augmented Generation (RAG) using the Gemini API in Agent Platform**.

Changes in this repository:
- Converted the notebook to a public-safe, output-free, documented example.
- Removed the temporary Qwiklabs project identifier and lab-specific saved output.
- Organized sections and added reproducibility guidance.
- Included a **third-party compatibility utility** for a Google Gen AI SDK workflow, based on the uploaded lab adaptation, not presented as Google's official utility.
- Fixed an `OSError: File name too long` issue caused when interpreting a long text prompt as a filesystem path.

The included `LICENSE` states the Apache 2.0 terms chosen for this repository. Original third-party data and downloaded assets are not republished; obtain and use those items under their own applicable terms. Attribution does not imply endorsement by Google.
