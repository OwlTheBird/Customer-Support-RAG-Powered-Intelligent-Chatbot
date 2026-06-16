# Customer-Support-RAG-Powered-Intelligent-Chatbot

This repository contains a fully optimized, evaluated, and tuned Retrieval-Augmented Generation (RAG) pipeline for an intelligent customer support chatbot.

## Project Status & Evaluation Results

### 1. Flask Model Endpoint & Server Setup
- **Status:** ✅ FINISHED
- **Details:** The Flask API (`app.py`) is configured and handles `POST /api/chat` requests, correctly interfacing with the vector database and the Gemini API for response generation.

### 2. Golden Dataset Curation
- **Status:** ✅ FINISHED
- **Details:** Generated a robust dataset of 75 queries (50 in-domain, 25 out-of-domain) with expected contexts and answers. The dataset was optimized to 45 questions for further tuning.
- **Output File:** `endpoint/evaluation/golden_dataset.json`

### 3. Chunking & Retrieval Experimentation
- **Status:** ✅ FINISHED
- **Details:** Implemented and evaluated multiple chunk sizes (e.g., 500/50, 1000/200, 2000/200) and top-k retrieval strategies to identify the most effective context retrieval method.
- **Output File:** `endpoint/evaluation/chunking_results.json`

### 4. Prompt Tuning (A/B Testing)
- **Status:** ✅ FINISHED
- **Details:** Tested three distinct system prompts (Baseline, Detailed, Conversational) across different temperatures (0.1, 0.7) to identify the optimal hyperparameter configuration for the RAG system.
- **Output File:** `endpoint/evaluation/prompt_tuning_results.json`

### 5. Traditional NLP Evaluation (BLEU & ROUGE)
- **Status:** ✅ FINISHED
- **Details:** Evaluated the generated responses against the expected reference answers from the Golden Dataset, outputting precise syntactical overlap metrics.
- **Output File:** `endpoint/evaluation/results.json`

### 6. RAGAS Semantic Evaluation (Faithfulness, Answer Relevancy, Context Precision)
- **Status:** ✅ FINISHED
- **Details:** The evaluation utilized `gemini-3.1-flash-lite` to perform semantic LLM-as-a-judge evaluations using the RAGAS framework.
- **Results:**
  - **Faithfulness (mean):** 0.7973 (PASS - Target > 0.7)
  - **Context Precision:** 1.0000 (PASS - Target > 0.8)
  - **Answer Relevancy:** 0.0000 (FAIL - See limitation below)
- **Limitation Noted:** The Answer Relevancy metric failed exclusively because RAGAS requires generating multiple candidates (`n>1`), which is not supported by the `gemini-3.1-flash-lite` API.
- **Output File:** `endpoint/evaluation/ragas_results.json`

### 7. Comprehensive Evaluation Report Generation
- **Status:** ✅ FINISHED
- **Details:** Compiled an extensive Markdown evaluation report outlining all metrics, findings, and configurations for the entire RAG pipeline.
- **Output File:** `endpoint/docs/evaluation_report.md`