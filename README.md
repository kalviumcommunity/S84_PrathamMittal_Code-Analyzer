# AI Code Analyzer

---

## 🚀 Project Overview
The **AI Code Analyzer** uses the **Gemini API** (`gemini-1.5-flash` for text, `embedding-001` for embeddings) to analyze code snippets for syntax/logical errors, suggest fixes, and summarize functionality across multiple programming languages. This project fulfills all **18 tasks** from the GEN AI course, showcasing generative AI techniques.

---

## 🌟 Features
- **Prompting**:
  - System/user prompts with RTFC framework
  - Zero-shot, one-shot, multi-shot, dynamic, chain-of-thought (CoT) prompting
- **Evaluation**: Automated pipeline with judge prompt
- **Token Logging**: Estimated token counts
- **Hyperparameters**: Temperature, top P, simulated top K
- **Advanced Features**: Stop sequences, JSON output, function calling
- **Vector Search**: Embeddings, FAISS database, cosine/L2/dot product similarities

---

## 🛠 Technical Stack
- **Language**: Python 3.8+
- **API**: Gemini API (`gemini-1.5-flash`, `embedding-001`)
- **Dependencies**:
  - `google-generativeai`
  - `pytest`
  - `python-dotenv`
  - `numpy`
  - `faiss-cpu`
- **Testing**: Pytest
- **Data**: JSON dataset (`data/eval_dataset.json`)

---

## 📚 Setup Instructions
1. Clone: `https://github.com/kalviumcommunity/S84_PrathamMittal_Code-Analyzer.git`
2. Create venv: `python -m venv venv`
3. Activate: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install: `pip install -r requirements.txt`
5. Configure: Add `GOOGLE_API_KEY=your_key` to `.env`
6. Test: `pytest tests/`
7. Run: `python src/main.py`

---

## 🎥 Video Explanations
Videos (2-5 min) for each task, uploaded to [Google Drive].


---

## 🗂 Development Plan
- **PR 1 (Tasks 1-9)**: README, prompts, evaluation, tokens
- **PR 2 (Tasks 10-12)**: Temperature, top P, top K
- **PR 3 (Tasks 13-15)**: Stop sequence, JSON output, function calling
- **PR 4 (Tasks 16-18)**: Embeddings, FAISS, similarities

---

## 📋 Task Details
1. README: Project documentation
2. Prompts: RTFC framework
3. Zero-Shot: No examples
4. One-Shot: Single example
5. Multi-Shot: Multiple examples
6. Dynamic: Language-specific
7. CoT: Step-by-step reasoning
8. Evaluation: Scoring pipeline
9. Tokens: Estimated counts
10. Temperature: Randomness (0.7)
11. Top P: Nucleus sampling (0.9)
12. Top K: Simulated (0.1 temp)
13. Stop Sequence: Halt at `###`
14. JSON Output: Structured results
15. Function Calling: Code formatting
16. Embeddings: Vector representations
17. Vector DB: FAISS storage
18. Similarities: Cosine, L2, dot product

---

## 🐛 Troubleshooting
- **API Key**: Verify `GOOGLE_API_KEY` in `.env`
- **FAISS**: Ensure `numpy`, `faiss-cpu`
- **Output**: Check `eval_dataset.json`
- **Git**: Verify repository setup

---

## 📬 Contact
Open an issue on [GitHub](https://github.com/kalviumcommunity/S84_PrathamMittal_Code-Analyzer).

---