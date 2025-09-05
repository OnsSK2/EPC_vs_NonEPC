# EPC vs Non-EPC

This project provides a Python script (`epc_scorer.py`) designed to distinguish Energy Performance Certificates (EPC / Energieausweis) from non-EPC documents.

The scoring approach combines:
- OCR with Tesseract
- Structural analysis of tables, logos, and signatures
- Detection of keywords and specific patterns
- TF-IDF similarity
- Numerical and visual content verification

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/OnsSK2/EPC_vs_NonEPC.git
   cd EPC_vs_NonEPC
