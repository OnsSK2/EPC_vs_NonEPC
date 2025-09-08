# EPC vs Non-EPC - Visual Document Analyzer

This project provides a Python script (`epc_scorer_visual.py`) designed to distinguish Energy Performance Certificates (EPC / Energieausweis) from non-EPC documents using **exclusive visual analysis**.

## 🎯 Key Feature

**100% Visual Analysis - No Text Extraction**  
The system uses only visual features and layout patterns without any textual content extraction or semantic understanding.

---




### 🚫 No Text Extraction - Pure Visual Features
The system uses **8 visual feature categories** without any text reading:

1. **Structural Analysis** - Document layout and organization scoring
2. **HOG Features** - Histogram of Oriented Gradients for pattern recognition  
3. **Edge Density** - Canny edge detection with orientation analysis
4. **Visual Density** - Ink distribution and quadrant analysis
5. **Table Detection** - Tabular structure identification
6. **Signature Patterns** - Signature and stamp zone detection
7. **Form Elements** - Checkboxes, grids, footer detection
8. **Complexity Scoring** - Overall visual complexity measurement


## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/OnsSK2/EPC_vs_NonEPC.git
   cd EPC_vs_NonEPC
