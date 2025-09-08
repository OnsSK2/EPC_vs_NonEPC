# EPC vs Non-EPC - Visual Document Analyzer

## 🎯 Objective Achieved ✅

This project successfully addresses the core objective: **Create a scoring function that significantly increases differentiation between EPC and non-EPC documents using exclusive visual analysis.**




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

### ⚖️ Weighted Scoring System
```python
weights = {
    'structure': 0.3,      # Document layout organization
    'complexity': 0.25,    # Visual complexity matching
    'hog': 0.15,          # Gradient pattern similarity  
    'edges': 0.1,         # Edge distribution
    'density': 0.1,       # Ink density patterns
    'tables': 0.05,       # Table structure presence
    'signature': 0.05     # Signature patterns
}

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/OnsSK2/EPC_vs_NonEPC.git
   cd EPC_vs_NonEPC
