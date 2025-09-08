import numpy as np
import cv2
import os
from scipy.spatial.distance import cosine

class EPCVisualScorer:
    def __init__(self, reference_image_path):
        """
        Initialize EPC scorer with visual analysis only
        No text extraction or semantic analysis is used
        """
        if not reference_image_path or not os.path.exists(reference_image_path):
            raise ValueError("Reference image path is invalid")
            
        self.reference_image_path = reference_image_path
        self.reference_features = self._extract_visual_features(reference_image_path)
        
        print(f"Visual reference loaded: {os.path.basename(reference_image_path)}")
        print(f"Reference structure score: {self.reference_features['structure_score']:.3f}")
        print(f"Reference complexity: {self.reference_features['complexity_score']:.3f}")
    
    def _load_and_preprocess_image(self, image_path):
        """Load and preprocess image for visual analysis"""
        if not os.path.exists(image_path):
            return None
            
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize for consistency
        height, width = gray.shape
        if max(height, width) > 1000:
            scale = 1000 / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height))
        
        # Normalization
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        return gray
    
    def _extract_hog_features_custom(self, gray_image):
        """Custom HOG implementation without scikit-image"""
        # Resize for consistency
        resized = cv2.resize(gray_image, (64, 64))
        
        # Calculate gradients
        gx = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitude and orientation
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180
        
        # Orientation histogram (8 bins)
        hog_features = np.zeros(8)
        bin_size = 180 / 8
        
        for i in range(orientation.shape[0]):
            for j in range(orientation.shape[1]):
                bin_idx = int(orientation[i, j] / bin_size) % 8
                hog_features[bin_idx] += magnitude[i, j]
        
        # Normalization
        if np.sum(hog_features) > 0:
            hog_features = hog_features / np.sum(hog_features)
        else:
            hog_features = np.zeros(8)
        
        return hog_features
    
    def _calculate_complexity_score(self, gray_image):
        """Calculate document complexity score"""
        score = 0.0
        
        # Edge density
        edges = cv2.Canny(gray_image, 50, 150)
        edge_density = np.sum(edges) / edges.size
        score += min(edge_density * 2, 0.3)
        
        # Texture complexity
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        score += min(np.std(gradient_magnitude) / 1000, 0.3)
        
        # Form structure complexity
        table_score = self._detect_table_structures(gray_image)
        score += table_score * 0.4
        
        return min(score, 1.0)
    
    def _extract_visual_features(self, image_path):
        """Extract visual features without text analysis"""
        gray = self._load_and_preprocess_image(image_path)
        if gray is None:
            return None
            
        features = {}
        
        # 1. Global structure score
        features['structure_score'] = self._calculate_structure_score(gray)
        
        # 2. Complexity score
        features['complexity_score'] = self._calculate_complexity_score(gray)
        
        # 3. Histogram of Oriented Gradients (HOG custom)
        features['hog_features'] = self._extract_hog_features_custom(gray)
        
        # 4. Edge detection and shapes
        features['edge_features'] = self._extract_edge_features(gray)
        
        # 5. Visual density analysis
        features['density_features'] = self._analyze_visual_density(gray)
        
        # 6. Table and structure detection
        features['table_features'] = self._detect_tables(gray)
        
        # 7. Signature pattern detection
        features['signature_features'] = self._detect_signature_patterns(gray)
        
        # 8. Form elements detection
        features['form_features'] = self._detect_form_elements(gray)
        
        # Store grayscale image for SSIM
        features['gray_image'] = gray
        
        return features
    
    def _detect_form_elements(self, gray_image):
        """Detect specific form elements that distinguish EPC from non-EPC"""
        features = {
            'has_checkboxes': False,
            'has_grid_pattern': False,
            'has_footer_elements': False,
        }
        
        # Checkbox detection
        _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        checkbox_count = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect_ratio = w / max(h, 1)
            
            if (10 < area < 200 and 0.7 < aspect_ratio < 1.3):
                checkbox_count += 1
        
        features['has_checkboxes'] = checkbox_count > 2
        features['checkbox_count'] = checkbox_count
        
        # Grid pattern detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, vertical_kernel)
        
        grid_score = (np.sum(horizontal_lines > 0) + np.sum(vertical_lines > 0)) / (2 * gray_image.size)
        features['has_grid_pattern'] = grid_score > 0.01
        features['grid_score'] = grid_score
        
        # Footer elements detection (bottom 15% of image)
        height, width = gray_image.shape
        footer_region = gray_image[int(height*0.85):, :]
        footer_edges = cv2.Canny(footer_region, 50, 150)
        footer_density = np.sum(footer_edges) / footer_edges.size
        features['has_footer_elements'] = footer_density > 0.05
        
        return features
    
    def _calculate_structure_score(self, gray_image):
        """Calculate structure score based on visual features"""
        score = 0.0
        
        # 1. Border detection
        edges = cv2.Canny(gray_image, 50, 150)
        edge_density = np.sum(edges) / (gray_image.shape[0] * gray_image.shape[1])
        score += min(edge_density * 2, 0.2)
        
        # 2. Table detection (more strict)
        table_score = self._detect_table_structures(gray_image)
        score += table_score
        
        # 3. Visual density
        _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        ink_density = 1.0 - (np.sum(binary == 0) / binary.size)
        score += min(ink_density * 0.5, 0.2)
        
        # 4. Form elements detection
        form_features = self._detect_form_elements(gray_image)
        if form_features['has_checkboxes']:
            score += 0.1
        if form_features['has_grid_pattern']:
            score += 0.1
        if form_features['has_footer_elements']:
            score += 0.1
        
        return min(score, 1.0)
    
    def _extract_edge_features(self, gray_image):
        """Extract edge-based features"""
        edges = cv2.Canny(gray_image, 50, 150)
        
        # Calculate edge orientation for better analysis
        gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1)
        magnitude, orientation = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        
        horizontal_edges = np.sum((orientation > 160) | (orientation < 20)) / orientation.size
        vertical_edges = np.sum((orientation > 70) & (orientation < 110)) / orientation.size
        
        features = {
            'edge_density': np.sum(edges) / edges.size,
            'horizontal_edges': horizontal_edges,
            'vertical_edges': vertical_edges,
            'edge_orientation_std': np.std(orientation[edges > 0]) if np.any(edges > 0) else 0
        }
        
        return features
    
    def _analyze_visual_density(self, gray_image):
        """Analyze visual density and distribution"""
        _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        
        features = {
            'ink_density': 1.0 - (np.sum(binary == 0) / binary.size),
            'horizontal_density_var': np.var(np.mean(binary == 0, axis=1)),
            'vertical_density_var': np.var(np.mean(binary == 0, axis=0)),
            'quadrant_density': self._analyze_quadrant_density(binary)
        }
        
        return features
    
    def _analyze_quadrant_density(self, binary_image):
        """Analyze density by quadrants"""
        h, w = binary_image.shape
        quadrants = [
            binary_image[0:h//2, 0:w//2],  # Top-left
            binary_image[0:h//2, w//2:w],   # Top-right
            binary_image[h//2:h, 0:w//2],   # Bottom-left
            binary_image[h//2:h, w//2:w]    # Bottom-right
        ]
        
        densities = [1.0 - (np.sum(quad == 0) / (quad.size + 1e-6)) for quad in quadrants]
        return densities
    
    def _detect_tables(self, gray_image):
        """Detect tabular structures"""
        # Use multiple thresholding methods for better detection
        thresh1 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        thresh2 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        combined = cv2.bitwise_or(thresh1, thresh2)
        
        # Morphological operations to enhance table structures
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        table_features = {
            'table_like_contours': 0,
            'average_aspect_ratio': 0,
            'total_table_area': 0
        }
        
        aspect_ratios = []
        table_areas = []
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / max(h, 1)
            area = w * h
            img_area = gray_image.shape[0] * gray_image.shape[1]
            
            # Stricter criteria for tables
            if (0.3 < aspect_ratio < 4 and 
                area > img_area * 0.005 and area < img_area * 0.5 and 
                w > 40 and h > 20):
                table_features['table_like_contours'] += 1
                aspect_ratios.append(aspect_ratio)
                table_areas.append(area)
        
        if aspect_ratios:
            table_features['average_aspect_ratio'] = np.mean(aspect_ratios)
            table_features['total_table_area'] = np.sum(table_areas) / (img_area + 1e-6)
        
        return table_features
    
    def _detect_table_structures(self, gray_image):
        """Detect table structures and return score"""
        table_features = self._detect_tables(gray_image)
        score = 0.0
        
        if table_features['table_like_contours'] > 3:
            score += 0.4
        
        if table_features['total_table_area'] > 0.08:
            score += 0.3
            
        return min(score, 0.7)
    
    def _detect_signature_patterns(self, gray_image):
        """Detect visual signature patterns"""
        features = {
            'has_signature_like_pattern': False,
            'signature_zone_density': 0.0,
            'signature_complexity': 0.0
        }
        
        # Detect dense zones with more strict criteria
        _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / max(h, 1)
            area = w * h
            
            # Stricter signature criteria
            if (500 < area < 10000 and 2.5 < aspect_ratio < 7.0):
                roi = gray_image[y:y+h, x:x+w]
                if roi.size > 0:
                    # Check complexity
                    gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0)
                    gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1)
                    magnitude = np.sqrt(gx**2 + gy**2)
                    complexity = np.var(magnitude)
                    
                    if complexity > 5000:  # Higher complexity threshold
                        features['has_signature_like_pattern'] = True
                        features['signature_zone_density'] = np.sum(binary[y:y+h, x:x+w] == 255) / (w * h)
                        features['signature_complexity'] = complexity / 10000
                        break
        
        return features
    def _calculate_feature_similarity(self, ref_features, cand_features):
        """Calculate similarity between visual feature sets"""
        if not ref_features or not cand_features:
            return 0.0
            
        similarity = 0.0
        
        # REVISED WEIGHTS - More emphasis on structure and complexity
        weights = {
            'structure': 0.3,
            'complexity': 0.25,
            'hog': 0.15,
            'edges': 0.1,
            'density': 0.1,
            'tables': 0.05,
            'signature': 0.05
        }
        
        # 1. Structural similarity
        structure_sim = 1.0 - abs(ref_features['structure_score'] - cand_features['structure_score'])
        similarity += weights['structure'] * structure_sim
        
        # 2. Complexity similarity
        complexity_sim = 1.0 - abs(ref_features['complexity_score'] - cand_features['complexity_score'])
        similarity += weights['complexity'] * complexity_sim
        
        # 3. HOG similarity (cosine)
        if len(ref_features['hog_features']) > 0 and len(cand_features['hog_features']) > 0:
            try:
                hog_sim = 1.0 - cosine(ref_features['hog_features'], cand_features['hog_features'])
                similarity += weights['hog'] * max(hog_sim, 0)
            except:
                similarity += weights['hog'] * 0.3
        
        # 4. Edge similarity
        edge_sim = 0.0
        edge_count = 0
        for key in ref_features['edge_features']:
            if key in cand_features['edge_features']:
                ref_val = ref_features['edge_features'][key]
                cand_val = cand_features['edge_features'][key]
                edge_sim += 1.0 - min(abs(ref_val - cand_val) / (max(abs(ref_val), abs(cand_val)) + 1e-6), 1.0)
                edge_count += 1
        if edge_count > 0:
            edge_sim /= edge_count
        similarity += weights['edges'] * edge_sim
        
        # 5. Density similarity
        density_sim = 0.0
        density_count = 0
        for key in ['ink_density', 'horizontal_density_var', 'vertical_density_var']:
            if key in ref_features['density_features'] and key in cand_features['density_features']:
                ref_val = ref_features['density_features'][key]
                cand_val = cand_features['density_features'][key]
                density_sim += 1.0 - min(abs(ref_val - cand_val) / (max(abs(ref_val), abs(cand_val)) + 1e-6), 1.0)
                density_count += 1
        
        # Quadrant similarity
        if 'quadrant_density' in ref_features['density_features'] and 'quadrant_density' in cand_features['density_features']:
            quadrant_sim = 0.0
            for r, c in zip(ref_features['density_features']['quadrant_density'], 
                          cand_features['density_features']['quadrant_density']):
                quadrant_sim += 1.0 - min(abs(r - c) / (max(abs(r), abs(c)) + 1e-6), 1.0)
            quadrant_sim /= 4
            density_sim += quadrant_sim
            density_count += 1
            
        if density_count > 0:
            density_sim /= density_count
        similarity += weights['density'] * density_sim
        
        # 6. Table similarity
        table_sim = 0.0
        table_count = 0
        for key in ['table_like_contours', 'total_table_area']:
            if key in ref_features['table_features'] and key in cand_features['table_features']:
                ref_val = ref_features['table_features'][key]
                cand_val = cand_features['table_features'][key]
                if max(ref_val, cand_val) > 0:
                    table_sim += min(ref_val, cand_val) / max(ref_val, cand_val)
                    table_count += 1
        if table_count > 0:
            table_sim /= table_count
        similarity += weights['tables'] * table_sim
        
        # 7. Signature similarity
        signature_sim = 0.0
        if (ref_features['signature_features']['has_signature_like_pattern'] and 
            cand_features['signature_features']['has_signature_like_pattern']):
            signature_sim += 0.4
            
            density_sim_sig = 1.0 - min(abs(ref_features['signature_features']['signature_zone_density'] - 
                                          cand_features['signature_features']['signature_zone_density']), 1.0)
            signature_sim += 0.3 * density_sim_sig
            
            complexity_sim = 1.0 - min(abs(ref_features['signature_features']['signature_complexity'] - 
                                         cand_features['signature_features']['signature_complexity']), 1.0)
            signature_sim += 0.3 * complexity_sim
            
        similarity += weights['signature'] * signature_sim
        
        return min(similarity, 1.0)
    
    def calculate_visual_similarity(self, candidate_image_path):
        """Calculate visual similarity with reference"""
        if not os.path.exists(candidate_image_path):
            return 0.0
            
        # If comparing reference with itself, return perfect score
        if os.path.abspath(candidate_image_path) == os.path.abspath(self.reference_image_path):
            return 1.0
            
        # Extract visual features from candidate
        candidate_features = self._extract_visual_features(candidate_image_path)
        if candidate_features is None:
            return 0.0
        
        # Calculate similarity
        similarity = self._calculate_feature_similarity(self.reference_features, candidate_features)
        
        # ENHANCED FINAL ADJUSTMENT with strict penalties for non-EPC
        complexity_diff = abs(self.reference_features['complexity_score'] - 
                            candidate_features['complexity_score'])
        
        structure_diff = abs(self.reference_features['structure_score'] - 
                           candidate_features['structure_score'])
        
        # Apply strong penalties for significant differences
        if complexity_diff > 0.3:
            similarity *= 0.5  # Strong penalty for complexity mismatch
        elif complexity_diff > 0.15:
            similarity *= 0.7  # Moderate penalty
        
        if structure_diff > 0.4:
            similarity *= 0.4  # Very strong penalty for structure mismatch
        elif structure_diff > 0.2:
            similarity *= 0.6  # Strong penalty
        
        # Additional penalty if candidate lacks key EPC features
        form_features = candidate_features['form_features']
        if not form_features['has_checkboxes'] and not form_features['has_grid_pattern']:
            similarity *= 0.6  # Penalty for missing form elements
        
        return max(0.0, min(1.0, similarity))

def benchmark_visual_epc_scorer(image_files, reference_image_path):
    """Benchmark function for visual scorer with buffer zone handling"""
    scorer = EPCVisualScorer(reference_image_path)
    
    results = []
    for image_path in image_files:
        if not os.path.exists(image_path):
            continue
            
        filename = os.path.basename(image_path)
        
        # Calculate visual score
        score = scorer.calculate_visual_similarity(image_path)
        
        # ✅ CORRECTED: EXPLICIT BUFFER ZONE HANDLING
        if score >= 0.88:
            # EPC ZONE: ≥ 0.88 → Valid EPC classification
            is_epc = True
            status = '✅'
        elif score <= 0.60:
            # NON-EPC ZONE: ≤ 0.60 → Valid Non-EPC classification
            is_epc = False
            status = '✅'
        else:
            # 🚨 BUFFER ZONE: 0.61-0.87 → EXPLICIT REJECTION
            is_epc = False  # Classified as Non-EPC but rejected
            status = '❌'   # Explicit cross mark for buffer zone documents
        
        results.append({
            'filename': filename,
            'is_epc': is_epc,
            'score': score,
            'status': status
        })
    
    return results

def print_visual_results_table(results):
    """Print results in table format"""
    if not results:
        print("No results to display.")
        return
        
    print("EPC Visual Scoring Test Results")
    print("=" * 70)
    print(f"{'Document':<30} {'Is EPC':<10} {'Score':<10} {'Status':<10}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['filename']:<30} {str(result['is_epc']):<10} {result['score']:.3f}      {result['status']:<10}")
    
    epc_scores = [r['score'] for r in results if r['is_epc']]
    non_epc_scores = [r['score'] for r in results if not r['is_epc']]
    
    if epc_scores: 
        print(f"\nAverage EPC score: {sum(epc_scores)/len(epc_scores):.3f}")
    if non_epc_scores: 
        print(f"Average non-EPC score: {sum(non_epc_scores)/len(non_epc_scores):.3f}")
    if epc_scores and non_epc_scores:
        separation = abs(sum(epc_scores)/len(epc_scores) - sum(non_epc_scores)/len(non_epc_scores))
        print(f"Score separation: {separation:.3f}")

# Main execution
if __name__ == "__main__":
    print("German EPC Visual Scoring System")
    print("=" * 60)
    print("100% VISUAL ANALYSIS - STRICT SCORE SEPARATION")
    print("=" * 60)
    
    
    reference_image_path = os.path.join( "reference.png")
    test_image_paths = [
        os.path.join( "reference.png"),
        os.path.join( "match.png"),
        os.path.join( "not_epc1.png")
    ]
    
    image_files_exist = all(os.path.exists(path) for path in test_image_paths)
    
    if image_files_exist:
        print("Performing visual analysis of images...")
        results = benchmark_visual_epc_scorer(test_image_paths, reference_image_path)
        print_visual_results_table(results)
        
        print("\n" + "="*60)
        print("FINAL RESULTS - VISUAL ANALYSIS")
        print("="*60)
        print(f"{'Document':<40} {'Score':<10} {'Type':<10} {'Status':<10}")
        print("-"*60)
        
        for result in results:
            doc_type = "EPC" if result['is_epc'] else "Non-EPC"
            print(f"{result['filename']:<40} {result['score']:.3f}      {doc_type:<10} {result['status']:<10}")
    else:

        print("Image files not found. Please check paths.")
