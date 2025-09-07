# epc_scorer_visual.py
import numpy as np
import cv2
import os
from scipy.spatial.distance import cosine

class EPCVisualScorer:
    def __init__(self, reference_image_path):
        """
        Initialise le scoreur EPC avec analyse visuelle exclusive
        Aucune extraction ou analyse de texte n'est utilisée
        """
        if not reference_image_path or not os.path.exists(reference_image_path):
            raise ValueError("Le chemin de l'image de référence est invalide")
            
        self.reference_image_path = reference_image_path
        self.reference_features = self._extract_visual_features(reference_image_path)
        
        print(f"Référence visuelle chargée: {os.path.basename(reference_image_path)}")
        print(f"Score structurel de référence: {self.reference_features['structure_score']:.3f}")
    
    def _load_and_preprocess_image(self, image_path):
        """Charge et prétraite une image pour l'analyse visuelle"""
        if not os.path.exists(image_path):
            return None
            
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Normalisation
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        return gray
    
    def _extract_hog_features_custom(self, gray_image):
        """Implémentation custom de HOG sans scikit-image"""
        # Redimensionner pour uniformité
        resized = cv2.resize(gray_image, (64, 64))
        
        # Calcul des gradients
        gx = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitude et orientation
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180
        
        # Histogramme des orientations (8 bins)
        hog_features = np.zeros(8)
        bin_size = 180 / 8
        
        for i in range(orientation.shape[0]):
            for j in range(orientation.shape[1]):
                bin_idx = int(orientation[i, j] / bin_size) % 8
                hog_features[bin_idx] += magnitude[i, j]
        
        # Normalisation
        hog_features = hog_features / (np.sum(hog_features) + 1e-6)
        
        return hog_features
    
    def _extract_visual_features(self, image_path):
        """Extrait les caractéristiques visuelles sans analyse textuelle"""
        gray = self._load_and_preprocess_image(image_path)
        if gray is None:
            return None
            
        features = {}
        
        # 1. Score structurel global
        features['structure_score'] = self._calculate_structure_score(gray)
        
        # 2. Histogramme des gradients orientés (HOG custom)
        features['hog_features'] = self._extract_hog_features_custom(gray)
        
        # 3. Détection des contours et formes
        features['edge_features'] = self._extract_edge_features(gray)
        
        # 4. Détection des keypoints (ORB)
        features['keypoints'], features['descriptors'] = self._extract_orb_features(gray)
        
        # 5. Analyse de la densité visuelle
        features['density_features'] = self._analyze_visual_density(gray)
        
        # 6. Détection des tableaux et structures
        features['table_features'] = self._detect_tables(gray)
        
        # 7. Détection des signatures visuelles
        features['signature_features'] = self._detect_signature_patterns(gray)
        
        return features
    
    def _calculate_structure_score(self, gray_image):
        """Calcule un score de structure basé sur les caractéristiques visuelles"""
        score = 0.0
        
        # 1. Détection des bordures
        edges = cv2.Canny(gray_image, 50, 150)
        edge_density = np.sum(edges) / (gray_image.shape[0] * gray_image.shape[1])
        if edge_density > 0.1:
            score += 0.2
        
        # 2. Détection des tableaux
        table_score = self._detect_table_structures(gray_image)
        score += table_score * 0.3
        
        # 3. Densité visuelle
        _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        ink_density = 1.0 - (np.sum(binary == 0) / binary.size)
        if ink_density > 0.2:  # Documents denses
            score += 0.2
        
        # 4. Détection de motifs répétitifs (typiques des formulaires)
        horizontal_var = np.var(np.mean(gray_image, axis=1))
        vertical_var = np.var(np.mean(gray_image, axis=0))
        
        if horizontal_var > 1000 or vertical_var > 1000:
            score += 0.2
        
        # 5. Détection de zones de signature
        signature_score = self._detect_signature_zones(gray_image)
        score += signature_score * 0.1
        
        return min(score, 1.0)
    
    def _extract_edge_features(self, gray_image):
        """Extrait des caractéristiques basées sur les contours"""
        edges = cv2.Canny(gray_image, 50, 150)
        
        features = {
            'edge_density': np.sum(edges) / edges.size,
            'horizontal_edges': np.sum(np.diff(edges, axis=0) > 0) / edges.size,
            'vertical_edges': np.sum(np.diff(edges, axis=1) > 0) / edges.size
        }
        
        return features
    
    def _extract_orb_features(self, gray_image):
        """Extrait les keypoints et descripteurs ORB"""
        orb = cv2.ORB_create(nfeatures=500)
        keypoints, descriptors = orb.detectAndCompute(gray_image, None)
        
        if descriptors is None:
            descriptors = np.array([])
            
        return keypoints, descriptors
    
    def _analyze_visual_density(self, gray_image):
        """Analyse la densité et répartition visuelle"""
        _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        
        features = {
            'ink_density': 1.0 - (np.sum(binary == 0) / binary.size),
            'horizontal_density_var': np.var(np.mean(binary == 0, axis=1)),
            'vertical_density_var': np.var(np.mean(binary == 0, axis=0)),
            'quadrant_density': self._analyze_quadrant_density(binary)
        }
        
        return features
    
    def _analyze_quadrant_density(self, binary_image):
        """Analyse la densité par quadrant"""
        h, w = binary_image.shape
        quadrants = [
            binary_image[0:h//2, 0:w//2],  # Top-left
            binary_image[0:h//2, w//2:w],   # Top-right
            binary_image[h//2:h, 0:w//2],   # Bottom-left
            binary_image[h//2:h, w//2:w]    # Bottom-right
        ]
        
        densities = [1.0 - (np.sum(quad == 0) / quad.size) for quad in quadrants]
        return densities
    
    def _detect_tables(self, gray_image):
        """Détecte les structures tabulaires"""
        # Application d'un seuil adaptatif
        thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Détection des contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        table_features = {
            'table_like_contours': 0,
            'average_aspect_ratio': 0,
            'total_table_area': 0
        }
        
        aspect_ratios = []
        table_areas = []
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            
            # Critères pour les tableaux
            if (0.2 < aspect_ratio < 5 and 
                1000 < area < 50000 and 
                w > 50 and h > 50):
                table_features['table_like_contours'] += 1
                aspect_ratios.append(aspect_ratio)
                table_areas.append(area)
        
        if aspect_ratios:
            table_features['average_aspect_ratio'] = np.mean(aspect_ratios)
            table_features['total_table_area'] = np.sum(table_areas) / (gray_image.shape[0] * gray_image.shape[1])
        
        return table_features
    
    def _detect_table_structures(self, gray_image):
        """Détecte les structures de tableau et retourne un score"""
        table_features = self._detect_tables(gray_image)
        score = 0.0
        
        # Seuils réduits pour mieux détecter les tableaux
        if table_features['table_like_contours'] > 2:  # Réduit de 3 à 2
            score += 0.3
        
        if table_features['total_table_area'] > 0.05:  # Réduit de 0.1 à 0.05
            score += 0.2
            
        return min(score, 0.5)
    
    def _detect_signature_zones(self, gray_image):
        """Détecte les zones potentielles de signature"""
        # Recherche de zones avec une densité d'encre élevée
        _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        signature_zones = cv2.findNonZero(binary)
        
        if signature_zones is not None:
            x, y, w, h = cv2.boundingRect(signature_zones)
            aspect_ratio = w / h if h > 0 else 0
            
            # Les signatures ont généralement un ratio aspect spécifique
            if 2.0 < aspect_ratio < 8.0:
                return 0.15
                
        return 0.0
    
    def _detect_signature_patterns(self, gray_image):
        """Détecte les patterns visuels de signature"""
        features = {
            'has_signature_like_pattern': False,
            'signature_zone_density': 0.0,
            'signature_complexity': 0.0
        }
        
        # Détection des zones denses
        _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        non_zero = cv2.findNonZero(binary)
        
        if non_zero is not None:
            x, y, w, h = cv2.boundingRect(non_zero)
            aspect_ratio = w / h if h > 0 else 0
            
            # Caractéristiques des signatures
            zone_density = np.sum(binary[y:y+h, x:x+w] == 0) / (w * h)
            
            if (2.0 < aspect_ratio < 8.0 and zone_density > 0.3):
                features['has_signature_like_pattern'] = True
                features['signature_zone_density'] = zone_density
                
                # Complexité basée sur la variance des gradients
                roi = gray_image[y:y+h, x:x+w]
                if roi.size > 0:
                    gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0)
                    gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1)
                    magnitude = np.sqrt(gx**2 + gy**2)
                    features['signature_complexity'] = np.var(magnitude) / 1000
        
        return features
    
    def _calculate_feature_similarity(self, ref_features, cand_features):
        """Calcule la similarité entre deux sets de features visuelles"""
        if not ref_features or not cand_features:
            return 0.0
            
        similarity = 0.0
        
        # POIDS OPTIMISÉS - Signature augmentée
        weights = {
            'structure': 0.25,  # Réduit de 0.3
            'hog': 0.2,
            'edges': 0.15,
            'density': 0.15,
            'tables': 0.1,
            'signature': 0.15   # Augmenté de 0.1
        }
        
        # 1. Similarité structurelle
        structure_sim = 1.0 - abs(ref_features['structure_score'] - cand_features['structure_score'])
        similarity += weights['structure'] * structure_sim
        
        # 2. Similarité HOG (cosine similarity)
        if len(ref_features['hog_features']) > 0 and len(cand_features['hog_features']) > 0:
            hog_sim = 1.0 - cosine(ref_features['hog_features'], cand_features['hog_features'])
            similarity += weights['hog'] * max(hog_sim, 0)
        
        # 3. Similarité des contours
        edge_sim = 0.0
        for key in ref_features['edge_features']:
            if key in cand_features['edge_features']:
                edge_sim += 1.0 - abs(ref_features['edge_features'][key] - cand_features['edge_features'][key])
        edge_sim /= len(ref_features['edge_features'])
        similarity += weights['edges'] * edge_sim
        
        # 4. Similarité de densité
        density_sim = 0.0
        for key in ['ink_density', 'horizontal_density_var', 'vertical_density_var']:
            if key in ref_features['density_features'] and key in cand_features['density_features']:
                density_sim += 1.0 - abs(ref_features['density_features'][key] - cand_features['density_features'][key])
        
        # Similarité des quadrants
        if 'quadrant_density' in ref_features['density_features'] and 'quadrant_density' in cand_features['density_features']:
            quadrant_sim = sum(1.0 - abs(r - c) for r, c in zip(ref_features['density_features']['quadrant_density'], 
                                                              cand_features['density_features']['quadrant_density'])) / 4
            density_sim += quadrant_sim
            
        density_sim /= 4  # Normalisation
        similarity += weights['density'] * density_sim
        
        # 5. Similarité des tableaux
        table_sim = 0.0
        for key in ['table_like_contours', 'total_table_area']:
            if key in ref_features['table_features'] and key in cand_features['table_features']:
                ref_val = ref_features['table_features'][key]
                cand_val = cand_features['table_features'][key]
                if ref_val + cand_val > 0:
                    table_sim += min(ref_val, cand_val) / max(ref_val, cand_val)
        table_sim /= 2
        similarity += weights['tables'] * table_sim
        
        # 6. Similarité des signatures - AVEC BONUS
        signature_sim = 0.0
        if (ref_features['signature_features']['has_signature_like_pattern'] and 
            cand_features['signature_features']['has_signature_like_pattern']):
            signature_sim += 0.5
            
            # Similarité de densité de signature
            density_sim_sig = 1.0 - abs(ref_features['signature_features']['signature_zone_density'] - 
                                      cand_features['signature_features']['signature_zone_density'])
            signature_sim += 0.3 * density_sim_sig
            
            # Similarité de complexité
            complexity_sim = 1.0 - abs(ref_features['signature_features']['signature_complexity'] - 
                                     cand_features['signature_features']['signature_complexity'])
            signature_sim += 0.2 * complexity_sim
            
            # BONUS ADDITIONNEL POUR SIGNATURES
            if signature_sim > 0.5:
                signature_sim = min(signature_sim + 0.1, 1.0)
            
        similarity += weights['signature'] * signature_sim
        
        return min(similarity, 1.0)
    
    def calculate_visual_similarity(self, candidate_image_path):
        """Calcule la similarité visuelle avec la référence"""
        if not os.path.exists(candidate_image_path):
            return 0.0
            
        # Extraire les features visuelles du candidat
        candidate_features = self._extract_visual_features(candidate_image_path)
        if candidate_features is None:
            return 0.0
        
        # Calculer la similarité
        similarity = self._calculate_feature_similarity(self.reference_features, candidate_features)
        
        # AJUSTEMENT FINAL OPTIMISÉ
        if similarity > 0.65:  # Seuil réduit
            # Boost pour les vrais EPC
            adjusted_similarity = 0.8 + (similarity - 0.65) * 1.5
        else:
            # Réduction pour les non-EPC
            adjusted_similarity = similarity * 0.5
            
        return max(0.0, min(1.0, adjusted_similarity))

def benchmark_visual_epc_scorer(image_files, reference_image_path):
    """Fonction de benchmark pour le scoreur visuel"""
    scorer = EPCVisualScorer(reference_image_path)
    
    results = []
    for image_path in image_files:
        if not os.path.exists(image_path):
            continue
            
        # Déterminer le type basé sur le nom du fichier
        filename = os.path.basename(image_path)
        is_epc = "reference" in filename.lower() or "match" in filename.lower()
        
        # Calculer le score visuel
        score = scorer.calculate_visual_similarity(image_path)
        
        results.append({
            'filename': filename,
            'is_epc': is_epc,
            'score': score,
            'status': '✅' if (is_epc and score >= 0.88) or (not is_epc and score <= 0.6) else '❌'
        })
    
    return results

def print_visual_results_table(results):
    """Affiche les résultats sous forme de tableau"""
    if not results:
        print("Aucun résultat à afficher.")
        return
        
    print("Résultats des tests de scoring visuel EPC")
    print("=" * 70)
    print(f"{'Document':<30} {'Est EPC':<10} {'Score':<10} {'Statut':<10}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['filename']:<30} {str(result['is_epc']):<10} {result['score']:.3f}      {result['status']:<10}")
    
    epc_scores = [r['score'] for r in results if r['is_epc']]
    non_epc_scores = [r['score'] for r in results if not r['is_epc']]
    
    if epc_scores: 
        print(f"\nScore moyen EPC: {sum(epc_scores)/len(epc_scores):.3f}")
    if non_epc_scores: 
        print(f"Score moyen non-EPC: {sum(non_epc_scores)/len(non_epc_scores):.3f}")
    if epc_scores and non_epc_scores:
        separation = abs(sum(epc_scores)/len(epc_scores) - sum(non_epc_scores)/len(non_epc_scores))
        print(f"Séparation des scores: {separation:.3f}")

# Exemple d'utilisation principale
if __name__ == "__main__":
    print("Système de scoring visuel EPC pour documents allemands")
    print("=" * 60)
    print("ANALYSE 100% VISUELLE - AUCUNE EXTRACTION DE TEXTE")
    print("=" * 60)
    
    base_path = r"D:\Desktop\EPC vs nonEPC"
    reference_image_path = os.path.join(base_path, "reference.png")
    test_image_paths = [
        os.path.join(base_path, "reference.png"),
        os.path.join(base_path, "match.png"),
        os.path.join(base_path, "not_epc1.png")
    ]
    
    image_files_exist = all(os.path.exists(path) for path in test_image_paths)
    
    if image_files_exist:
        print("Analyse visuelle des images...")
        results = benchmark_visual_epc_scorer(test_image_paths, reference_image_path)
        print_visual_results_table(results)
        
        print("\n" + "="*60)
        print("TABLEAU DE RÉSULTATS FINAL - ANALYSE VISUELLE")
        print("="*60)
        print(f"{'Document':<40} {'Score':<10} {'Type':<10} {'Statut':<10}")
        print("-"*60)
        
        for result in results:
            doc_type = "EPC" if result['is_epc'] else "Non-EPC"
            print(f"{result['filename']:<40} {result['score']:.3f}      {doc_type:<10} {result['status']:<10}")
    else:
        print("Les fichiers images n'existent pas. Vérifiez les chemins.")