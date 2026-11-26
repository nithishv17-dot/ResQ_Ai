"""
src/features/image_descriptors.py
=================================
Deterministic Visual Descriptor Extractor for ResQ_Ai Phase 2.

Implements `BugBiteDescriptorExtractor`:
1. Color & Inflammation (CIELAB & HSV):
   - Channel-wise mean, standard deviation, and skewness across 6 channels.
   - 16-bin normalized histograms per channel.
   - Clinical erythema/redness index (a* channel) and necrotic/dark center contrast (L* channel).
2. Texture Descriptors (GLCM & LBP):
   - Gray-Level Co-occurrence Matrix (contrast, dissimilarity, homogeneity, energy,
     correlation across distances [1, 3] and angles [0, pi/4, pi/2, 3*pi/4]).
   - Rotation-invariant uniform Local Binary Patterns (P=24, R=3).
3. Keypoint Descriptors (SIFT with ORB Fallback):
   - SIFT 128-dim keypoint detection and descriptors.
   - Robust fallback to ORB when SIFT yields zero keypoints.
   - Fixed-length aggregation via statistical pooling (mean, std, max) + optional BoVW codebook.
4. Lesion Morphology & Geometry:
   - Adaptive/Otsu thresholding on CIELAB a* channel + morphological cleanup.
   - Contour shape metrics: area ratio, circularity, eccentricity, solidity, aspect ratio, extent.
5. Explainable AI (XAI) Biomarkers:
   - Interpretable clinical metrics dictionary (erythema score, circularity, texture contrast, necrosis).
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from scipy import stats
from PIL import Image

try:
    import cv2
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    CV2_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as _cv2_err:
    cv2 = None
    graycomatrix = None
    graycoprops = None
    local_binary_pattern = None
    CV2_AVAILABLE = False

logger = logging.getLogger("BugBiteDescriptorExtractor")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class BugBiteDescriptorExtractor:
    """
    Modular, deterministic visual feature extraction engine for insect bite classification.

    Accepts an RGB image (NumPy array, PIL Image, or file path), standardizes it to
    224x224, and outputs a clean, finite 1D feature vector for downstream classifiers.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        glcm_levels: int = 32,
        glcm_distances: Tuple[int, ...] = (1, 3),
        glcm_angles: Tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
        lbp_points: int = 24,
        lbp_radius: int = 3,
        sift_max_features: int = 200,
        bovw_codebook: Optional[Any] = None,
        bovw_clusters: int = 64,
    ):
        """
        Initializes extractor parameters.

        Args:
            target_size: (width, height) resolution, default (224, 224).
            glcm_levels: Gray levels quantization for GLCM (default 32 for low latency).
            glcm_distances: Co-occurrence pixel distances (default 1, 3).
            glcm_angles: Co-occurrence direction angles (0, 45, 90, 135 deg).
            lbp_points: Uniform LBP circular neighbor points (default 24).
            lbp_radius: Uniform LBP radius (default 3).
            sift_max_features: Upper bound for SIFT keypoints (default 200).
            bovw_codebook: Optional pre-fitted KMeans model for BoVW quantization.
            bovw_clusters: Number of clusters if BoVW is used (default 64).
        """
        self.target_size = target_size
        self.glcm_levels = glcm_levels
        self.glcm_distances = list(glcm_distances)
        self.glcm_angles = list(glcm_angles)
        self.lbp_points = lbp_points
        self.lbp_radius = lbp_radius
        self.sift_max_features = sift_max_features
        self.bovw_codebook = bovw_codebook
        self.bovw_clusters = bovw_clusters

        # SIFT and ORB detectors (if cv2 is installed)
        if CV2_AVAILABLE:
            self.sift = cv2.SIFT_create(nfeatures=self.sift_max_features)
            self.orb = cv2.ORB_create(nfeatures=self.sift_max_features)
        else:
            self.sift = None
            self.orb = None

    # -----------------------------------------------------------------------
    # 1. Standardization & Preprocessing
    # -----------------------------------------------------------------------

    def standardize_image_rgb(self, image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        """
        Loads and standardizes input to a uint8 3-channel RGB image of shape (224, 224, 3).
        """
        if isinstance(image, str):
            bgr = cv2.imread(image)
            if bgr is None:
                raise FileNotFoundError(f"Could not read image from path: {image}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.ndim == 3:
                if image.shape[2] == 4:
                    rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                elif image.shape[2] == 3:
                    rgb = image.copy()
                else:
                    raise ValueError(f"Unsupported channel count: {image.shape[2]}")
            else:
                raise ValueError(f"Unsupported array shape: {image.shape}")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Ensure uint8 [0, 255]
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255.0).astype(np.uint8)
            else:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        # Resize to (width, height)
        if (rgb.shape[1], rgb.shape[0]) != self.target_size:
            rgb = cv2.resize(rgb, self.target_size, interpolation=cv2.INTER_AREA)

        return rgb

    # -----------------------------------------------------------------------
    # 2. Color & Inflammation Descriptors (CIELAB & HSV)
    # -----------------------------------------------------------------------

    def extract_color_inflammation_features(self, rgb: np.ndarray) -> np.ndarray:
        """
        Extracts channel-wise statistical moments (mean, std, skewness),
        16-bin normalized histograms, and clinical erythema / necrotic contrast features.
        """
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)  # L in [0, 255], a in [0, 255], b in [0, 255]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)  # H in [0, 179], S in [0, 255], V in [0, 255]

        channels = [
            lab[:, :, 0],  # L*
            lab[:, :, 1],  # a* (erythema/redness axis)
            lab[:, :, 2],  # b*
            hsv[:, :, 0],  # H
            hsv[:, :, 1],  # S
            hsv[:, :, 2],  # V
        ]
        channel_ranges = [
            (0, 256),
            (0, 256),
            (0, 256),
            (0, 180),
            (0, 256),
            (0, 256),
        ]

        moments = []
        histograms = []

        for ch, (r_min, r_max) in zip(channels, channel_ranges):
            flat = ch.ravel().astype(np.float32)
            # Moments: mean, std, skewness
            c_mean = float(np.mean(flat))
            c_std = float(np.std(flat))
            # Safe skewness calculation
            if c_std > 1e-4:
                c_skew = float(stats.skew(flat))
            else:
                c_skew = 0.0
            moments.extend([c_mean, c_std, c_skew])

            # 16-bin normalized histogram
            hist, _ = np.histogram(flat, bins=16, range=(r_min, r_max), density=False)
            hist = hist.astype(np.float32)
            sum_h = np.sum(hist)
            if sum_h > 0:
                hist /= sum_h
            histograms.extend(hist.tolist())

        # Clinical Erythema & Necrotic Center Contrast
        L_ch = lab[:, :, 0].astype(np.float32)
        a_ch = lab[:, :, 1].astype(np.float32)

        # 1. Erythema metrics (a* channel, OpenCV: 128 is neutral green-red baseline)
        erythema_mean = float(np.mean(a_ch))
        erythema_p90 = float(np.percentile(a_ch, 90))
        # High-redness pixel fraction (a* > 140)
        erythema_ratio_140 = float(np.mean(a_ch > 140.0))
        erythema_ratio_150 = float(np.mean(a_ch > 150.0))

        # 2. Necrotic / Dark Center Contrast (L* channel)
        # Define central region (50% center area: 56 to 168 in 224x224)
        h, w = L_ch.shape
        cy_min, cy_max = h // 4, 3 * h // 4
        cx_min, cx_max = w // 4, 3 * w // 4

        center_L = L_ch[cy_min:cy_max, cx_min:cx_max]
        center_a = a_ch[cy_min:cy_max, cx_min:cx_max]

        # Periphery mask
        periphery_mask = np.ones((h, w), dtype=bool)
        periphery_mask[cy_min:cy_max, cx_min:cx_max] = False
        periphery_L = L_ch[periphery_mask]
        periphery_a = a_ch[periphery_mask]

        center_L_mean = float(np.mean(center_L))
        periphery_L_mean = float(np.mean(periphery_L))
        center_L_min = float(np.min(center_L))

        # Necrotic depression: Periphery L* minus Center L* (higher when center is darker/necrotic)
        necrotic_contrast = periphery_L_mean - center_L_mean
        necrotic_ratio = center_L_mean / (periphery_L_mean + 1e-5)

        # Central erythema delta
        center_a_mean = float(np.mean(center_a))
        periphery_a_mean = float(np.mean(periphery_a))
        erythema_delta = center_a_mean - periphery_a_mean

        clinical_color_metrics = [
            erythema_mean,
            erythema_p90,
            erythema_ratio_140,
            erythema_ratio_150,
            center_L_mean,
            periphery_L_mean,
            center_L_min,
            necrotic_contrast,
            necrotic_ratio,
            erythema_delta,
        ]

        color_vector = np.array(moments + histograms + clinical_color_metrics, dtype=np.float32)
        np.nan_to_num(color_vector, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return color_vector

    # -----------------------------------------------------------------------
    # 3. Texture Descriptors (GLCM & LBP)
    # -----------------------------------------------------------------------

    def extract_texture_features(self, gray: np.ndarray) -> np.ndarray:
        """
        Computes GLCM properties across distances [1, 3] and angles [0, 45, 90, 135 deg],
        plus uniform rotation-invariant LBP histogram with P=24, R=3.
        """
        # Quantize gray levels for low-latency GLCM (32 levels)
        if self.glcm_levels < 256 and gray.max() >= self.glcm_levels:
            quant_factor = max(1, 256 // self.glcm_levels)
            quantized = (gray // quant_factor).astype(np.uint8)
            if quantized.max() >= self.glcm_levels:
                quantized = np.clip(quantized, 0, self.glcm_levels - 1)
        else:
            quantized = gray

        glcm = graycomatrix(
            quantized,
            distances=self.glcm_distances,
            angles=self.glcm_angles,
            levels=self.glcm_levels,
            symmetric=True,
            normed=True,
        )

        glcm_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        glcm_values = []
        for prop in glcm_props:
            matrix = graycoprops(glcm, prop)  # shape: (len(distances), len(angles))
            glcm_values.extend(matrix.flatten().tolist())

        # Uniform Local Binary Patterns (P=24, R=3)
        lbp = local_binary_pattern(gray, P=self.lbp_points, R=self.lbp_radius, method='uniform')
        n_lbp_bins = self.lbp_points + 2  # 26 bins for P=24 uniform
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_lbp_bins, range=(0, n_lbp_bins), density=False)
        lbp_hist = lbp_hist.astype(np.float32)
        sum_lbp = np.sum(lbp_hist)
        if sum_lbp > 0:
            lbp_hist /= sum_lbp

        texture_vector = np.concatenate([np.array(glcm_values, dtype=np.float32), lbp_hist]).astype(np.float32)
        np.nan_to_num(texture_vector, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return texture_vector

    # -----------------------------------------------------------------------
    # 4. Keypoint Descriptors (SIFT with ORB Fallback)
    # -----------------------------------------------------------------------

    def extract_keypoint_descriptors(self, gray: np.ndarray) -> np.ndarray:
        """
        Extracts SIFT keypoint descriptors, falling back to ORB if 0 keypoints are detected.
        Aggregates into a fixed-length vector via statistical pooling (mean, std, max)
        and optional BoVW codebook quantization.
        """
        _, descriptors = self.sift.detectAndCompute(gray, None)

        if descriptors is None or len(descriptors) == 0:
            # Fallback to ORB
            _, orb_desc = self.orb.detectAndCompute(gray, None)
            if orb_desc is not None and len(orb_desc) > 0:
                # ORB descriptors are 32-dim uint8; pad to 128-dim float32 for consistency
                descriptors = np.zeros((len(orb_desc), 128), dtype=np.float32)
                descriptors[:, :32] = orb_desc.astype(np.float32)
            else:
                descriptors = np.empty((0, 128), dtype=np.float32)
        else:
            descriptors = descriptors.astype(np.float32)

        # Statistical Pooling across keypoint descriptors
        if len(descriptors) > 0:
            desc_mean = np.mean(descriptors, axis=0)  # 128
            desc_std = np.std(descriptors, axis=0)    # 128
            desc_max = np.max(descriptors, axis=0)    # 128
            kp_count = float(len(descriptors)) / float(self.sift_max_features)
        else:
            desc_mean = np.zeros(128, dtype=np.float32)
            desc_std = np.zeros(128, dtype=np.float32)
            desc_max = np.zeros(128, dtype=np.float32)
            kp_count = 0.0

        pooled_summary = np.concatenate([desc_mean, desc_std, desc_max, [kp_count]])

        # Optional BoVW Codebook histogram
        if self.bovw_codebook is not None:
            if len(descriptors) > 0:
                cluster_ids = self.bovw_codebook.predict(descriptors)
                bovw_hist = np.bincount(cluster_ids, minlength=self.bovw_clusters).astype(np.float32)
                s = np.sum(bovw_hist)
                if s > 0:
                    bovw_hist /= s
            else:
                bovw_hist = np.zeros(self.bovw_clusters, dtype=np.float32)
            keypoint_vector = np.concatenate([pooled_summary, bovw_hist]).astype(np.float32)
        else:
            keypoint_vector = pooled_summary.astype(np.float32)

        np.nan_to_num(keypoint_vector, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return keypoint_vector

    # -----------------------------------------------------------------------
    # 5. Lesion Morphology & Geometry
    # -----------------------------------------------------------------------

    def extract_morphology_features(self, rgb: np.ndarray) -> np.ndarray:
        """
        Segments the primary inflamed lesion region on the CIELAB a* channel
        using Otsu/adaptive thresholding, and computes geometric contour metrics.
        """
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        a_channel = lab[:, :, 1]

        # Gaussian smoothing to reduce skin pore noise
        blurred = cv2.GaussianBlur(a_channel, (5, 5), 0)

        # Thresholding for elevated erythema
        # Using Otsu thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological opening and closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find external contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_h, img_w = rgb.shape[:2]
        total_area = float(img_h * img_w)

        if contours and len(contours) > 0:
            # Select primary (largest) contour
            primary = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(primary))
            perimeter = float(cv2.arcLength(primary, True))

            # Area ratio
            area_ratio = area / total_area

            # Circularity: 4 * pi * Area / Perimeter^2
            if perimeter > 1e-4:
                circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
                circularity = float(np.clip(circularity, 0.0, 1.0))
            else:
                circularity = 0.0

            # Convex Hull & Solidity
            hull = cv2.convexHull(primary)
            hull_area = float(cv2.contourArea(hull))
            solidity = float(area / (hull_area + 1e-5)) if hull_area > 0 else 0.0
            solidity = float(np.clip(solidity, 0.0, 1.0))

            # Bounding Box & Aspect Ratio / Extent
            bx, by, bw, bh = cv2.boundingRect(primary)
            aspect_ratio = float(bw) / float(bh + 1e-5)
            extent = float(area / (float(bw * bh) + 1e-5))

            # Eccentricity via fitted ellipse
            if len(primary) >= 5:
                try:
                    ellipse = cv2.fitEllipse(primary)
                    (cx, cy), (d1, d2), angle = ellipse
                    major = max(d1, d2)
                    minor = min(d1, d2)
                    if major > 1e-4:
                        eccentricity = float(np.sqrt(1.0 - (minor / major) ** 2))
                    else:
                        eccentricity = 0.0
                except cv2.error:
                    eccentricity = 0.0
            else:
                eccentricity = 0.0
        else:
            # Fallback when no contour is segmented
            area_ratio = 0.0
            circularity = 0.0
            eccentricity = 0.0
            solidity = 0.0
            aspect_ratio = 1.0
            extent = 0.0
            perimeter = 0.0

        morphology_vector = np.array([
            area_ratio,
            circularity,
            eccentricity,
            solidity,
            aspect_ratio,
            extent,
            perimeter / float(img_h + img_w),
            np.log1p(len(contours)),
        ], dtype=np.float32)

        np.nan_to_num(morphology_vector, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return morphology_vector

    # -----------------------------------------------------------------------
    # 6. Combined Feature Vector Extraction
    # -----------------------------------------------------------------------

    def extract_features(self, image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        """
        Extracts the full fixed-length visual descriptor feature vector for an image.

        Combines:
            - Color & Inflammation (moments, histograms, erythema, necrosis)
            - Texture (GLCM 72 dims + LBP 26 dims)
            - Keypoint Summaries (SIFT/ORB pooling + optional BoVW)
            - Lesion Morphology (area, circularity, solidity, eccentricity, etc.)

        Returns:
            np.ndarray: 1D float32 array with zero NaNs/Infs.
        """
        if not CV2_AVAILABLE:
            logger.warning("cv2 is unavailable. Returning zero feature vector.")
            return np.zeros(583, dtype=np.float32)

        rgb = self.standardize_image_rgb(image)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        color_feat = self.extract_color_inflammation_features(rgb)
        texture_feat = self.extract_texture_features(gray)
        keypoint_feat = self.extract_keypoint_descriptors(gray)
        morph_feat = self.extract_morphology_features(rgb)

        combined = np.concatenate([color_feat, texture_feat, keypoint_feat, morph_feat]).astype(np.float32)
        np.nan_to_num(combined, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return combined

    # -----------------------------------------------------------------------
    # 7. Explainable AI (XAI) Biomarkers Extraction
    # -----------------------------------------------------------------------

    def extract_interpretable_metrics(self, image: Union[np.ndarray, Image.Image, str]) -> Dict[str, float]:
        """
        Extracts key interpretable clinical and visual biomarkers for UI/XAI display.

        Returns:
            Dict[str, float] with keys:
                - erythema_score: Normalized redness score [0.0, 1.0].
                - lesion_circularity: Circularity metric [0.0, 1.0] (high for tick bullseye rash).
                - texture_contrast: Average GLCM texture contrast.
                - lesion_area_ratio: Proportion of image covered by inflamed lesion.
                - necrotic_center_contrast: Lightness difference between lesion center and periphery.
                - lesion_solidity: Contour compactness and absence of dendritic edge spreading.
                - lesion_eccentricity: Elongation score [0.0, 1.0].
        """
        if not CV2_AVAILABLE:
            return {
                "erythema_score": 0.0,
                "lesion_circularity": 0.0,
                "texture_contrast": 0.0,
                "lesion_area_ratio": 0.0,
                "necrotic_center_contrast": 0.0,
                "lesion_solidity": 0.0,
                "lesion_eccentricity": 0.0,
            }

        rgb = self.standardize_image_rgb(image)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

        a_ch = lab[:, :, 1].astype(np.float32)
        L_ch = lab[:, :, 0].astype(np.float32)

        # 1. Erythema Score (a* in OpenCV: 128 neutral, > 145 marked erythema)
        mean_a = float(np.mean(a_ch))
        erythema_score = float(np.clip((mean_a - 128.0) / 30.0, 0.0, 1.0))

        # 2. Necrotic Center Contrast
        h, w = L_ch.shape
        center_L = L_ch[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        periphery_mask = np.ones((h, w), dtype=bool)
        periphery_mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = False
        periphery_L = L_ch[periphery_mask]
        necrotic_contrast = float(np.mean(periphery_L) - np.mean(center_L))

        # 3. Texture Contrast (GLCM contrast at distance 1)
        quantized = (gray // max(1, 256 // self.glcm_levels)).astype(np.uint8)
        glcm = graycomatrix(
            quantized,
            distances=[1],
            angles=[0.0, np.pi / 2],
            levels=self.glcm_levels,
            symmetric=True,
            normed=True,
        )
        texture_contrast = float(np.mean(graycoprops(glcm, 'contrast')))

        # 4. Morphology
        morph = self.extract_morphology_features(rgb)
        area_ratio = float(morph[0])
        circularity = float(morph[1])
        eccentricity = float(morph[2])
        solidity = float(morph[3])

        return {
            "erythema_score": round(erythema_score, 4),
            "lesion_circularity": round(circularity, 4),
            "texture_contrast": round(texture_contrast, 4),
            "lesion_area_ratio": round(area_ratio, 4),
            "necrotic_center_contrast": round(necrotic_contrast, 4),
            "lesion_solidity": round(solidity, 4),
            "lesion_eccentricity": round(eccentricity, 4),
        }
