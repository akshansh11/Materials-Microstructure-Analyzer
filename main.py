import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage, spatial
from skimage import (
    filters, segmentation, morphology, measure, feature,
    restoration, exposure, util
)
from sklearn.cluster import KMeans
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class MicrostructureAnalyzer:
    def __init__(self):
        self.image = None
        self.segmented = None
        self.labeled_image = None
        self.properties = {}
        
    def load_image(self, uploaded_file):
        """Load and preprocess image"""
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            if image.mode != 'L':
                image = image.convert('L')
            self.image = np.array(image)
            return True
        return False
    
    def apply_preprocessing(self, method, **kwargs):
        """Apply various preprocessing techniques"""
        if self.image is None:
            return None
            
        if method == "Gaussian Blur":
            return filters.gaussian(self.image, sigma=kwargs.get('sigma', 1))
        elif method == "Median Filter":
            return filters.median(self.image, morphology.disk(kwargs.get('size', 3)))
        elif method == "Bilateral Filter":
            return restoration.denoise_bilateral(self.image, 
                                               sigma_color=kwargs.get('sigma_color', 0.1),
                                               sigma_spatial=kwargs.get('sigma_spatial', 15))
        elif method == "CLAHE":
            return exposure.equalize_adapthist(self.image)
        else:
            return self.image
    
    def segment_image(self, method, preprocessed_image, **kwargs):
        """Apply various segmentation algorithms"""
        if method == "Otsu Thresholding":
            threshold = filters.threshold_otsu(preprocessed_image)
            return preprocessed_image > threshold
            
        elif method == "Multi-Otsu":
            thresholds = filters.threshold_multiotsu(preprocessed_image, 
                                                   classes=kwargs.get('classes', 3))
            return np.digitize(preprocessed_image, bins=thresholds)
            
        elif method == "Adaptive Thresholding":
            block_size = kwargs.get('block_size', 35)
            if block_size % 2 == 0:
                block_size += 1
            img_uint8 = (preprocessed_image * 255).astype(np.uint8)
            return cv2.adaptiveThreshold(img_uint8, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, block_size, 2) > 0
            
        elif method == "Watershed":
            # Create markers for watershed
            distance = ndimage.distance_transform_edt(preprocessed_image > filters.threshold_otsu(preprocessed_image))
            coords = feature.peak_local_maxima(distance, min_distance=kwargs.get('min_distance', 20))
            mask = np.zeros(distance.shape, dtype=bool)
            mask[tuple(coords)] = True
            markers, _ = ndimage.label(mask)
            return segmentation.watershed(-distance, markers, mask=preprocessed_image > filters.threshold_otsu(preprocessed_image))
            
        elif method == "K-means Clustering":
            pixel_values = preprocessed_image.reshape((-1, 1))
            pixel_values = np.float32(pixel_values)
            k = kwargs.get('k', 3)
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(pixel_values)
            return labels.reshape(preprocessed_image.shape)
            
        elif method == "Region Growing":
            # Simple region growing implementation
            threshold = kwargs.get('threshold', 0.1)
            seed_point = kwargs.get('seed_point', (preprocessed_image.shape[0]//2, preprocessed_image.shape[1]//2))
            return self._region_growing(preprocessed_image, seed_point, threshold)
            
        elif method == "Active Contours (Snake)":
            # Simplified active contour using morphological operations
            binary = preprocessed_image > filters.threshold_otsu(preprocessed_image)
            return morphology.binary_closing(binary, morphology.disk(5))
            
        return preprocessed_image > filters.threshold_otsu(preprocessed_image)
    
    def _region_growing(self, image, seed, threshold):
        """Simple region growing implementation"""
        h, w = image.shape
        visited = np.zeros_like(image, dtype=bool)
        result = np.zeros_like(image, dtype=bool)
        
        stack = [seed]
        seed_value = image[seed]
        
        while stack:
            y, x = stack.pop()
            if visited[y, x]:
                continue
                
            if abs(image[y, x] - seed_value) <= threshold:
                visited[y, x] = True
                result[y, x] = True
                
                # Add neighbors
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                        stack.append((ny, nx))
        
        return result
    
    def calculate_properties(self):
        """Calculate all microstructure properties"""
        if self.segmented is None:
            return
            
        # Label connected components
        self.labeled_image = measure.label(self.segmented.astype(int), connectivity=2)
        regions = measure.regionprops(self.labeled_image)
        
        self.properties = {}
        
        # Geometric Properties
        self._calculate_geometric_properties(regions)
        
        # Spatial Distribution Properties
        self._calculate_spatial_properties(regions)
        
        # Connectivity and Topology
        self._calculate_connectivity_properties()
        
        # Statistical Properties
        self._calculate_statistical_properties(regions)
    
    def _calculate_geometric_properties(self, regions):
        """Calculate geometric properties"""
        if len(regions) == 0:
            return
            
        areas = [r.area for r in regions]
        perimeters = [r.perimeter for r in regions]
        aspect_ratios = [r.major_axis_length / max(r.minor_axis_length, 1e-6) for r in regions]
        equivalent_diameters = [r.equivalent_diameter for r in regions]
        circularities = [4 * np.pi * r.area / max(r.perimeter**2, 1e-6) for r in regions]
        elongations = [1 - r.minor_axis_length / max(r.major_axis_length, 1e-6) for r in regions]
        
        # Area fraction
        total_area = self.image.shape[0] * self.image.shape[1]
        phase_area = np.sum(self.segmented)
        area_fraction = phase_area / total_area
        
        self.properties['geometric'] = {
            'area_fraction': area_fraction,
            'areas': areas,
            'perimeters': perimeters,
            'aspect_ratios': aspect_ratios,
            'equivalent_diameters': equivalent_diameters,
            'circularities': circularities,
            'elongations': elongations,
            'mean_area': np.mean(areas),
            'std_area': np.std(areas),
            'mean_aspect_ratio': np.mean(aspect_ratios),
            'mean_circularity': np.mean(circularities)
        }
    
    def _calculate_spatial_properties(self, regions):
        """Calculate spatial distribution properties"""
        if len(regions) < 2:
            return
            
        # Get centroids
        centroids = np.array([r.centroid for r in regions])
        
        # Nearest neighbor distances
        distances = spatial.distance_matrix(centroids, centroids)
        np.fill_diagonal(distances, np.inf)
        nearest_distances = np.min(distances, axis=1)
        
        # Orientation distribution
        orientations = [r.orientation for r in regions]
        
        # Homogeneity index (coefficient of variation of nearest neighbor distances)
        homogeneity = np.std(nearest_distances) / max(np.mean(nearest_distances), 1e-6)
        
        self.properties['spatial'] = {
            'nearest_neighbor_distances': nearest_distances,
            'mean_nearest_distance': np.mean(nearest_distances),
            'std_nearest_distance': np.std(nearest_distances),
            'orientations': orientations,
            'homogeneity_index': homogeneity,
            'centroids': centroids
        }
    
    def _calculate_connectivity_properties(self):
        """Calculate connectivity and topology properties"""
        # Euler number
        euler_number = measure.euler_number(self.segmented)
        
        # Interface area (perimeter of segmented regions)
        interface_area = np.sum(filters.sobel(self.segmented.astype(float)) > 0)
        
        # Connectivity analysis
        connected_components = measure.label(self.segmented, connectivity=2)
        num_components = np.max(connected_components)
        
        # Simple tortuosity estimate
        if np.sum(self.segmented) > 0:
            distance_transform = ndimage.distance_transform_edt(self.segmented)
            skeleton = morphology.skeletonize(self.segmented)
            tortuosity = np.sum(skeleton) / max(np.sum(self.segmented), 1)
        else:
            tortuosity = 0
        
        self.properties['connectivity'] = {
            'euler_number': euler_number,
            'interface_area': interface_area,
            'num_connected_components': num_components,
            'tortuosity': tortuosity,
            'connectivity_ratio': num_components / max(np.sum(self.segmented > 0), 1)
        }
    
    def _calculate_statistical_properties(self, regions):
        """Calculate statistical properties"""
        if len(regions) == 0:
            return
            
        areas = [r.area for r in regions]
        
        # Basic statistics
        mean_size = np.mean(areas)
        std_size = np.std(areas)
        skewness = self._calculate_skewness(areas)
        
        # Texture analysis using local binary patterns
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(self.image, P=8, R=1, method='uniform')
        texture_variance = np.var(lbp)
        
        # Fractal dimension estimation using box counting
        fractal_dim = self._calculate_fractal_dimension(self.segmented)
        
        self.properties['statistical'] = {
            'mean_size': mean_size,
            'std_size': std_size,
            'size_skewness': skewness,
            'texture_variance': texture_variance,
            'fractal_dimension': fractal_dim,
            'size_distribution': areas
        }
    
    def _calculate_skewness(self, data):
        """Calculate skewness of data"""
        if len(data) < 2:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_fractal_dimension(self, binary_image):
        """Estimate fractal dimension using box counting"""
        def _box_count(image, box_size):
            h, w = image.shape
            boxes = 0
            for i in range(0, h, box_size):
                for j in range(0, w, box_size):
                    if np.any(image[i:i+box_size, j:j+box_size]):
                        boxes += 1
            return boxes
        
        sizes = [2, 4, 8, 16, 32]
        counts = []
        for size in sizes:
            if size < min(binary_image.shape):
                counts.append(_box_count(binary_image, size))
        
        if len(counts) > 1:
            coeffs = np.polyfit(np.log(sizes[:len(counts)]), np.log(counts), 1)
            return -coeffs[0]
        return 0

def main():
    st.set_page_config(
        page_title="Microstructure Image Analysis",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("Microstructure Image Analysis Tool")
    st.markdown("Advanced image segmentation and property extraction for materials science")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = MicrostructureAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Sidebar for controls
    st.sidebar.header("Controls")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Microstructure Image",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp']
    )
    
    if uploaded_file is not None:
        if analyzer.load_image(uploaded_file):
            st.sidebar.success("Image loaded successfully!")
            
            # Preprocessing options
            st.sidebar.subheader("Preprocessing")
            preprocess_method = st.sidebar.selectbox(
                "Preprocessing Method",
                ["None", "Gaussian Blur", "Median Filter", "Bilateral Filter", "CLAHE"]
            )
            
            # Preprocessing parameters
            preprocess_params = {}
            if preprocess_method == "Gaussian Blur":
                preprocess_params['sigma'] = st.sidebar.slider("Sigma", 0.5, 5.0, 1.0, 0.1)
            elif preprocess_method == "Median Filter":
                preprocess_params['size'] = st.sidebar.slider("Filter Size", 3, 15, 5, 2)
            elif preprocess_method == "Bilateral Filter":
                preprocess_params['sigma_color'] = st.sidebar.slider("Sigma Color", 0.01, 0.5, 0.1, 0.01)
                preprocess_params['sigma_spatial'] = st.sidebar.slider("Sigma Spatial", 5, 50, 15, 5)
            
            # Apply preprocessing
            if preprocess_method == "None":
                preprocessed = analyzer.image
            else:
                preprocessed = analyzer.apply_preprocessing(preprocess_method, **preprocess_params)
            
            # Segmentation options
            st.sidebar.subheader("Segmentation")
            segmentation_method = st.sidebar.selectbox(
                "Segmentation Method",
                ["Otsu Thresholding", "Multi-Otsu", "Adaptive Thresholding", 
                 "Watershed", "K-means Clustering", "Region Growing", "Active Contours (Snake)"]
            )
            
            # Segmentation parameters
            seg_params = {}
            if segmentation_method == "Multi-Otsu":
                seg_params['classes'] = st.sidebar.slider("Number of Classes", 2, 5, 3)
            elif segmentation_method == "Adaptive Thresholding":
                seg_params['block_size'] = st.sidebar.slider("Block Size", 3, 51, 35, 2)
            elif segmentation_method == "Watershed":
                seg_params['min_distance'] = st.sidebar.slider("Min Distance", 5, 50, 20)
            elif segmentation_method == "K-means Clustering":
                seg_params['k'] = st.sidebar.slider("Number of Clusters", 2, 8, 3)
            elif segmentation_method == "Region Growing":
                seg_params['threshold'] = st.sidebar.slider("Threshold", 0.01, 0.5, 0.1, 0.01)
            
            # Apply segmentation
            if st.sidebar.button("Apply Segmentation"):
                analyzer.segmented = analyzer.segment_image(segmentation_method, preprocessed, **seg_params)
                analyzer.calculate_properties()
                st.sidebar.success("Segmentation and analysis complete!")
            
            # Main content area
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(analyzer.image, cmap='gray')
                ax.set_title("Original Microstructure")
                ax.axis('off')
                st.pyplot(fig)
                
                if preprocess_method != "None":
                    st.subheader("Preprocessed Image")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(preprocessed, cmap='gray')
                    ax.set_title(f"Preprocessed ({preprocess_method})")
                    ax.axis('off')
                    st.pyplot(fig)
            
            with col2:
                if analyzer.segmented is not None:
                    st.subheader("Segmented Image")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    if segmentation_method in ["Multi-Otsu", "K-means Clustering", "Watershed"]:
                        ax.imshow(analyzer.segmented, cmap='tab10')
                    else:
                        ax.imshow(analyzer.segmented, cmap='binary')
                    ax.set_title(f"Segmented ({segmentation_method})")
                    ax.axis('off')
                    st.pyplot(fig)
                    
                    if analyzer.labeled_image is not None:
                        st.subheader("Labeled Components")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.imshow(analyzer.labeled_image, cmap='nipy_spectral')
                        ax.set_title("Connected Components")
                        ax.axis('off')
                        st.pyplot(fig)
            
            # Properties display
            if analyzer.properties:
                st.header("Extracted Properties")
                
                # Create tabs for different property categories
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Geometric Properties", 
                    "Spatial Distribution", 
                    "Connectivity & Topology", 
                    "Statistical Properties"
                ])
                
                with tab1:
                    if 'geometric' in analyzer.properties:
                        geom = analyzer.properties['geometric']
                        
                        # Key metrics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Area Fraction", f"{geom['area_fraction']:.3f}")
                        col2.metric("Mean Area", f"{geom['mean_area']:.1f}")
                        col3.metric("Mean Aspect Ratio", f"{geom['mean_aspect_ratio']:.2f}")
                        col4.metric("Mean Circularity", f"{geom['mean_circularity']:.3f}")
                        
                        # Distributions
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Area distribution
                            fig = px.histogram(
                                x=geom['areas'], 
                                nbins=20,
                                title="Grain Size Distribution",
                                labels={'x': 'Area (pixels²)', 'y': 'Frequency'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Aspect ratio distribution
                            fig = px.histogram(
                                x=geom['aspect_ratios'], 
                                nbins=20,
                                title="Aspect Ratio Distribution",
                                labels={'x': 'Aspect Ratio', 'y': 'Frequency'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Shape factors
                        shape_data = pd.DataFrame({
                            'Circularity': geom['circularities'],
                            'Elongation': geom['elongations'],
                            'Equivalent Diameter': geom['equivalent_diameters']
                        })
                        
                        fig = px.scatter_matrix(
                            shape_data,
                            title="Shape Factor Correlations",
                            height=600
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    if 'spatial' in analyzer.properties:
                        spatial = analyzer.properties['spatial']
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Mean Nearest Distance", f"{spatial['mean_nearest_distance']:.2f}")
                        col2.metric("Distance Std Dev", f"{spatial['std_nearest_distance']:.2f}")
                        col3.metric("Homogeneity Index", f"{spatial['homogeneity_index']:.3f}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Nearest neighbor distribution
                            fig = px.histogram(
                                x=spatial['nearest_neighbor_distances'],
                                nbins=20,
                                title="Nearest Neighbor Distance Distribution",
                                labels={'x': 'Distance (pixels)', 'y': 'Frequency'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Orientation distribution
                            fig = px.histogram(
                                x=np.degrees(spatial['orientations']),
                                nbins=20,
                                title="Grain Orientation Distribution",
                                labels={'x': 'Orientation (degrees)', 'y': 'Frequency'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Spatial correlation plot
                        if len(spatial['centroids']) > 1:
                            centroids = spatial['centroids']
                            fig = px.scatter(
                                x=centroids[:, 1], 
                                y=centroids[:, 0],
                                title="Grain Centroid Distribution",
                                labels={'x': 'X Position', 'y': 'Y Position'}
                            )
                            fig.update_yaxis(autorange="reversed")
                            st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    if 'connectivity' in analyzer.properties:
                        conn = analyzer.properties['connectivity']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Euler Number", f"{conn['euler_number']}")
                        col2.metric("Interface Area", f"{conn['interface_area']}")
                        col3.metric("Connected Components", f"{conn['num_connected_components']}")
                        col4.metric("Tortuosity", f"{conn['tortuosity']:.3f}")
                        
                        # Connectivity visualization
                        connectivity_data = {
                            'Property': ['Euler Number', 'Interface Area', 'Components', 'Tortuosity'],
                            'Value': [
                                conn['euler_number'], 
                                conn['interface_area'], 
                                conn['num_connected_components'], 
                                conn['tortuosity']
                            ]
                        }
                        
                        fig = px.bar(
                            connectivity_data,
                            x='Property',
                            y='Value',
                            title="Connectivity and Topology Metrics"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab4:
                    if 'statistical' in analyzer.properties:
                        stats = analyzer.properties['statistical']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Mean Size", f"{stats['mean_size']:.1f}")
                        col2.metric("Size Std Dev", f"{stats['std_size']:.1f}")
                        col3.metric("Size Skewness", f"{stats['size_skewness']:.3f}")
                        col4.metric("Fractal Dimension", f"{stats['fractal_dimension']:.3f}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Size distribution with statistics
                            fig = px.histogram(
                                x=stats['size_distribution'],
                                nbins=30,
                                title="Detailed Size Distribution with Statistics",
                                labels={'x': 'Size (pixels²)', 'y': 'Frequency'}
                            )
                            
                            # Add vertical lines for mean and std
                            fig.add_vline(
                                x=stats['mean_size'], 
                                line_dash="dash", 
                                line_color="red",
                                annotation_text="Mean"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Box plot for size distribution
                            fig = px.box(
                                y=stats['size_distribution'],
                                title="Size Distribution Box Plot",
                                labels={'y': 'Size (pixels²)'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistical summary
                        st.subheader("Statistical Summary")
                        summary_stats = pd.DataFrame({
                            'Metric': ['Count', 'Mean', 'Std Dev', 'Min', '25%', '50%', '75%', 'Max', 'Skewness'],
                            'Value': [
                                len(stats['size_distribution']),
                                np.mean(stats['size_distribution']),
                                np.std(stats['size_distribution']),
                                np.min(stats['size_distribution']),
                                np.percentile(stats['size_distribution'], 25),
                                np.median(stats['size_distribution']),
                                np.percentile(stats['size_distribution'], 75),
                                np.max(stats['size_distribution']),
                                stats['size_skewness']
                            ]
                        })
                        st.table(summary_stats)
                
                # Export results
                st.header("Export Results")
                if st.button("Generate Comprehensive Report"):
                    report = self._generate_report(analyzer.properties)
                    st.download_button(
                        "Download Report",
                        report,
                        "microstructure_analysis_report.txt",
                        "text/plain"
                    )
    
    else:
        st.info("Please upload a microstructure image to begin analysis")
        
        # Show example of what the app can do
        st.subheader("Supported Analysis Features")
        
        features_col1, features_col2 = st.columns(2)
        
        with features_col1:
            st.markdown("""
            **Segmentation Algorithms:**
            - Otsu Thresholding
            - Multi-Otsu Classification
            - Adaptive Thresholding
            - Watershed Segmentation
            - K-means Clustering
            - Region Growing
            - Active Contours
            """)
            
            st.markdown("""
            **Preprocessing Options:**
            - Gaussian Blur
            - Median Filtering
            - Bilateral Filtering
            - CLAHE Enhancement
            """)
        
        with features_col2:
            st.markdown("""
            **Extracted Properties:**
            - Area/Volume fractions
            - Grain size distributions
            - Shape factors and morphology
            - Spatial correlations
            - Connectivity analysis
            - Texture characterization
            - Statistical distributions
            - Fractal analysis
            """)

def _generate_report(properties):
    """Generate comprehensive analysis report"""
    report = "MICROSTRUCTURE ANALYSIS REPORT\n"
    report += "=" * 50 + "\n\n"
    
    if 'geometric' in properties:
        geom = properties['geometric']
        report += "GEOMETRIC PROPERTIES:\n"
        report += f"Area Fraction: {geom['area_fraction']:.4f}\n"
        report += f"Mean Grain Area: {geom['mean_area']:.2f} pixels²\n"
        report += f"Area Standard Deviation: {geom['std_area']:.2f} pixels²\n"
        report += f"Mean Aspect Ratio: {geom['mean_aspect_ratio']:.3f}\n"
        report += f"Mean Circularity: {geom['mean_circularity']:.3f}\n\n"
    
    if 'spatial' in properties:
        spatial = properties['spatial']
        report += "SPATIAL DISTRIBUTION PROPERTIES:\n"
        report += f"Mean Nearest Neighbor Distance: {spatial['mean_nearest_distance']:.2f} pixels\n"
        report += f"Distance Standard Deviation: {spatial['std_nearest_distance']:.2f} pixels\n"
        report += f"Homogeneity Index: {spatial['homogeneity_index']:.4f}\n\n"
    
    if 'connectivity' in properties:
        conn = properties['connectivity']
        report += "CONNECTIVITY AND TOPOLOGY:\n"
        report += f"Euler Number: {conn['euler_number']}\n"
        report += f"Interface Area: {conn['interface_area']} pixels\n"
        report += f"Connected Components: {conn['num_connected_components']}\n"
        report += f"Tortuosity: {conn['tortuosity']:.4f}\n\n"
    
    if 'statistical' in properties:
        stats = properties['statistical']
        report += "STATISTICAL PROPERTIES:\n"
        report += f"Mean Size: {stats['mean_size']:.2f} pixels²\n"
        report += f"Size Standard Deviation: {stats['std_size']:.2f} pixels²\n"
        report += f"Size Skewness: {stats['size_skewness']:.4f}\n"
        report += f"Texture Variance: {stats['texture_variance']:.4f}\n"
        report += f"Fractal Dimension: {stats['fractal_dimension']:.4f}\n"
    
    return report

if __name__ == "__main__":
    main()
