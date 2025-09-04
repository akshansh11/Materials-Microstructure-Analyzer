# Microstructure Image Analysis Tool

A comprehensive Streamlit-based application for analyzing microstructure images using advanced segmentation algorithms and extracting quantitative properties for materials science applications.

## Features

### Segmentation Algorithms
- **Otsu Thresholding** - Automatic threshold selection for binary segmentation
- **Multi-Otsu** - Multi-class segmentation with adjustable number of classes
- **Adaptive Thresholding** - Local threshold adaptation for varying illumination
- **Watershed Segmentation** - Marker-based region segmentation for touching objects
- **K-means Clustering** - Pixel intensity-based clustering
- **Region Growing** - Seed-based region expansion
- **Active Contours (Snake)** - Morphological contour detection

### Preprocessing Options
- **Gaussian Blur** - Noise reduction with adjustable sigma
- **Median Filter** - Salt-and-pepper noise removal
- **Bilateral Filter** - Edge-preserving smoothing
- **CLAHE** - Contrast Limited Adaptive Histogram Equalization

### Extracted Properties

#### Geometric Properties
- Area fraction/Volume fraction
- Grain size distribution
- Aspect ratios
- Shape factors (circularity, elongation, roundness)
- Perimeter measurements
- Equivalent diameter

#### Spatial Distribution Properties
- Nearest neighbor distances
- Clustering analysis
- Orientation distribution
- Spatial correlation functions
- Homogeneity indices

#### Connectivity and Topology
- Connectivity analysis
- Percolation threshold estimation
- Euler number
- Tortuosity measurements
- Interface area calculations

#### Statistical Properties
- Phase statistics (mean, std dev, skewness)
- Texture analysis using Local Binary Patterns
- Fractal dimension estimation
- Stereological parameters

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/microstructure-analyzer.git
cd microstructure-analyzer
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run main.py
```

## Requirements

- streamlit
- numpy
- opencv-python-headless
- Pillow
- matplotlib
- seaborn
- scipy
- scikit-image
- scikit-learn
- pandas
- plotly
- kaleido

## Usage

1. **Upload Image**: Click "Upload Microstructure Image" in the sidebar and select your image file (PNG, JPG, JPEG, TIFF, BMP)

2. **Preprocessing** (Optional): Select a preprocessing method and adjust parameters:
   - Gaussian Blur: Adjust sigma value
   - Median Filter: Set filter size
   - Bilateral Filter: Configure sigma color and spatial parameters
   - CLAHE: Automatic contrast enhancement

3. **Segmentation**: Choose a segmentation algorithm and configure parameters:
   - Multi-Otsu: Set number of classes
   - Adaptive Thresholding: Adjust block size
   - Watershed: Configure minimum distance between peaks
   - K-means: Set number of clusters
   - Region Growing: Set threshold value

4. **Apply Segmentation**: Click "Apply Segmentation" to process the image

5. **View Results**: Explore the extracted properties in four categories:
   - Geometric Properties
   - Spatial Distribution
   - Connectivity & Topology
   - Statistical Properties

6. **Export**: Generate and download a comprehensive analysis report

## Supported Image Formats

- PNG
- JPEG/JPG
- TIFF
- BMP

## Output

The application provides:
- Interactive visualizations of all properties
- Statistical distributions and histograms
- Correlation matrices and scatter plots
- Quantitative metrics and measurements
- Downloadable analysis reports

## Applications

This tool is designed for:
- Materials science research
- Metallurgical analysis
- Ceramic microstructure characterization
- Composite material analysis
- Quality control in manufacturing
- Academic research and education

## Example Analysis Results

The app extracts and visualizes:
- Grain size distributions
- Shape factor correlations
- Spatial distribution patterns
- Connectivity metrics
- Statistical summaries

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Technical Details

### Image Processing Pipeline
1. **Image Loading**: Automatic conversion to grayscale
2. **Preprocessing**: Optional noise reduction and enhancement
3. **Segmentation**: Multiple algorithms for different microstructure types
4. **Labeling**: Connected component analysis
5. **Property Extraction**: Comprehensive quantitative analysis

### Performance Considerations
- Optimized for images up to 2048x2048 pixels
- Efficient algorithms for large grain counts
- Memory-conscious processing for batch analysis

## Troubleshooting

### Common Issues

**ImportError with OpenCV:**
- Ensure you're using `opencv-python-headless` instead of `opencv-python`
- For deployment on cloud platforms, the headless version is required

**Memory Issues:**
- Reduce image size for very large images
- Use simpler segmentation algorithms for initial analysis

**Segmentation Quality:**
- Try different preprocessing methods
- Adjust segmentation parameters
- Consider Multi-Otsu for multi-phase materials

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Uses [scikit-image](https://scikit-image.org/) for image processing
- Visualization powered by [Plotly](https://plotly.com/)
- Scientific computing with [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/)

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{microstructure_analyzer,
  title={Microstructure Image Analysis Tool},
  author={Akshansh Mishra},
  year={2025},
  url={https://github.com/akshansh11}
}
```

Contact
For questions, suggestions, or collaboration opportunities, please open an issue or contact akshansh@aifablab.com.
