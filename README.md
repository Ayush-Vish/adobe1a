# Challenge 1a: PDF Processing Solution

## Overview
This is a **sample solution** for Challenge 1a of the Adobe India Hackathon 2025. The challenge requires implementing a PDF processing solution that extracts structured data from PDF documents and outputs JSON files. The solution must be containerized using Docker and meet specific performance and resource constraints.

## Official Challenge Guidelines

### Submission Requirements
- **GitHub Project**: Complete code repository with working solution
- **Dockerfile**: Must be present in the root directory and functional
- **README.md**:  Documentation explaining the solution, models, and libraries used

### Build Command
```bash
docker build --platform linux/amd64 -t <reponame.someidentifier> .
```

### Run Command
```bash
docker run --rm -v $(pwd)/input:/app/input:ro -v $(pwd)/output/repoidentifier/:/app/output --network none <reponame.someidentifier>
```

### Critical Constraints
- **Execution Time**: ≤ 10 seconds for a 50-page PDF
- **Model Size**: ≤ 200MB (if using ML models)
- **Network**: No internet access allowed during runtime execution
- **Runtime**: Must run on CPU (amd64) with 8 CPUs and 16 GB RAM
- **Architecture**: Must work on AMD64, not ARM-specific

### Key Requirements
- **Automatic Processing**: Process all PDFs from `/app/input` directory
- **Output Format**: Generate `filename.json` for each `filename.pdf`
- **Input Directory**: Read-only access only
- **Open Source**: All libraries, models, and tools must be open source
- **Cross-Platform**: Test on both simple and complex PDFs

## Sample Solution Structure
```
Challenge_1a/
├── sample_dataset/
│   ├── outputs/         # JSON files provided as outputs.
│   ├── pdfs/            # Input PDF files
│   └── schema/          # Output schema definition
│       └── output_schema.json
├── Dockerfile           # Docker container configuration
├── process_pdfs.py      # Sample processing script
└── README.md           # This file
```

## Sample Implementation

### Implemented Solution: Advanced PDF Structure Extraction
The `process_pdfs.py` implements a **production-ready solution** that demonstrates:
- **Multi-layered heading detection** using fuzzy logic and statistical analysis
- **Robust PDF parsing** with PyMuPDF for accurate text and font extraction
- **Intelligent document structure recognition** across diverse document formats
- **High-performance processing** optimized for the 10-second constraint

**Key Features:**
- **Fuzzy Logic Classification**: 7-component weighted scoring system for heading detection
- **Statistical Font Analysis**: Dynamic thresholding based on document-specific typography
- **Pattern Recognition**: Comprehensive regex patterns for section numbering schemes
- **TOC Extraction**: Leverages embedded bookmarks when available
- **Error Recovery**: Multiple fallback strategies ensure robust processing

### Core Processing Pipeline
```python
def extract_outline_ultra_precise(pdf_path: str) -> Dict:
    # 1. Try embedded TOC/bookmarks first
    toc = doc.get_toc()
    if toc:
        return process_toc(toc)
    
    # 2. Statistical font analysis
    font_analyzer = PrecisionFontAnalyzer()
    body_size, size_to_level, metadata = font_analyzer.analyze_document_fonts(doc)
    
    # 3. Fuzzy logic heading classification
    fuzzy_classifier = AdvancedFuzzyHeadingClassifier()
    for text_span in document:
        score = fuzzy_classifier.calculate_fuzzy_heading_score(
            text, font_size, position, context...)
        
    # 4. Post-processing and validation
    return validate_and_structure_outline(candidates)
```

### Production Docker Configuration
```dockerfile
FROM --platform=linux/amd64 python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY process_pdfs.py .
CMD ["python", "process_pdfs.py"]
```

**Dependencies:**
- **PyMuPDF**: High-performance PDF parsing library
- **Python 3.10**: Optimized for performance and compatibility

## Expected Output Format

### Required JSON Structure
Each PDF should generate a corresponding JSON file that **must conform to the schema** defined in `sample_dataset/schema/output_schema.json`.


## Advanced Implementation Features

### Multi-Layered Heading Detection
1. **Primary Layer**: Embedded TOC/bookmark extraction
2. **Secondary Layer**: Statistical font analysis with dynamic thresholing
3. **Tertiary Layer**: Fuzzy logic classification with 7-component scoring
4. **Validation Layer**: Context validation and duplicate removal

### Fuzzy Logic Scoring Components
- **Font Size Analysis (25%)**: Gaussian membership functions
- **Length Optimization (20%)**: Trapezoidal membership for ideal heading lengths  
- **Pattern Matching (25%)**: Regex-based recognition of numbering schemes
- **Semantic Analysis (15%)**: Recognition of common section headers
- **Typography Analysis (10%)**: Bold, italic, case analysis
- **Position Analysis (5%)**: Page position weighting
- **Whitespace Analysis (10%)**: Isolation and spacing patterns

### Performance Optimizations
- **Memory Efficiency**: Streaming page-by-page processing
- **Statistical Clustering**: Advanced font size hierarchy detection
- **Early Termination**: Skip obvious non-heading content
- **Vectorized Operations**: Batch processing of font metrics
- **Resource Monitoring**: Integrated memory and time tracking


## Testing Your Solution

### Local Testing
```bash
# Build the Docker image
docker build --platform linux/amd64 -t pdf-processor .

# Test with sample data
docker run --rm -v $(pwd)/sample_dataset/pdfs:/app/input:ro -v $(pwd)/sample_dataset/outputs:/app/output --network none pdf-processor
```

### Validation Checklist
- [ ] All PDFs in input directory are processed
- [ ] JSON output files are generated for each PDF
- [ ] Output format matches required structure
- [ ] **Output conforms to schema** in `sample_dataset/schema/output_schema.json`
- [ ] Processing completes within 10 seconds for 50-page PDFs
- [ ] Solution works without internet access
- [ ] Memory usage stays within 16GB limit
- [ ] Compatible with AMD64 architecture

---

**Important**: This is a production-ready implementation that demonstrates advanced PDF structure extraction techniques. The solution employs sophisticated fuzzy logic, statistical analysis, and pattern recognition to achieve robust heading detection across diverse document formats while meeting all challenge constraints. 