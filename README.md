# Professional Orthodontic Wire Generator v2.0

## Overview

The Professional Orthodontic Wire Generator is a Python-based tool that automatically generates clinically accurate orthodontic arch wires from dental STL models. It uses advanced 3D mesh processing and anatomical zoning to create wire designs suitable for orthodontic treatment planning and manufacturing.

## Key Features

### 1. **Anatomical Zoning Classification**
- **Robust tooth detection** using angular segmentation
- **Automatic classification** of teeth into three categories:
  - **Incisors (Blue)**: The 4 front teeth in the center of the arch
  - **Canines (Yellow)**: The 2 corner teeth flanking the incisors
  - **Posterior (Green)**: All remaining teeth (premolars and molars)

### 2. **Intelligent Bracket Positioning**
- Clinically accurate bracket heights based on tooth type
- **Facial surface detection** ensures brackets are placed on the outer surface
- **3.5mm clearance offset** prevents wire penetration through teeth
- Adaptive positioning for various arch forms

### 3. **Non-Penetrating Wire Path**
- Smooth spline interpolation between bracket points
- Automatic detection and handling of posterior gaps
- Symmetrical wire generation for balanced forces
- Clinical validation of wire clearance

### 4. **Production-Ready Output**
- CSV export with bending instructions for CNC wire benders
- Feed distances and bend angles for each point
- Clinical notes for special considerations
- Quality validation before export

## Installation

### Prerequisites
```bash
pip install open3d numpy scipy
```

### Required Libraries
- **Open3D**: 3D mesh processing and visualization
- **NumPy**: Numerical computations
- **SciPy**: Spline interpolation and optimization

## Usage

### Basic Usage
```python
from orthodontic_wire_generator import ProfessionalOrthodonticWireGenerator

# Create generator instance
generator = ProfessionalOrthodonticWireGenerator(
    stl_path="path/to/dental_model.stl",
    arch_type='auto',  # 'auto', 'upper', or 'lower'
    wire_size='0.018'  # Standard orthodontic wire size
)

# Generate wire
result = generator.generate_professional_wire()

# Visualize results
if result:
    generator.visualize_professional(result)
```

### Wire Sizes Available
- `'0.012'`: Initial alignment (0.3048mm)
- `'0.014'`: Light forces (0.3556mm)
- `'0.016'`: Medium forces (0.4064mm)
- `'0.018'`: Standard treatment (0.4572mm)
- `'0.020'`: Heavy forces (0.5080mm)
- Rectangular wires: `'0.016x0.022'`, `'0.018x0.025'`, etc.

## Algorithm Details

### 1. Tooth Detection Process
```
1. Load STL mesh and identify anatomical axes
2. Sample vertices at clinical crown level
3. Perform angular segmentation around arch
4. Filter segments by vertex count and position
5. Calculate tooth centers using outer vertices
```

### 2. Anatomical Zoning Classification
```
1. Sort all detected teeth by anterior-posterior position
2. Select the 6 most anterior teeth
3. Sort these 6 teeth by left-right position
4. Classify:
   - Outermost 2 as canines (positions 0 and 5)
   - Inner 4 as incisors (positions 1-4)
   - All remaining teeth as posterior
```

### 3. Bracket Positioning Algorithm
```
1. Determine clinical bracket height for tooth type
2. Sample vertices at bracket level (±2mm tolerance)
3. Calculate radial distances from arch center
4. Select vertices in outer 15% (facial surface)
5. Apply 3.5mm outward offset for wire clearance
```

### 4. Wire Path Generation
```
1. Sort brackets by angular position
2. Identify and handle posterior gap
3. Create spline interpolation through brackets
4. Validate non-penetration
5. Generate smooth continuous path
```

## Color Coding System

The system uses a standardized color scheme for easy identification:

| Tooth Type | Color | Description |
|------------|-------|-------------|
| Incisors | **Blue** | Central and lateral incisors (4 teeth) |
| Canines | **Yellow** | Corner teeth (2 teeth) |
| Posterior | **Green** | Premolars and molars (remaining teeth) |
| Wire | **Silver** | Stainless steel appearance |

## Output Files

### CSV Bending Instructions
The exported CSV file contains:
- **Header**: Arch type, wire size, generation date
- **Columns**:
  - Point: Sequential bend point number
  - Feed_mm: Cumulative wire feed distance
  - Bend_deg: Bend angle in degrees
  - X_mm, Y_mm, Z_mm: 3D coordinates
  - Tooth_Type: Classification (Incisor/Canine/Posterior)
  - Notes: Clinical considerations

### Example CSV Output
```csv
Professional Orthodontic Wire - Clinical Instructions
Arch Type: LOWER
Wire Size: 0.018
Generated: Mon Nov 25 2024

Point,Feed_mm,Bend_deg,X_mm,Y_mm,Z_mm,Tooth_Type,Notes
1,0.00,0.0,-15.23,8.45,22.10,Posterior,Posterior anchor
2,8.45,12.3,-10.12,10.23,23.45,Posterior,Posterior anchor
3,16.78,-8.5,-5.34,11.89,24.12,Canine,Canine position
...
```

## Quality Validation

The system performs automatic quality checks:

1. **Tooth Count**: Minimum 10 teeth detected
2. **Bracket Count**: One bracket per tooth
3. **Color Coding**: Exactly 4 incisors and 2 canines
4. **Wire Clearance**: All brackets properly offset
5. **Symmetry**: Balanced left-right distribution

## Troubleshooting

### Common Issues and Solutions

1. **Incorrect Tooth Classification**
   - **Cause**: Poor mesh quality or unusual arch form
   - **Solution**: Ensure STL has sufficient resolution (>50k vertices)

2. **Wire Penetration**
   - **Cause**: Incorrect facial surface detection
   - **Solution**: Increase clinical offset or check mesh orientation

3. **Missing Teeth**
   - **Cause**: Gaps in mesh or insufficient vertices
   - **Solution**: Use complete dental scans without missing teeth

4. **Asymmetric Wire**
   - **Cause**: Uneven tooth distribution
   - **Solution**: Check for missing teeth or mesh artifacts

## Advanced Configuration

### Custom Bracket Heights
```python
generator.BRACKET_HEIGHTS['lower']['canine'] = 5.5  # Modify canine height
```

### Adjust Wire Clearance
```python
# In calculate_robust_bracket_positions()
clinical_offset = 4.0  # Increase from default 3.5mm
```

### Custom Tooth Detection Parameters
```python
# In detect_teeth_with_anatomical_zoning()
expected_teeth = 12  # For partial arches
height_tolerance = 3.0  # Wider crown sampling
```

## Clinical Considerations

1. **Wire Material**: Designed for stainless steel wires
2. **Force System**: Assumes passive wire placement
3. **Bracket System**: Compatible with standard bracket slots
4. **Treatment Stage**: Suitable for alignment phase

## Version History

### v2.0 (Current)
- Implemented anatomical zoning classification
- Fixed color coding to match clinical standards
- Improved facial surface detection
- Enhanced wire clearance calculations
- Added quality validation system

### v1.0
- Initial implementation
- Basic tooth detection
- Simple classification system

## License and Disclaimer

This software is provided for educational and research purposes. Clinical use requires validation by a licensed orthodontist. The developers assume no liability for clinical outcomes.

## Support

For technical support or bug reports, please include:
1. STL file (or sample)
2. Console output
3. Generated CSV file
4. Screenshots of visualization

---

*Professional Orthodontic Wire Generator v2.0 - Anatomically Accurate Wire Design*
