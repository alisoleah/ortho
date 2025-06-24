# orthodontic_processor.py - Backend Processing Logic
# This file handles all the wire generation processing
# Import your existing wire generator here

import numpy as np
import time
import traceback
from typing import Dict, List, Any, Optional

# Import your existing wire generator
try:
    from tes_stl9claude_latestin import CleanLingualWireGenerator
    WIRE_GENERATOR_AVAILABLE = True
    print("✅ Successfully imported CleanLingualWireGenerator from tes_stl9claude_latestin.py")
except ImportError as e:
    print(f"⚠️ Could not import CleanLingualWireGenerator: {e}")
    print("Using mock processor for demonstration")
    WIRE_GENERATOR_AVAILABLE = False

class OrthodonticProcessor:
    """Backend processor for orthodontic wire generation."""
    
    def __init__(self):
        """Initialize the processor."""
        self.generator = None
        self.last_result = None
        self.processing_status = "Ready"
        
        print("🔧 Orthodontic Processor initialized")
        if WIRE_GENERATOR_AVAILABLE:
            print("✅ Real wire generator available")
        else:
            print("⚠️ Using mock mode - place tes_stl9claude_latestin.py in same directory")
    
    def generate_wire(self, file_path: str, arch_type: str = "auto", 
                     wire_size: str = "0.018") -> Dict[str, Any]:
        """
        Generate orthodontic wire from STL file.
        
        Args:
            file_path: Path to STL file
            arch_type: Type of arch ("auto", "upper", "lower")
            wire_size: Wire size specification
            
        Returns:
            Dictionary containing wire generation results
        """
        print(f"\n{'='*60}")
        print("STARTING WIRE GENERATION")
        print(f"{'='*60}")
        print(f"File: {file_path}")
        print(f"Arch Type: {arch_type}")
        print(f"Wire Size: {wire_size}")
        
        try:
            self.processing_status = "Processing"
            start_time = time.time()
            
            if WIRE_GENERATOR_AVAILABLE:
                # Use your actual wire generator
                result = self._process_with_real_generator(file_path, arch_type, wire_size)
            else:
                # Use mock generator for testing
                result = self._process_with_mock_generator(file_path, arch_type, wire_size)
            
            # Post-process results
            processed_result = self._post_process_results(result, start_time)
            
            self.last_result = processed_result
            self.processing_status = "Complete"
            
            print(f"✅ Wire generation completed successfully!")
            print(f"Processing time: {time.time() - start_time:.2f} seconds")
            
            return processed_result
            
        except Exception as e:
            self.processing_status = "Error"
            error_msg = f"Wire generation failed: {str(e)}"
            print(f"❌ {error_msg}")
            print(f"Stack trace:\n{traceback.format_exc()}")
            raise Exception(error_msg)
    
    def _process_with_real_generator(self, file_path: str, arch_type: str, 
                                   wire_size: str) -> Dict[str, Any]:
        """Process using the real CleanLingualWireGenerator."""
        print("🔧 Using CleanLingualWireGenerator...")
        
        # Initialize your wire generator
        self.generator = CleanLingualWireGenerator(
            stl_path=file_path,
            arch_type=arch_type,
            wire_size=wire_size
        )
        
        # Generate the wire using your existing code
        result = self.generator.generate_clean_lingual_wire()
        
        if result is None:
            raise Exception("Wire generator returned no results")
        
        return result
    
    def _process_with_mock_generator(self, file_path: str, arch_type: str, 
                                   wire_size: str) -> Dict[str, Any]:
        """Process using mock generator for testing."""
        print("🔧 Using mock generator (for testing)...")
        
        # Simulate processing time
        processing_steps = [
            "Loading mesh data...",
            "Detecting teeth...", 
            "Classifying tooth types...",
            "Positioning brackets...",
            "Generating wire path...",
            "Creating visualization data..."
        ]
        
        for i, step in enumerate(processing_steps):
            print(f"  {step}")
            time.sleep(0.3)  # Simulate processing time
        
        # Return mock data that matches your real generator's output structure
        return {
            'mesh': None,  # Would contain Open3D mesh in real version
            'wire_mesh': None,  # Would contain Open3D wire mesh
            'bracket_markers': None,  # Would contain Open3D bracket markers
            'bracket_positions': self._generate_mock_bracket_positions(),
            'wire_path': [],  # Would contain wire path points
            'is_valid': True,
            'surface': 'lingual'
        }
    
    def _generate_mock_bracket_positions(self) -> List[Dict[str, Any]]:
        """Generate mock bracket positions for testing."""
        # Create realistic bracket positions for a lower arch
        positions = [
            # Left side posterior to anterior
            {'position': np.array([-35, -10, 2]), 'tooth_type': 'posterior'},
            {'position': np.array([-30, -5, 3]), 'tooth_type': 'posterior'},
            {'position': np.array([-25, 0, 4]), 'tooth_type': 'posterior'},
            {'position': np.array([-20, 5, 4]), 'tooth_type': 'posterior'},
            {'position': np.array([-15, 8, 4]), 'tooth_type': 'canine'},
            {'position': np.array([-10, 10, 4]), 'tooth_type': 'incisor'},
            {'position': np.array([-5, 11, 4]), 'tooth_type': 'incisor'},
            
            # Right side anterior to posterior  
            {'position': np.array([5, 11, 4]), 'tooth_type': 'incisor'},
            {'position': np.array([10, 10, 4]), 'tooth_type': 'incisor'},
            {'position': np.array([15, 8, 4]), 'tooth_type': 'canine'},
            {'position': np.array([20, 5, 4]), 'tooth_type': 'posterior'},
            {'position': np.array([25, 0, 4]), 'tooth_type': 'posterior'},
            {'position': np.array([30, -5, 3]), 'tooth_type': 'posterior'},
            {'position': np.array([35, -10, 2]), 'tooth_type': 'posterior'},
        ]
        
        # Add additional metadata to each bracket
        for i, bracket in enumerate(positions):
            bracket.update({
                'tooth_index': i,
                'tooth_center': bracket['position'].copy(),
                'normal': np.array([0, -1, 0]),  # Inward normal for lingual
                'height': 4.0,
                'surface': 'lingual'
            })
        
        return positions
    
    def _post_process_results(self, raw_result: Dict[str, Any], 
                            start_time: float) -> Dict[str, Any]:
        """Post-process the raw results into a standardized format."""
        print("🔧 Post-processing results...")
        
        bracket_positions = raw_result.get('bracket_positions', [])
        
        # Count tooth types
        tooth_counts = {'incisor': 0, 'canine': 0, 'posterior': 0}
        processed_brackets = []
        
        for bracket in bracket_positions:
            tooth_type = bracket.get('tooth_type', 'unknown')
            if tooth_type in tooth_counts:
                tooth_counts[tooth_type] += 1
            
            # Ensure position is in the right format
            position = bracket['position']
            if hasattr(position, 'tolist'):
                position_list = position.tolist()
            elif isinstance(position, (list, tuple)):
                position_list = list(position)
            else:
                position_list = [float(position[0]), float(position[1]), float(position[2])]
            
            processed_brackets.append({
                'type': tooth_type,
                'position': position_list,
                'color': self._get_tooth_color(tooth_type),
                'index': len(processed_brackets)
            })
        
        # Calculate wire length
        wire_length = self._calculate_wire_length(processed_brackets)
        
        # Calculate classification accuracy
        classification_accuracy = self._calculate_classification_accuracy(tooth_counts)
        
        # Generate bending instructions
        bending_instructions = self._generate_bending_instructions(processed_brackets)
        
        # Create standardized result
        processed_result = {
            'teeth_detected': len(bracket_positions),
            'incisors': tooth_counts['incisor'],
            'canines': tooth_counts['canine'], 
            'posterior': tooth_counts['posterior'],
            'wire_length': wire_length,
            'classification_accuracy': classification_accuracy,
            'bracket_positions': processed_brackets,
            'bending_instructions': bending_instructions,
            'processing_time': time.time() - start_time,
            'raw_result': raw_result,
            'metadata': {
                'generator_type': 'real' if WIRE_GENERATOR_AVAILABLE else 'mock',
                'surface': 'lingual',
                'timestamp': time.time()
            }
        }
        
        # Validation
        self._validate_results(processed_result)
        
        return processed_result
    
    def _get_tooth_color(self, tooth_type: str) -> str:
        """Get color for tooth type."""
        color_map = {
            'incisor': 'green',
            'canine': 'yellow', 
            'posterior': 'blue',
            'unknown': 'gray'
        }
        return color_map.get(tooth_type, 'gray')
    
    def _calculate_wire_length(self, bracket_positions: List[Dict[str, Any]]) -> float:
        """Calculate total wire length."""
        if len(bracket_positions) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(bracket_positions) - 1):
            p1 = np.array(bracket_positions[i]['position'])
            p2 = np.array(bracket_positions[i + 1]['position'])
            segment_length = np.linalg.norm(p2 - p1)
            total_length += segment_length
        
        return total_length
    
    def _calculate_classification_accuracy(self, tooth_counts: Dict[str, int]) -> float:
        """Calculate classification accuracy score."""
        expected = {'incisor': 4, 'canine': 2}
        score = 0
        total = len(expected)
        
        for tooth_type, expected_count in expected.items():
            actual_count = tooth_counts.get(tooth_type, 0)
            if actual_count == expected_count:
                score += 1
        
        return (score / total) * 100 if total > 0 else 0
    
    def _generate_bending_instructions(self, bracket_positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate step-by-step bending instructions."""
        if not bracket_positions:
            return []
        
        instructions = []
        cumulative_feed = 0.0
        
        for i, bracket in enumerate(bracket_positions):
            # Calculate feed distance
            if i > 0:
                prev_pos = np.array(bracket_positions[i-1]['position'])
                curr_pos = np.array(bracket['position'])
                segment_length = np.linalg.norm(curr_pos - prev_pos)
                cumulative_feed += segment_length
            
            # Calculate bend angle (simplified calculation)
            bend_angle = 0.0
            if 0 < i < len(bracket_positions) - 1:
                p1 = np.array(bracket_positions[i-1]['position'])
                p2 = np.array(bracket['position'])
                p3 = np.array(bracket_positions[i+1]['position'])
                
                v1 = p1 - p2
                v2 = p3 - p2
                
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    v1_norm = v1 / np.linalg.norm(v1)
                    v2_norm = v2 / np.linalg.norm(v2)
                    
                    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1, 1)
                    angle_rad = np.arccos(dot_product)
                    bend_angle = np.degrees(angle_rad)
                    
                    # Adjust for realism
                    if bend_angle > 170:
                        bend_angle = 180 - bend_angle
            
            # Create instruction
            instruction = {
                'step': i + 1,
                'feed_mm': round(cumulative_feed, 1),
                'bend_deg': round(bend_angle, 1),
                'location': f"{bracket['type'].title()} #{i+1}",
                'notes': self._get_clinical_note(bracket['type'], bend_angle, i, len(bracket_positions))
            }
            
            instructions.append(instruction)
        
        return instructions
    
    def _get_clinical_note(self, tooth_type: str, bend_angle: float, 
                          position: int, total_positions: int) -> str:
        """Generate clinical notes for bending instructions."""
        notes = []
        
        # Position notes
        if position == 0:
            notes.append("Wire start")
        elif position == total_positions - 1:
            notes.append("Wire end")
        
        # Tooth-specific notes
        if tooth_type == 'incisor':
            notes.append("Incisal edge")
        elif tooth_type == 'canine':
            notes.append("Canine cusp tip")
        elif tooth_type == 'posterior':
            notes.append("Buccal cusp")
        
        # Bend-specific notes
        if abs(bend_angle) > 20:
            notes.append("Major bend")
        elif abs(bend_angle) > 10:
            notes.append("Moderate bend")
        elif abs(bend_angle) > 5:
            notes.append("Minor bend")
        else:
            notes.append("Straight")
        
        # Surface note
        notes.append("Lingual placement")
        
        return "; ".join(notes) if notes else "Standard placement"
    
    def _validate_results(self, result: Dict[str, Any]) -> None:
        """Validate the processing results."""
        print("🔧 Validating results...")
        
        required_fields = [
            'teeth_detected', 'incisors', 'canines', 'posterior',
            'wire_length', 'classification_accuracy', 'bracket_positions'
        ]
        
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate counts
        if result['teeth_detected'] < 6:
            print("⚠️  Warning: Low tooth count detected")
        
        if result['classification_accuracy'] < 50:
            print("⚠️  Warning: Low classification accuracy")
        
        if result['wire_length'] < 30 or result['wire_length'] > 200:
            print("⚠️  Warning: Unusual wire length")
        
        print("✅ Results validation passed")
    
    def get_processing_status(self) -> str:
        """Get current processing status."""
        return self.processing_status
    
    def get_last_result(self) -> Optional[Dict[str, Any]]:
        """Get the last processing result."""
        return self.last_result
    
    def export_to_json(self, result: Dict[str, Any], filename: str) -> None:
        """Export results to JSON format."""
        import json
        
        # Create exportable data (remove non-serializable objects)
        export_data = {
            'metadata': {
                'software': 'Dental Archwire Design System v3.0',
                'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
                'surface': 'lingual'
            },
            'classification': {
                'teeth_detected': result['teeth_detected'],
                'incisors': result['incisors'],
                'canines': result['canines'],
                'posterior': result['posterior'],
                'accuracy': result['classification_accuracy']
            },
            'wire_specifications': {
                'length_mm': result['wire_length'],
                'length_inches': result['wire_length'] / 25.4,
                'surface': 'lingual'
            },
            'bracket_positions': result['bracket_positions'],
            'bending_instructions': result['bending_instructions']
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def export_to_csv(self, result: Dict[str, Any], filename: str) -> None:
        """Export bending instructions to CSV format."""
        import csv
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header
            writer.writerow(['Lingual Wire Bending Instructions'])
            writer.writerow([f'Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}'])
            writer.writerow([f'Total Length: {result["wire_length"]:.1f}mm'])
            writer.writerow([f'Classification Accuracy: {result["classification_accuracy"]:.1f}%'])
            writer.writerow([])
            
            # Instructions table
            writer.writerow(['Step', 'Feed (mm)', 'Bend (°)', 'Location', 'Notes'])
            
            for instruction in result['bending_instructions']:
                writer.writerow([
                    instruction['step'],
                    instruction['feed_mm'],
                    instruction['bend_deg'],
                    instruction['location'],
                    instruction['notes']
                ])

# Utility functions for integration testing
def test_processor():
    """Test the processor with mock data."""
    print("🧪 Testing OrthodonticProcessor...")
    
    processor = OrthodonticProcessor()
    
    # Test with mock file
    try:
        result = processor.generate_wire(
            file_path="test.stl",
            arch_type="lower", 
            wire_size="0.018"
        )
        
        print(f"✅ Test passed!")
        print(f"   Teeth detected: {result['teeth_detected']}")
        print(f"   Classification accuracy: {result['classification_accuracy']:.1f}%")
        print(f"   Wire length: {result['wire_length']:.1f}mm")
        print(f"   Bending instructions: {len(result['bending_instructions'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def validate_integration():
    """Validate integration with CleanLingualWireGenerator."""
    print("🔍 Validating integration...")
    
    if WIRE_GENERATOR_AVAILABLE:
        print("✅ CleanLingualWireGenerator successfully imported")
        
        # Test if the class has required methods
        required_methods = ['generate_clean_lingual_wire']
        
        for method in required_methods:
            if hasattr(CleanLingualWireGenerator, method):
                print(f"✅ Method '{method}' found")
            else:
                print(f"❌ Method '{method}' missing")
                return False
        
        print("✅ Integration validation passed")
        return True
    else:
        print("⚠️  CleanLingualWireGenerator not available - using mock mode")
        return False

# Main execution for testing
if __name__ == "__main__":
    print("="*60)
    print("ORTHODONTIC PROCESSOR - STANDALONE TEST")
    print("="*60)
    
    # Validate integration
    integration_ok = validate_integration()
    
    # Test processor
    test_ok = test_processor()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Integration: {'✅ PASSED' if integration_ok else '⚠️  MOCK MODE'}")
    print(f"Processor Test: {'✅ PASSED' if test_ok else '❌ FAILED'}")
    
    if integration_ok and test_ok:
        print("\n🎉 All tests passed! Ready for GUI integration.")
    elif test_ok:
        print("\n⚠️  Tests passed in mock mode. Place tes_stl9claude_latestin.py in same directory for full functionality.")
    else:
        print("\n❌ Tests failed. Check error messages above.")
    
    print("\nTo use with GUI: python orthodontic_gui.py")