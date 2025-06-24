# Clean Lingual Orthodontic Wire Generator - Error Free Version
# Only wire segments between blue brackets, with proper variable handling
# pip install open3d numpy scipy

import numpy as np
import open3d as o3d
import os
from scipy import interpolate, signal
from scipy.spatial import distance_matrix
import csv

class CleanLingualWireGenerator:
    """Clean lingual orthodontic wire generator - error free."""
    
    # Standard wire dimensions (in inches, converted to mm)
    WIRE_SIZES = {
        '0.012': 0.3048,
        '0.014': 0.3556,
        '0.016': 0.4064,
        '0.018': 0.4572,
        '0.020': 0.5080,
        '0.016x0.022': (0.4064, 0.5588),
        '0.018x0.025': (0.4572, 0.6350),
        '0.019x0.025': (0.4826, 0.6350),
        '0.021x0.025': (0.5334, 0.6350)
    }
    
    def __init__(self, stl_path, arch_type='auto', wire_size='0.018'):
        """Initialize the wire generator."""
        self.stl_path = stl_path
        self.wire_size = wire_size
        self.wire_radius = self.WIRE_SIZES.get(wire_size, 0.4572) / 2
        
        # Detect arch type
        if arch_type == 'auto':
            self.arch_type = 'lower' if 'lower' in stl_path.lower() else 'upper'
        else:
            self.arch_type = arch_type
            
        self.mesh = None
        self.teeth = []
        self.bracket_positions = []
        self.wire_path = []
        
    def load_mesh(self):
        """Load and process the STL mesh."""
        print(f"\n{'='*70}")
        print("CLEAN LINGUAL ORTHODONTIC WIRE GENERATOR")
        print("*** ERROR FREE VERSION ***")
        print(f"{'='*70}")
        print(f"\nLoading: {os.path.basename(self.stl_path)}")
        print(f"Arch Type: {self.arch_type.upper()}")
        print(f"Wire Size: {self.wire_size}")
        print(f"Wire Position: LINGUAL (Inner/Tongue Side)")
        
        try:
            self.mesh = o3d.io.read_triangle_mesh(self.stl_path)
            if not self.mesh.has_triangles():
                raise ValueError("No triangles found in mesh")
            
            # Clean and prepare mesh
            self.mesh.remove_duplicated_vertices()
            self.mesh.remove_degenerate_triangles()
            self.mesh.compute_vertex_normals()
            self.mesh.compute_triangle_normals()
            
            # Natural tooth color
            self.mesh.paint_uniform_color([0.95, 0.93, 0.88])
            
            print(f"✓ Loaded: {len(self.mesh.vertices)} vertices, {len(self.mesh.triangles)} triangles")
            return True
            
        except Exception as e:
            print(f"✗ Error loading mesh: {e}")
            return False
    
    def detect_teeth_simple(self):
        """Simple tooth detection method."""
        print(f"\n{'─'*50}")
        print("SIMPLE TOOTH DETECTION")
        print(f"{'─'*50}")
        
        vertices = np.asarray(self.mesh.vertices)
        bbox = self.mesh.get_axis_aligned_bounding_box()
        center = self.mesh.get_center()
        extent = bbox.get_extent()
        
        # Identify anatomical axes
        self.lr_axis = np.argmax(extent)  # Left-Right (widest)
        self.height_axis = np.argmin(extent)  # Occlusal-Gingival (smallest)
        self.ap_axis = 3 - self.lr_axis - self.height_axis  # Anterior-Posterior
        
        print(f"Anatomical axes identified:")
        print(f"  • Left-Right axis: {self.lr_axis}")
        print(f"  • Anterior-Posterior axis: {self.ap_axis}")
        print(f"  • Occlusal-Gingival axis: {self.height_axis}")
        
        # Sample at crown level
        crown_ratio = 0.25 if self.arch_type == 'upper' else 0.75
        crown_level = bbox.min_bound[self.height_axis] + extent[self.height_axis] * crown_ratio
        
        # Get crown vertices
        height_tolerance = max(2.0, extent[self.height_axis] * 0.15)
        crown_mask = np.abs(vertices[:, self.height_axis] - crown_level) < height_tolerance
        crown_vertices = vertices[crown_mask]
        
        print(f"Crown level: {crown_level:.1f}mm")
        print(f"Vertices at crown: {len(crown_vertices)}")
        
        # Simple angular segmentation
        self.teeth = self._simple_angular_segmentation(crown_vertices, center)
        
        # Apply simple classification: exactly 4 incisors + 2 canines
        self._apply_simple_classification()
        
        print(f"\nDetected {len(self.teeth)} teeth")
        
        return len(self.teeth) >= 6
    
    def _simple_angular_segmentation(self, crown_vertices, center):
        """Simple angular segmentation to find teeth."""
        # Calculate angles
        angles = np.arctan2(
            crown_vertices[:, self.ap_axis] - center[self.ap_axis],
            crown_vertices[:, self.lr_axis] - center[self.lr_axis]
        )
        
        # Find posterior gap
        sorted_angles = np.sort(angles)
        angle_diffs = np.diff(sorted_angles)
        angle_diffs = np.append(angle_diffs, sorted_angles[0] + 2*np.pi - sorted_angles[-1])
        
        posterior_gap_idx = np.argmax(angle_diffs)
        posterior_gap_size = angle_diffs[posterior_gap_idx]
        
        # Define segments
        expected_teeth = 14
        active_angle_range = 2 * np.pi - posterior_gap_size
        angle_per_tooth = active_angle_range / expected_teeth
        start_angle = sorted_angles[(posterior_gap_idx + 1) % len(sorted_angles)]
        
        teeth = []
        for i in range(expected_teeth + 2):
            tooth_start = start_angle + i * angle_per_tooth
            tooth_end = tooth_start + angle_per_tooth
            
            # Normalize angles
            tooth_start = np.mod(tooth_start + np.pi, 2*np.pi) - np.pi
            tooth_end = np.mod(tooth_end + np.pi, 2*np.pi) - np.pi
            
            # Get vertices in segment
            if tooth_start < tooth_end:
                angle_mask = (angles >= tooth_start) & (angles < tooth_end)
            else:
                angle_mask = (angles >= tooth_start) | (angles < tooth_end)
            
            segment_vertices = crown_vertices[angle_mask]
            
            if len(segment_vertices) < 30:
                continue
            
            # Calculate tooth center
            tooth_center = np.mean(segment_vertices, axis=0)
            
            tooth_angle = np.arctan2(
                tooth_center[self.ap_axis] - center[self.ap_axis],
                tooth_center[self.lr_axis] - center[self.lr_axis]
            )
            
            teeth.append({
                'center': tooth_center,
                'vertices': segment_vertices,
                'angle': tooth_angle,
                'ap_position': tooth_center[self.ap_axis],
                'lr_position': tooth_center[self.lr_axis],
                'index': len(teeth),
                'type': 'posterior'  # Default to posterior
            })
        
        # Filter too-close teeth
        filtered_teeth = []
        sorted_teeth = sorted(teeth, key=lambda t: t['angle'])
        
        for i, tooth in enumerate(sorted_teeth):
            if i == 0:
                filtered_teeth.append(tooth)
            else:
                prev_tooth = filtered_teeth[-1]
                dist = np.linalg.norm(tooth['center'] - prev_tooth['center'])
                if dist > 3.0:
                    filtered_teeth.append(tooth)
        
        return filtered_teeth
    
    def _apply_simple_classification(self):
        """Simple classification: exactly 4 incisors and 2 canines."""
        print(f"\n{'─'*40}")
        print("APPLYING SIMPLE CLASSIFICATION")
        print(f"{'─'*40}")
        
        if len(self.teeth) < 6:
            print("Not enough teeth for classification")
            return
        
        # Reset all to posterior
        for tooth in self.teeth:
            tooth['type'] = 'posterior'
        
        # Sort by anterior position (highest AP values are most anterior)
        teeth_by_ap = sorted(self.teeth, key=lambda t: t['ap_position'], reverse=True)
        
        # Take the 6 most anterior teeth
        anterior_6 = teeth_by_ap[:6]
        
        # Among these 6, sort by left-right position
        anterior_6_by_lr = sorted(anterior_6, key=lambda t: t['lr_position'])
        
        # Simple assignment: outer 2 = canines, inner 4 = incisors
        if len(anterior_6_by_lr) >= 6:
            # Assign canines (outermost)
            anterior_6_by_lr[0]['type'] = 'canine'  # Leftmost
            anterior_6_by_lr[5]['type'] = 'canine'  # Rightmost
            
            # Assign incisors (inner 4)
            for i in range(1, 5):
                anterior_6_by_lr[i]['type'] = 'incisor'
        
        # Verify counts
        incisor_count = sum(1 for t in self.teeth if t['type'] == 'incisor')
        canine_count = sum(1 for t in self.teeth if t['type'] == 'canine')
        
        print(f"Classification result:")
        print(f"  • Incisors: {incisor_count}")
        print(f"  • Canines: {canine_count}")
        
        # If not exactly 4 incisors and 2 canines, use fallback
        if incisor_count != 4 or canine_count != 2:
            print(f"  ⚠️  Using fallback method...")
            self._fallback_classification()
    
    def _fallback_classification(self):
        """Fallback: force exactly 4 incisors and 2 canines."""
        # Reset all
        for tooth in self.teeth:
            tooth['type'] = 'posterior'
        
        # Find center of arch
        mesh_center = self.mesh.get_center()
        
        # Calculate centrality score for each tooth
        for tooth in self.teeth:
            # Distance from LR center
            lr_center = mesh_center[self.lr_axis]
            lr_distance = abs(tooth['lr_position'] - lr_center)
            
            # Anterior score (higher AP = more anterior)
            ap_score = tooth['ap_position']
            
            # Combined centrality (lower is more central)
            tooth['centrality'] = lr_distance - ap_score * 0.1
        
        # Sort by centrality (most central first)
        teeth_by_centrality = sorted(self.teeth, key=lambda t: t['centrality'])
        
        # Assign the 6 most central teeth
        if len(teeth_by_centrality) >= 6:
            central_6 = teeth_by_centrality[:6]
            
            # Sort these 6 by LR position
            central_6_by_lr = sorted(central_6, key=lambda t: t['lr_position'])
            
            # Assign: outer 2 = canines, inner 4 = incisors
            central_6_by_lr[0]['type'] = 'canine'  # Leftmost
            central_6_by_lr[5]['type'] = 'canine'  # Rightmost
            
            for i in range(1, 5):
                central_6_by_lr[i]['type'] = 'incisor'
        
        # Final verification
        final_incisors = sum(1 for t in self.teeth if t['type'] == 'incisor')
        final_canines = sum(1 for t in self.teeth if t['type'] == 'canine')
        
        print(f"  • Final: Incisors: {final_incisors}")
        print(f"  • Final: Canines: {final_canines}")
    
    def calculate_lingual_bracket_positions(self):
        """Calculate bracket positions on LINGUAL surface."""
        print(f"\n{'─'*50}")
        print("LINGUAL BRACKET POSITIONING")
        print(f"{'─'*50}")
        
        self.bracket_positions = []
        mesh_center = self.mesh.get_center()
        
        for i, tooth in enumerate(self.teeth):
            # Bracket height
            bracket_height = 4.0
            
            tooth_vertices = tooth['vertices']
            tooth_center = tooth['center']
            
            # Calculate target height
            if self.arch_type == 'upper':
                target_height = np.min(tooth_vertices[:, self.height_axis]) + bracket_height
            else:
                target_height = np.max(tooth_vertices[:, self.height_axis]) - bracket_height
            
            # Get vertices at bracket level
            height_tolerance = 2.0
            bracket_level_mask = np.abs(tooth_vertices[:, self.height_axis] - target_height) < height_tolerance
            bracket_level_vertices = tooth_vertices[bracket_level_mask]
            
            if len(bracket_level_vertices) < 10:
                bracket_pos = tooth_center.copy()
                bracket_pos[self.height_axis] = target_height
                
                # INWARD direction
                horizontal_vector = bracket_pos - mesh_center
                horizontal_vector[self.height_axis] = 0
                if np.linalg.norm(horizontal_vector) > 0:
                    inward_normal = -horizontal_vector / np.linalg.norm(horizontal_vector)
                else:
                    inward_normal = np.array([0, -1, 0])
            else:
                # Find INNERMOST vertices (lingual side)
                tooth_horizontal = tooth_center.copy()
                tooth_horizontal[self.height_axis] = 0
                center_horizontal = mesh_center.copy()
                center_horizontal[self.height_axis] = 0
                
                radial_vector = tooth_horizontal - center_horizontal
                if np.linalg.norm(radial_vector) > 0:
                    radial_direction = radial_vector / np.linalg.norm(radial_vector)
                else:
                    radial_direction = np.array([1, 0, 0])
                
                # Project vertices
                radial_distances = []
                for vertex in bracket_level_vertices:
                    vertex_horizontal = vertex.copy()
                    vertex_horizontal[self.height_axis] = 0
                    vertex_radial = vertex_horizontal - center_horizontal
                    radial_dist = np.dot(vertex_radial, radial_direction)
                    radial_distances.append(radial_dist)
                
                radial_distances = np.array(radial_distances)
                
                # INNERMOST vertices (15th percentile)
                percentile_15 = np.percentile(radial_distances, 15)
                lingual_mask = radial_distances <= percentile_15
                lingual_vertices = bracket_level_vertices[lingual_mask]
                
                if len(lingual_vertices) > 3:
                    bracket_pos = np.mean(lingual_vertices, axis=0)
                else:
                    bracket_pos = bracket_level_vertices[np.argmin(radial_distances)]
                
                inward_normal = -radial_direction.copy()
            
            # Apply offset
            clinical_offset = 2.0
            bracket_pos = bracket_pos + inward_normal * clinical_offset
            
            # Store bracket
            self.bracket_positions.append({
                'position': bracket_pos,
                'tooth_type': tooth['type'],
                'tooth_index': i,
                'tooth_center': tooth_center,
                'normal': inward_normal,
                'height': bracket_height,
                'surface': 'lingual'
            })
        
        print(f"✓ Positioned {len(self.bracket_positions)} LINGUAL brackets")
    
    def create_smooth_wire_path(self):
        """Create smooth wire path."""
        print(f"\n{'─'*50}")
        print("WIRE PATH GENERATION")
        print(f"{'─'*50}")
        
        if not self.bracket_positions:
            return None
        
        # Extract positions and sort by angle
        positions = np.array([b['position'] for b in self.bracket_positions])
        mesh_center = self.mesh.get_center()
        
        angles = []
        for pos in positions:
            angle = np.arctan2(
                pos[self.ap_axis] - mesh_center[self.ap_axis],
                pos[self.lr_axis] - mesh_center[self.lr_axis]
            )
            angles.append(angle)
        
        sorted_indices = np.argsort(angles)
        sorted_positions = positions[sorted_indices]
        sorted_brackets = [self.bracket_positions[i] for i in sorted_indices]
        
        # Create wire path
        self.wire_path = []
        for i, (pos, bracket) in enumerate(zip(sorted_positions, sorted_brackets)):
            self.wire_path.append({
                'position': pos.copy(),
                'type': 'bracket',
                'tooth_type': bracket['tooth_type'],
                'index': i,
                'surface': 'lingual'
            })
        
        print(f"✓ Generated wire path with {len(self.wire_path)} points")
        return self.wire_path
    
    def create_clean_visualization_meshes(self):
        """Create clean visualization meshes - only blue brackets and their wire segments."""
        print(f"\n{'─'*50}")
        print("CREATING CLEAN VISUALIZATION")
        print(f"{'─'*50}")
        
        # Identify blue brackets (posterior teeth excluding first posterior on each side)
        blue_brackets = []
        
        # First, find first posterior on each side
        left_posterior_brackets = [b for b in self.bracket_positions 
                                 if b['position'][self.lr_axis] < 0 and b['tooth_type'] == 'posterior']
        right_posterior_brackets = [b for b in self.bracket_positions 
                                  if b['position'][self.lr_axis] >= 0 and b['tooth_type'] == 'posterior']
        
        first_posterior_left = None
        first_posterior_right = None
        
        if left_posterior_brackets:
            first_posterior_left = max(left_posterior_brackets, 
                                     key=lambda b: b['position'][self.ap_axis])
        
        if right_posterior_brackets:
            first_posterior_right = max(right_posterior_brackets, 
                                      key=lambda b: b['position'][self.ap_axis])
        
        # Collect blue brackets (posterior but not first posterior on each side)
        for bracket in self.bracket_positions:
            if (bracket['tooth_type'] == 'posterior' and 
                bracket is not first_posterior_left and 
                bracket is not first_posterior_right):
                blue_brackets.append(bracket)
        
        print(f"Found {len(blue_brackets)} blue brackets")
        
        # Create wire mesh - only segments between blue brackets
        wire_mesh = o3d.geometry.TriangleMesh()
        
        if len(blue_brackets) > 1:
            blue_positions = [b['position'] for b in blue_brackets]
            
            for i in range(len(blue_positions) - 1):
                p1 = blue_positions[i]
                p2 = blue_positions[i + 1]
                segment_length = np.linalg.norm(p2 - p1)
                
                if segment_length < 0.01 or segment_length > 30.0:
                    continue
                
                cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                    radius=self.wire_radius,
                    height=segment_length,
                    resolution=16
                )
                
                # Orient cylinder
                z_axis = np.array([0, 0, 1])
                segment_dir = (p2 - p1) / segment_length
                
                rotation_axis = np.cross(z_axis, segment_dir)
                if np.linalg.norm(rotation_axis) > 1e-6:
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                    angle = np.arccos(np.clip(np.dot(z_axis, segment_dir), -1, 1))
                    R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
                    cylinder.rotate(R, center=[0, 0, 0])
                
                cylinder.translate((p1 + p2) / 2)
                wire_mesh += cylinder
            
            # Add spheres at blue bracket positions
            for pos in blue_positions:
                sphere = o3d.geometry.TriangleMesh.create_sphere(
                    radius=self.wire_radius * 1.2,
                    resolution=10
                )
                sphere.translate(pos)
                wire_mesh += sphere
        
        # Color wire gold
        wire_mesh.paint_uniform_color([0.85, 0.75, 0.45])
        wire_mesh.compute_vertex_normals()
        
        # Create bracket markers - color first and last blue brackets as yellow (corner teeth)
        bracket_markers = o3d.geometry.TriangleMesh()
        
        for i, bracket in enumerate(blue_brackets):
            # Create bracket box
            bracket_mesh = o3d.geometry.TriangleMesh.create_box(1.5, 1.8, 1.0)
            bracket_mesh.translate([-0.75, -0.9, -0.5])
            
            # Orient bracket
            if 'normal' in bracket and np.linalg.norm(bracket['normal']) > 0:
                z_axis = np.array([0, 0, 1])
                normal = bracket['normal']
                
                rotation_axis = np.cross(z_axis, normal)
                if np.linalg.norm(rotation_axis) > 1e-6:
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                    angle = np.arccos(np.clip(np.dot(z_axis, normal), -1, 1))
                    R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
                    bracket_mesh.rotate(R, center=[0, 0, 0])
            
            bracket_mesh.translate(bracket['position'])
            
            # Color: Yellow for first and last (corner teeth), Blue for middle teeth
            if i == 0 or i == len(blue_brackets) - 1:
                # First and last blue brackets = Yellow (corner teeth)
                bracket_mesh.paint_uniform_color([0.9, 0.8, 0.1])  # Yellow
            else:
                # Middle blue brackets = Blue
                bracket_mesh.paint_uniform_color([0.1, 0.3, 0.8])  # Blue
            
            bracket_markers += bracket_mesh
        
        bracket_markers.compute_vertex_normals()
        
        print(f"✓ Created {len(blue_brackets)} blue brackets")
        print(f"✓ First and last brackets colored YELLOW (corner teeth)")
        print(f"✓ Middle brackets colored BLUE")
        print("✓ Wire segments only between blue brackets")
        print("✓ No wire/brackets for incisors, canines, or first premolars")
        
        return wire_mesh, bracket_markers
    
    def visualize_clean_system(self, components):
        """Clean visualization."""
        print(f"\n{'─'*50}")
        print("CLEAN LINGUAL ORTHODONTIC VISUALIZATION")
        print(f"{'─'*50}")
        
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"Clean Lingual Wire - {self.arch_type.upper()} ARCH",
            width=1400,
            height=900
        )
        
        # Add components
        if components['mesh']:
            vis.add_geometry(components['mesh'])
        
        if components['wire_mesh']:
            vis.add_geometry(components['wire_mesh'])
        
        if components['bracket_markers']:
            vis.add_geometry(components['bracket_markers'])
        
        # Add coordinate system
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=15.0,
            origin=[0, 0, 0]
        )
        vis.add_geometry(coord_frame)
        
        # Configure view
        opt = vis.get_render_option()
        opt.background_color = np.array([0.92, 0.92, 0.95])
        opt.mesh_show_back_face = True
        opt.point_size = 3.0
        
        # Set view
        ctr = vis.get_view_control()
        ctr.set_zoom(0.7)
        
        print("\nVisualization Controls:")
        print("  • Mouse Left: Rotate view")
        print("  • Mouse Scroll: Zoom in/out")
        print("  • Mouse Right: Pan view")
        print("  • Q: Close window")
        
        print("\nCLEAN DISPLAY:")
        print("  • YELLOW brackets: First and last remaining posterior (corner teeth)")
        print("  • BLUE brackets: Middle remaining posterior teeth")
        print("  • GOLD wire: Only segments connecting all visible brackets")
        print("  • No display for: Incisors, canines, first premolars")
        print("  • Realistic hardware representation")
        
        # Count final results
        incisor_count = sum(1 for b in self.bracket_positions if b['tooth_type'] == 'incisor')
        canine_count = sum(1 for b in self.bracket_positions if b['tooth_type'] == 'canine')
        
        print(f"\nClassification Results:")
        print(f"  • Incisors detected: {incisor_count}")
        print(f"  • Canines detected: {canine_count}")
        print(f"  • Total teeth: {len(self.bracket_positions)}")
        
        # Run visualization
        vis.run()
        vis.destroy_window()
    
    def generate_clean_lingual_wire(self):
        """Main method to generate clean lingual wire."""
        
        # Step 1: Load mesh
        if not self.load_mesh():
            return None
        
        # Step 2: Simple tooth detection
        if not self.detect_teeth_simple():
            print("\n✗ Failed to detect sufficient teeth")
            return None
        
        # Step 3: Calculate lingual bracket positions
        self.calculate_lingual_bracket_positions()
        
        # Step 4: Create wire path
        if not self.create_smooth_wire_path():
            print("\n✗ Failed to create wire path")
            return None
        
        # Step 5: Create clean visualization
        wire_mesh, bracket_markers = self.create_clean_visualization_meshes()
        
        print(f"\n{'='*70}")
        print("CLEAN LINGUAL WIRE GENERATION COMPLETE")
        print(f"{'='*70}")
        
        return {
            'mesh': self.mesh,
            'wire_mesh': wire_mesh,
            'bracket_markers': bracket_markers,
            'bracket_positions': self.bracket_positions,
            'wire_path': self.wire_path
        }


def main():
    """Main execution for clean lingual wire generation."""
    print("\n" + "="*70)
    print("CLEAN LINGUAL ORTHODONTIC WIRE GENERATOR")
    print("*** ERROR FREE - BLUE BRACKETS ONLY ***")
    print("="*70 + "\n")
    
    # Configuration
    stl_path = "/Users/galala/STL con/assets/AyaKhairy_LowerJaw.stl"
    
    # Create clean generator
    generator = CleanLingualWireGenerator(
        stl_path=stl_path,
        arch_type='auto',
        wire_size='0.018'
    )
    
    # Generate clean lingual wire
    result = generator.generate_clean_lingual_wire()
    
    if result:
        print("\n✅ SUCCESS: Clean lingual wire generated!")
        
        # Show visualization
        generator.visualize_clean_system(result)
    else:
        print("\n❌ CRITICAL ERROR: Failed to generate clean lingual wire")


if __name__ == "__main__":
    main()