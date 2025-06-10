# Professional Orthodontic Wire Generator
# Analyzes dental STL to create clinically accurate arch wires
# pip install open3d numpy scipy

import numpy as np
import open3d as o3d
import os
from scipy import interpolate, signal
from scipy.spatial import distance_matrix
import csv

class OrthodonticWireGenerator:
    """Professional orthodontic wire generator with clinical considerations."""
    
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
    
    # Standard bracket heights by tooth type (mm from incisal/occlusal)
    BRACKET_HEIGHTS = {
        'upper': {
            'central_incisor': 4.5,
            'lateral_incisor': 4.0,
            'canine': 5.0,
            'first_premolar': 4.5,
            'second_premolar': 4.5,
            'first_molar': 3.5,
            'second_molar': 3.0
        },
        'lower': {
            'central_incisor': 4.0,
            'lateral_incisor': 4.0,
            'canine': 5.0,
            'first_premolar': 4.5,
            'second_premolar': 4.5,
            'first_molar': 3.5,
            'second_molar': 3.0
        }
    }
    
    def __init__(self, stl_path, arch_type='auto', wire_size='0.018'):
        """Initialize the wire generator."""
        self.stl_path = stl_path
        self.wire_size = wire_size
        self.wire_radius = self.WIRE_SIZES.get(wire_size, 0.4572) / 2
        
        # Detect arch type if auto
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
        print(f"\n--- Loading Orthodontic Model ---")
        print(f"File: {os.path.basename(self.stl_path)}")
        print(f"Arch type: {self.arch_type.upper()}")
        print(f"Wire size: {self.wire_size}")
        
        try:
            self.mesh = o3d.io.read_triangle_mesh(self.stl_path)
            if not self.mesh.has_triangles():
                raise ValueError("No triangles found in mesh")
            
            # Compute normals
            self.mesh.compute_vertex_normals()
            self.mesh.compute_triangle_normals()
            
            # Color teeth naturally
            self.mesh.paint_uniform_color([0.96, 0.94, 0.90])
            
            print(f"Loaded: {len(self.mesh.vertices)} vertices, {len(self.mesh.triangles)} triangles")
            return True
            
        except Exception as e:
            print(f"Error loading mesh: {e}")
            return False
    
    def detect_individual_teeth(self):
        """Detect individual teeth using curvature and clustering analysis."""
        print("\n--- Detecting Individual Teeth ---")
        
        vertices = np.asarray(self.mesh.vertices)
        print(f"Total vertices: {len(vertices)}")
        
        # Get mesh bounds
        bbox = self.mesh.get_axis_aligned_bounding_box()
        center = self.mesh.get_center()
        extent = bbox.get_extent()
        
        # Determine coordinate axes
        self.lr_axis = np.argmax(extent)  # Left-Right (widest)
        self.height_axis = np.argmin(extent)  # Height (smallest)
        self.ap_axis = 3 - self.lr_axis - self.height_axis  # Anterior-Posterior
        
        print(f"Axes - LR: {self.lr_axis}, AP: {self.ap_axis}, Height: {self.height_axis}")
        print(f"Extent - LR: {extent[self.lr_axis]:.1f}, AP: {extent[self.ap_axis]:.1f}, Height: {extent[self.height_axis]:.1f}")
        
        # Sample vertices at crown level
        if self.arch_type == 'upper':
            crown_height = np.min(vertices[:, self.height_axis]) + extent[self.height_axis] * 0.3
        else:
            crown_height = np.max(vertices[:, self.height_axis]) - extent[self.height_axis] * 0.3
        
        height_tolerance = extent[self.height_axis] * 0.15
        crown_mask = np.abs(vertices[:, self.height_axis] - crown_height) < height_tolerance
        crown_vertices = vertices[crown_mask]
        
        print(f"Crown level vertices: {len(crown_vertices)}")
        
        # Use angular segmentation for uniform tooth detection
        num_teeth = 16  # Slightly more segments to ensure we don't miss teeth
        
        # Convert to polar coordinates
        centered_vertices = crown_vertices - center
        angles = np.arctan2(centered_vertices[:, self.ap_axis], centered_vertices[:, self.lr_axis])
        radii = np.sqrt(centered_vertices[:, self.lr_axis]**2 + centered_vertices[:, self.ap_axis]**2)
        
        # Create uniform angular segments
        self.teeth = []
        angle_step = 2 * np.pi / (num_teeth + 2)  # +2 for posterior gap
        
        # Start from posterior left and go around
        start_angle = -np.pi + angle_step  # Skip first segment (posterior gap)
        
        for i in range(num_teeth):
            segment_start = start_angle + i * angle_step
            segment_end = segment_start + angle_step
            
            # Get vertices in this segment
            angle_mask = np.logical_and(angles >= segment_start, angles < segment_end)
            segment_vertices = crown_vertices[angle_mask]
            
            if len(segment_vertices) > 30:
                # Calculate tooth center as outermost point average
                segment_radii = radii[angle_mask]
                if len(segment_radii) > 0:
                    radius_threshold = np.percentile(segment_radii, 70)
                    outer_mask = segment_radii >= radius_threshold
                    outer_vertices = segment_vertices[outer_mask]
                    
                    if len(outer_vertices) > 0:
                        tooth_center = np.mean(outer_vertices, axis=0)
                    else:
                        tooth_center = np.mean(segment_vertices, axis=0)
                else:
                    tooth_center = np.mean(segment_vertices, axis=0)
                
                # Skip if too close to center
                dist_from_center = np.linalg.norm(tooth_center[[self.lr_axis, self.ap_axis]] - center[[self.lr_axis, self.ap_axis]])
                if dist_from_center < 10.0:
                    continue
                
                tooth_info = {
                    'center': tooth_center,
                    'vertices': segment_vertices,
                    'label': len(self.teeth),
                    'type': 'unknown',
                    'angle': np.arctan2(tooth_center[self.ap_axis] - center[self.ap_axis], 
                                      tooth_center[self.lr_axis] - center[self.lr_axis]),
                    'ap_position': tooth_center[self.ap_axis]
                }
                self.teeth.append(tooth_info)
        
        # Sort teeth by angle to ensure proper order around the arch
        self.teeth.sort(key=lambda t: t['angle'])
        
        # ADAPTIVE CLASSIFICATION based on position in the arch
        num_teeth = len(self.teeth)
        
        if num_teeth > 0:
            # Find which teeth are in the anterior (front) region
            # The anterior teeth have the highest AP values
            ap_values = [t['ap_position'] for t in self.teeth]
            max_ap = max(ap_values)
            min_ap = min(ap_values)
            ap_range = max_ap - min_ap
            
            # Also consider the arch curvature
            # In a horseshoe arch, teeth are distributed along the curve
            # We'll use both position along the curve AND AP position
            
            for i, tooth in enumerate(self.teeth):
                # Position along the sorted arch (0 to 1)
                position_ratio = i / (num_teeth - 1) if num_teeth > 1 else 0.5
                
                # AP position ratio (0 = posterior, 1 = anterior)
                ap_ratio = (tooth['ap_position'] - min_ap) / (ap_range + 1e-6)
                
                # Combine both metrics for robust classification
                # Teeth in the middle of the arch AND with high AP values are incisors
                middle_distance = abs(position_ratio - 0.5)
                
                # Classification logic
                if middle_distance < 0.15 and ap_ratio > 0.6:
                    # Central region with anterior position = incisors
                    if middle_distance < 0.08:
                        tooth['type'] = 'central_incisor'
                    else:
                        tooth['type'] = 'lateral_incisor'
                elif middle_distance < 0.25 and ap_ratio > 0.5:
                    # Slightly off-center with good AP position = canines
                    tooth['type'] = 'canine'
                elif position_ratio < 0.15 or position_ratio > 0.85:
                    # Ends of the arch = molars
                    tooth['type'] = 'second_molar'
                elif position_ratio < 0.30 or position_ratio > 0.70:
                    # Near the ends = first molars/second premolars
                    if position_ratio < 0.22 or position_ratio > 0.78:
                        tooth['type'] = 'first_molar'
                    else:
                        tooth['type'] = 'second_premolar'
                else:
                    # Everything else = first premolars
                    tooth['type'] = 'first_premolar'
            
            # Verify distribution makes sense
            type_counts = {}
            for tooth in self.teeth:
                type_counts[tooth['type']] = type_counts.get(tooth['type'], 0) + 1
            
            # If we have no incisors, force the most anterior teeth to be incisors
            if type_counts.get('central_incisor', 0) + type_counts.get('lateral_incisor', 0) == 0:
                # Find the 4 most anterior teeth
                anterior_teeth = sorted(self.teeth, key=lambda t: t['ap_position'], reverse=True)[:4]
                
                # Also check they're in the middle portion of the arch
                middle_teeth = []
                for t in anterior_teeth:
                    # Find the index of this tooth in the sorted list
                    for idx, tooth in enumerate(self.teeth):
                        if tooth is t:  # Same object reference
                            position_ratio = idx / (num_teeth - 1) if num_teeth > 1 else 0.5
                            if 0.3 < position_ratio < 0.7:
                                middle_teeth.append(t)
                            break
                
                if len(middle_teeth) >= 2:
                    middle_teeth[0]['type'] = 'central_incisor'
                    middle_teeth[1]['type'] = 'central_incisor'
                    if len(middle_teeth) >= 4:
                        middle_teeth[2]['type'] = 'lateral_incisor'
                        middle_teeth[3]['type'] = 'lateral_incisor'
        
        print(f"Detected {len(self.teeth)} teeth")
        for i, tooth in enumerate(self.teeth):
            print(f"  Tooth {i+1}: {tooth['type']} at angle {np.degrees(tooth['angle']):.1f}°, AP: {tooth['ap_position']:.1f}")
        
        # Final verification
        type_counts = {}
        for tooth in self.teeth:
            type_counts[tooth['type']] = type_counts.get(tooth['type'], 0) + 1
        print(f"Final tooth type distribution: {type_counts}")
        
        return len(self.teeth) >= 6
    
    def _classify_tooth(self, tooth_center, arch_center):
        """Classify tooth type based on position in the arch."""
        # Get position relative to arch center
        lr_pos = tooth_center[self.lr_axis] - arch_center[self.lr_axis]
        ap_pos = tooth_center[self.ap_axis] - arch_center[self.ap_axis]
        
        # Calculate angle from posterior (back) to anterior (front)
        # In lower jaw, anterior teeth have positive AP values
        angle = np.degrees(np.arctan2(ap_pos, lr_pos))
        
        # Normalize angle to 0-360
        if angle < 0:
            angle += 360
        
        # For lower jaw, the arch goes:
        # Left molars (150-180°) → Left premolars (120-150°) → Left canine (60-120°) 
        # → Incisors (0-60° and 300-360°) → Right canine (240-300°) 
        # → Right premolars (210-240°) → Right molars (180-210°)
        
        # Determine tooth type based on angular position
        if (angle >= 0 and angle < 30) or (angle >= 330 and angle <= 360):
            # Front center - incisors
            if abs(angle) < 15 or angle > 345:
                return 'central_incisor'
            else:
                return 'lateral_incisor'
        elif (angle >= 30 and angle < 60) or (angle >= 300 and angle < 330):
            # Slightly to the side - lateral incisors or canines
            if angle < 45 or angle > 315:
                return 'lateral_incisor'
            else:
                return 'canine'
        elif (angle >= 60 and angle < 90) or (angle >= 270 and angle < 300):
            # Corner position - canines
            return 'canine'
        elif (angle >= 90 and angle < 120) or (angle >= 240 and angle < 270):
            # Behind canines - first premolars
            return 'first_premolar'
        elif (angle >= 120 and angle < 140) or (angle >= 220 and angle < 240):
            # Mid-arch - second premolars
            return 'second_premolar'
        elif (angle >= 140 and angle < 160) or (angle >= 200 and angle < 220):
            # Back of arch - first molars
            return 'first_molar'
        else:
            # Very back - second molars
            return 'second_molar'
    
    def _sort_teeth_along_arch(self):
        """Sort teeth along the dental arch from left to right."""
        if not self.teeth:
            return []
        
        # Sort by left-right position (simple approach)
        sorted_teeth = sorted(self.teeth, key=lambda t: t['center'][self.lr_axis])
        
        # Verify the sorting makes sense by checking distances
        # Reorder if needed to follow the arch continuously
        final_order = [sorted_teeth[0]]
        remaining = sorted_teeth[1:]
        
        while remaining:
            last_tooth = final_order[-1]
            # Find nearest remaining tooth
            min_dist = float('inf')
            nearest_tooth = None
            nearest_idx = -1
            
            for idx, tooth in enumerate(remaining):
                dist = np.linalg.norm(tooth['center'] - last_tooth['center'])
                if dist < min_dist:
                    min_dist = dist
                    nearest_tooth = tooth
                    nearest_idx = idx
            
            if nearest_tooth is not None:
                final_order.append(nearest_tooth)
                remaining.pop(nearest_idx)
            else:
                break
        
        return final_order
    
    def calculate_bracket_positions(self):
        """Calculate clinical bracket positions for each tooth."""
        print("\n--- Calculating Bracket Positions ---")
        
        self.bracket_positions = []
        vertices = np.asarray(self.mesh.vertices)
        mesh_center = self.mesh.get_center()
        
        for i, tooth in enumerate(self.teeth):
            # Get bracket height for tooth type
            bracket_height = self.BRACKET_HEIGHTS[self.arch_type].get(
                tooth['type'], 4.5)  # Default 4.5mm
            
            tooth_vertices = tooth['vertices']
            tooth_center = tooth['center']
            
            # Find the OUTER surface of the tooth (buccal/labial)
            # Step 1: Get vertices at bracket height
            if self.arch_type == 'upper':
                target_height = np.min(tooth_vertices[:, self.height_axis]) + bracket_height
            else:
                target_height = np.max(tooth_vertices[:, self.height_axis]) - bracket_height
            
            # Wider tolerance to get more vertices
            height_tolerance = 2.0  # 2mm tolerance
            height_mask = np.abs(tooth_vertices[:, self.height_axis] - target_height) < height_tolerance
            bracket_level_vertices = tooth_vertices[height_mask]
            
            if len(bracket_level_vertices) < 10:
                # Not enough vertices, use robust fallback
                bracket_pos = tooth_center.copy()
                bracket_pos[self.height_axis] = target_height
            else:
                # Step 2: Find the true facial surface more robustly
                # Key insight: The facial surface is the one FURTHEST from the arch center
                # when projected onto the horizontal plane
                
                # Calculate vector from arch center to each vertex
                vectors_from_center = bracket_level_vertices - mesh_center
                vectors_from_center[:, self.height_axis] = 0  # Project to horizontal plane
                
                # Calculate distances in horizontal plane
                horizontal_distances = np.linalg.norm(vectors_from_center, axis=1)
                
                # Find the outermost vertices - use top 10% to be robust
                percentile_90 = np.percentile(horizontal_distances, 90)
                outer_mask = horizontal_distances >= percentile_90
                outer_vertices = bracket_level_vertices[outer_mask]
                
                if len(outer_vertices) > 3:
                    # Use the centroid of outermost vertices for stability
                    bracket_pos = np.mean(outer_vertices, axis=0)
                else:
                    # Fallback to single outermost point
                    bracket_pos = bracket_level_vertices[np.argmax(horizontal_distances)]
            
            # Step 3: Apply proper offset to ensure wire doesn't intersect
            # Calculate the outward direction from tooth center through bracket position
            tooth_to_bracket = bracket_pos - tooth_center
            tooth_to_bracket[self.height_axis] = 0  # Keep horizontal
            
            if np.linalg.norm(tooth_to_bracket) > 0:
                # This direction should point outward
                outward_direction = tooth_to_bracket / np.linalg.norm(tooth_to_bracket)
            else:
                # Fallback: use direction from arch center
                direction_from_center = bracket_pos - mesh_center
                direction_from_center[self.height_axis] = 0
                if np.linalg.norm(direction_from_center) > 0:
                    outward_direction = direction_from_center / np.linalg.norm(direction_from_center)
                else:
                    outward_direction = np.array([0, 1, 0])  # Default
            
            # Apply offset: bracket base (1mm) + wire clearance (1.5mm) = 2.5mm total
            total_offset = 2.5
            bracket_pos = bracket_pos + outward_direction * total_offset
            
            # Store bracket information
            bracket_info = {
                'position': bracket_pos,
                'tooth_type': tooth['type'],
                'tooth_index': i,
                'tooth_center': tooth_center,
                'direction': outward_direction
            }
            self.bracket_positions.append(bracket_info)
        
        print(f"Calculated {len(self.bracket_positions)} bracket positions")
        
        # Verify brackets are properly positioned
        for i, bracket in enumerate(self.bracket_positions):
            dist_from_center = np.linalg.norm(
                bracket['position'][:2] - mesh_center[:2]
            )
            dist_from_tooth = np.linalg.norm(
                bracket['position'] - bracket['tooth_center']
            )
            print(f"  Bracket {i+1} ({bracket['tooth_type']}): "
                  f"{dist_from_center:.1f}mm from arch center, "
                  f"{dist_from_tooth:.1f}mm from tooth center")
        
        return len(self.bracket_positions) > 0
    
    def create_ideal_archform(self):
        """Create ideal archform based on detected teeth and orthodontic principles."""
        print("\n--- Creating Ideal Archform ---")
        
        if not self.bracket_positions:
            return None
        
        # Extract bracket positions
        positions = np.array([b['position'] for b in self.bracket_positions])
        mesh_center = self.mesh.get_center()
        
        # Sort positions by angle to ensure proper ordering
        angles = []
        for pos in positions:
            angle = np.arctan2(
                pos[self.ap_axis] - mesh_center[self.ap_axis],
                pos[self.lr_axis] - mesh_center[self.lr_axis]
            )
            angles.append(angle)
        
        # Sort by angle
        sorted_indices = np.argsort(angles)
        sorted_positions = positions[sorted_indices]
        
        # Find the largest gap (posterior opening)
        max_gap_dist = 0
        max_gap_idx = -1
        
        for i in range(len(sorted_positions)):
            next_i = (i + 1) % len(sorted_positions)
            dist = np.linalg.norm(sorted_positions[next_i] - sorted_positions[i])
            if dist > max_gap_dist:
                max_gap_dist = dist
                max_gap_idx = i
        
        # Reorder to start after the gap
        if max_gap_idx >= 0 and max_gap_dist > 20.0:
            sorted_positions = np.vstack([
                sorted_positions[max_gap_idx+1:],
                sorted_positions[:max_gap_idx+1]
            ])
        
        # Create smooth archform with better interpolation
        # Use more points for smoother curve
        num_interp_points = len(sorted_positions) * 3
        
        # Calculate cumulative distances for parameterization
        distances = [0]
        for i in range(1, len(sorted_positions)):
            dist = np.linalg.norm(sorted_positions[i] - sorted_positions[i-1])
            distances.append(distances[-1] + dist)
        
        # Check if last segment closes the loop (it shouldn't)
        last_to_first = np.linalg.norm(sorted_positions[0] - sorted_positions[-1])
        if last_to_first < 20.0:  # If it's a closed loop, open it
            # Remove the last point to prevent closure
            sorted_positions = sorted_positions[:-1]
            distances = distances[:-1]
        
        if len(sorted_positions) < 3:
            return sorted_positions
        
        # Normalize distances
        if distances[-1] > 0:
            t_original = np.array(distances) / distances[-1]
        else:
            t_original = np.linspace(0, 1, len(sorted_positions))
        
        # Create new parameter values for interpolation
        t_smooth = np.linspace(0, 1, num_interp_points)
        
        # Use cubic spline interpolation with smoothing
        # Less smoothing for better tooth following
        try:
            from scipy import interpolate
            
            # Create periodic=False to prevent closing the curve
            lr_spline = interpolate.UnivariateSpline(
                t_original, sorted_positions[:, self.lr_axis], k=3, s=1.0)
            ap_spline = interpolate.UnivariateSpline(
                t_original, sorted_positions[:, self.ap_axis], k=3, s=1.0)
            height_spline = interpolate.UnivariateSpline(
                t_original, sorted_positions[:, self.height_axis], k=3, s=0.1)
            
            # Generate smooth archform
            smooth_arch = np.zeros((len(t_smooth), 3))
            smooth_arch[:, self.lr_axis] = lr_spline(t_smooth)
            smooth_arch[:, self.ap_axis] = ap_spline(t_smooth)
            smooth_arch[:, self.height_axis] = height_spline(t_smooth)
            
        except:
            # Fallback to linear interpolation
            smooth_arch = []
            for i in range(len(sorted_positions) - 1):
                p1 = sorted_positions[i]
                p2 = sorted_positions[i + 1]
                
                # Add intermediate points
                for j in range(3):
                    t = j / 3.0
                    interp_point = p1 * (1 - t) + p2 * t
                    smooth_arch.append(interp_point)
            
            smooth_arch.append(sorted_positions[-1])
            smooth_arch = np.array(smooth_arch)
        
        print(f"Created archform with {len(smooth_arch)} points")
        print(f"Archform is open at posterior (no closing segment)")
        
        return smooth_arch
    
    def add_clinical_bends(self, archform):
        """Add clinical bends at appropriate locations - INCLUDING ALL BRACKETS."""
        print("\n--- Adding Clinical Bends ---")
        
        # CRITICAL: Include ALL bracket positions to ensure complete wire
        clinical_wire = []
        
        # Add EVERY bracket position - no filtering
        print(f"Adding all {len(self.bracket_positions)} bracket positions to wire")
        
        for i, bracket in enumerate(self.bracket_positions):
            bracket_pos = bracket['position'].copy()
            
            clinical_wire.append({
                'position': bracket_pos,
                'type': 'bracket',
                'tooth_type': bracket['tooth_type'],
                'tooth_index': bracket['tooth_index'],
                'original_index': i
            })
        
        # Sort clinical wire points to follow the arch properly
        if len(clinical_wire) > 2:
            # Sort by angle around the arch to ensure proper order
            mesh_center = self.mesh.get_center()
            
            for point in clinical_wire:
                pos = point['position']
                angle = np.arctan2(
                    pos[self.ap_axis] - mesh_center[self.ap_axis],
                    pos[self.lr_axis] - mesh_center[self.lr_axis]
                )
                point['angle'] = angle
            
            # Sort by angle
            clinical_wire.sort(key=lambda p: p['angle'])
            
            # Find the posterior gap to properly order the wire
            max_gap = 0
            gap_index = -1
            
            for i in range(len(clinical_wire)):
                next_i = (i + 1) % len(clinical_wire)
                dist = np.linalg.norm(
                    clinical_wire[next_i]['position'] - clinical_wire[i]['position']
                )
                
                if dist > max_gap:
                    max_gap = dist
                    gap_index = i
            
            print(f"Found posterior gap of {max_gap:.1f}mm at index {gap_index}")
            
            # Reorder to start after the gap (if gap is large enough)
            if gap_index >= 0 and max_gap > 15.0:
                clinical_wire = clinical_wire[gap_index+1:] + clinical_wire[:gap_index+1]
                print(f"Reordered wire to start after posterior gap")
        
        # Verify we have all teeth represented
        print(f"\nClinical wire summary:")
        print(f"Total bend points: {len(clinical_wire)}")
        
        type_counts = {}
        for point in clinical_wire:
            tooth_type = point['tooth_type']
            type_counts[tooth_type] = type_counts.get(tooth_type, 0) + 1
        
        print("Tooth types in wire:")
        for tooth_type, count in sorted(type_counts.items()):
            print(f"  {tooth_type}: {count}")
        
        # Verify terminal teeth are included
        if clinical_wire:
            print(f"\nWire starts with: {clinical_wire[0]['tooth_type']}")
            print(f"Wire ends with: {clinical_wire[-1]['tooth_type']}")
        
        return clinical_wire
    
    def calculate_bending_instructions(self, clinical_wire):
        """Calculate wire bending machine instructions."""
        print("\n--- Calculating Bending Instructions ---")
        
        if len(clinical_wire) < 2:
            return None
        
        instructions = []
        cumulative_feed = 0.0
        
        for i in range(len(clinical_wire)):
            point = clinical_wire[i]['position']
            
            # Calculate feed distance
            if i > 0:
                segment_length = np.linalg.norm(
                    point - clinical_wire[i-1]['position'])
                cumulative_feed += segment_length
            
            # Calculate bend angle
            if 0 < i < len(clinical_wire) - 1:
                # Three points for angle calculation
                p1 = clinical_wire[i-1]['position']
                p2 = point
                p3 = clinical_wire[i+1]['position']
                
                # Vectors
                v1 = p1 - p2
                v2 = p3 - p2
                
                # Normalize
                v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
                v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
                
                # Calculate angle
                dot_product = np.clip(np.dot(v1_norm, v2_norm), -1, 1)
                angle_rad = np.arccos(dot_product)
                bend_angle = np.degrees(angle_rad)
                
                # Determine bend direction
                cross = np.cross(v1_norm, v2_norm)
                if cross[self.height_axis] < 0:
                    bend_angle = -bend_angle
            else:
                bend_angle = 0.0
            
            # Create instruction
            instruction = {
                'index': i,
                'feed_distance': cumulative_feed,
                'bend_angle': bend_angle,
                'position': point,
                'type': clinical_wire[i].get('type', 'standard'),
                'tooth_type': clinical_wire[i].get('tooth_type', 'unknown')
            }
            instructions.append(instruction)
        
        return instructions
    
    def create_wire_mesh(self, clinical_wire):
        """Create 3D mesh visualization of the wire."""
        print("\n--- Creating Wire Mesh ---")
        
        if not clinical_wire:
            return None
        
        positions = [p['position'] for p in clinical_wire]
        wire_mesh = o3d.geometry.TriangleMesh()
        
        # First, identify if there's a posterior gap by checking all distances
        distances = []
        for i in range(len(positions)):
            next_i = (i + 1) % len(positions)
            dist = np.linalg.norm(positions[next_i] - positions[i])
            distances.append(dist)
        
        # Find the largest gap
        max_dist = max(distances)
        max_dist_idx = distances.index(max_dist)
        
        print(f"Distances between points: min={min(distances):.1f}mm, max={max_dist:.1f}mm")
        
        # Determine gap threshold - use 15mm or 2x average distance
        avg_dist = sum(distances) / len(distances)
        gap_threshold = max(15.0, avg_dist * 2)
        
        print(f"Gap threshold: {gap_threshold:.1f}mm")
        
        # Create all segments except those crossing gaps
        segments_created = 0
        gaps_found = 0
        
        for i in range(len(positions)):
            # For the last position, check if it connects to the first
            if i == len(positions) - 1:
                # Don't connect last to first if it would cross the posterior gap
                if distances[i] > gap_threshold:
                    print(f"Not connecting last point to first (gap: {distances[i]:.1f}mm)")
                    gaps_found += 1
                    continue
                else:
                    p1 = positions[i]
                    p2 = positions[0]
            else:
                p1 = positions[i]
                p2 = positions[i + 1]
            
            segment_vec = p2 - p1
            segment_length = np.linalg.norm(segment_vec)
            
            # Skip if this is a gap
            if segment_length > gap_threshold:
                print(f"Skipping gap between points {i} and {(i+1)%len(positions)} (length: {segment_length:.1f}mm)")
                gaps_found += 1
                continue
            
            if segment_length < 0.01:  # Skip very short segments
                continue
            
            # Create cylinder for this segment
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=self.wire_radius,
                height=segment_length,
                resolution=10
            )
            
            # Orient cylinder
            z_axis = np.array([0, 0, 1])
            segment_dir = segment_vec / segment_length
            
            rotation_axis = np.cross(z_axis, segment_dir)
            if np.linalg.norm(rotation_axis) > 1e-6:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.arccos(np.clip(np.dot(z_axis, segment_dir), -1, 1))
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                    rotation_axis * angle)
                cylinder.rotate(R, center=[0, 0, 0])
            
            # Position cylinder
            cylinder.translate((p1 + p2) / 2)
            wire_mesh += cylinder
            segments_created += 1
            
            # Add sphere at p2 for smooth joints
            sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=self.wire_radius * 1.05, resolution=8)
            sphere.translate(p2)
            wire_mesh += sphere
        
        # Add larger terminal spheres at wire ends
        # Find actual wire endpoints (where gaps are)
        for i in range(len(positions)):
            is_endpoint = False
            
            # Check if this point has a gap before it
            prev_i = (i - 1) % len(positions)
            if np.linalg.norm(positions[i] - positions[prev_i]) > gap_threshold:
                is_endpoint = True
            
            # Check if this point has a gap after it
            next_i = (i + 1) % len(positions)
            if np.linalg.norm(positions[next_i] - positions[i]) > gap_threshold:
                is_endpoint = True
            
            if is_endpoint:
                end_sphere = o3d.geometry.TriangleMesh.create_sphere(
                    radius=self.wire_radius * 1.3, resolution=10)
                end_sphere.translate(positions[i])
                wire_mesh += end_sphere
        
        print(f"Created {segments_created} wire segments, found {gaps_found} gaps")
        
        # Color the wire RED
        wire_mesh.paint_uniform_color([1.0, 0.0, 0.0])
        wire_mesh.compute_vertex_normals()
        
        return wire_mesh
    
    def create_bracket_markers(self):
        """Create visual markers for bracket positions."""
        markers = o3d.geometry.TriangleMesh()
        
        for bracket in self.bracket_positions:
            # Create small cube for bracket
            cube = o3d.geometry.TriangleMesh.create_box(1.5, 2.0, 1.0)
            
            # Center the cube
            cube.translate([-0.75, -1.0, -0.5])
            
            # Position at bracket location
            cube.translate(bracket['position'])
            
            # Color based on tooth type
            if 'incisor' in bracket['tooth_type']:
                cube.paint_uniform_color([0.2, 0.2, 0.8])  # Blue
            elif 'canine' in bracket['tooth_type']:
                cube.paint_uniform_color([0.8, 0.8, 0.2])  # Yellow
            else:
                cube.paint_uniform_color([0.2, 0.8, 0.2])  # Green
                
            markers += cube
        
        return markers
    
    def export_bending_instructions(self, instructions, filename=None):
        """Export bending instructions to CSV file."""
        if filename is None:
            filename = f"orthodontic_wire_{self.arch_type}_{self.wire_size}.csv"
        
        print(f"\n--- Exporting to {filename} ---")
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header
            writer.writerow([
                'Point_Index', 'Feed_Distance_mm', 'Bend_Angle_deg',
                'X_mm', 'Y_mm', 'Z_mm', 'Type', 'Tooth_Type'
            ])
            
            # Data
            for inst in instructions:
                writer.writerow([
                    inst['index'],
                    f"{inst['feed_distance']:.3f}",
                    f"{inst['bend_angle']:.2f}",
                    f"{inst['position'][0]:.3f}",
                    f"{inst['position'][1]:.3f}",
                    f"{inst['position'][2]:.3f}",
                    inst['type'],
                    inst['tooth_type']
                ])
        
        print(f"Exported {len(instructions)} bending instructions")
        
        # Print summary
        print("\n--- Wire Summary ---")
        print(f"Wire type: {self.wire_size}")
        print(f"Total length: {instructions[-1]['feed_distance']:.1f} mm")
        print(f"Number of bends: {len([i for i in instructions if abs(i['bend_angle']) > 5])}")
        
    def optimize_for_production(self, clinical_wire):
        """Optimize wire for production by reducing bend points intelligently."""
        print("\n--- Optimizing for Production ---")
        
        # For a complete wire, we should keep ALL points to ensure full coverage
        # Only optimize if we have way too many points
        
        current_points = len(clinical_wire)
        print(f"Current bend points: {current_points}")
        
        # Don't optimize if we have a reasonable number
        if current_points <= 20:
            print("Keeping all points for complete wire coverage")
            return clinical_wire
        
        # If we must optimize, ensure we keep critical points
        # Priority: Keep all molars, all incisors, all canines, some premolars
        
        critical_indices = set()
        
        # Always keep endpoints (terminal molars)
        critical_indices.add(0)
        critical_indices.add(len(clinical_wire) - 1)
        
        # Keep all incisors and canines
        for i, point in enumerate(clinical_wire):
            tooth_type = point['tooth_type']
            if 'incisor' in tooth_type or 'canine' in tooth_type or 'molar' in tooth_type:
                critical_indices.add(i)
        
        # Keep some premolars to maintain continuity
        premolar_indices = []
        for i, point in enumerate(clinical_wire):
            if 'premolar' in point['tooth_type']:
                premolar_indices.append(i)
        
        # Keep every other premolar
        for i in range(0, len(premolar_indices), 2):
            critical_indices.add(premolar_indices[i])
        
        # Build optimized wire
        selected_indices = sorted(list(critical_indices))
        optimized_wire = [clinical_wire[i] for i in selected_indices]
        
        print(f"Optimized from {len(clinical_wire)} to {len(optimized_wire)} bend points")
        
        # Verify we still have good coverage
        tooth_types = [p['tooth_type'] for p in optimized_wire]
        type_counts = {}
        for t in tooth_types:
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"Tooth types after optimization: {type_counts}")
        
        return optimized_wire
    
    def generate_wire(self):
        """Main method to generate the orthodontic wire."""
        print("\n" + "="*70)
        print("PROFESSIONAL ORTHODONTIC WIRE GENERATION")
        print("="*70)
        
        # Load mesh
        if not self.load_mesh():
            return None
        
        # Detect teeth
        if not self.detect_individual_teeth():
            print("Failed to detect teeth")
            return None
        
        # Calculate bracket positions
        if not self.calculate_bracket_positions():
            print("Failed to calculate bracket positions")
            return None
        
        # Create ideal archform
        archform = self.create_ideal_archform()
        if archform is None:
            print("Failed to create archform")
            return None
        
        # Add clinical bends
        clinical_wire = self.add_clinical_bends(archform)
        
        # Optimize for production
        clinical_wire = self.optimize_for_production(clinical_wire)
        
        # Calculate bending instructions
        instructions = self.calculate_bending_instructions(clinical_wire)
        
        # Create visualizations
        wire_mesh = self.create_wire_mesh(clinical_wire)
        bracket_markers = self.create_bracket_markers()
        
        # Export instructions
        if instructions:
            self.export_bending_instructions(instructions)
        
        # Production readiness assessment
        print("\n--- Production Readiness Assessment ---")
        print(f"✓ Wire type: {self.wire_size}")
        print(f"✓ Number of bends: {len(clinical_wire)}")
        print(f"✓ Total wire length: {instructions[-1]['feed_distance']:.1f}mm")
        print(f"✓ Teeth detected: {len(self.teeth)}")
        
        # Adaptive production readiness based on actual teeth
        expected_bends = len(self.teeth) // 1.5  # Roughly 1 bend per 1.5 teeth
        acceptable_range = (max(8, expected_bends - 3), min(20, expected_bends + 3))
        
        is_production_ready = acceptable_range[0] <= len(clinical_wire) <= acceptable_range[1]
        
        print(f"\nExpected bend range for {len(self.teeth)} teeth: {acceptable_range[0]}-{acceptable_range[1]}")
        print(f"Production Ready: {'YES' if is_production_ready else 'NO'}")
        
        if not is_production_ready:
            if len(clinical_wire) < acceptable_range[0]:
                print("Recommendation: Increase bend points for better tooth control")
            else:
                print("Recommendation: Reduce bend points or use multi-segment approach")
        
        return {
            'mesh': self.mesh,
            'wire_mesh': wire_mesh,
            'bracket_markers': bracket_markers,
            'instructions': instructions,
            'clinical_wire': clinical_wire,
            'production_ready': is_production_ready
        }
    
    def visualize(self, components):
        """Visualize the results."""
        print("\n--- Visualization ---")
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"Professional Orthodontic Wire - {self.arch_type.title()}",
            width=1400, height=900
        )
        
        # Add components
        if components['mesh']:
            vis.add_geometry(components['mesh'])
        if components['wire_mesh']:
            vis.add_geometry(components['wire_mesh'])
        if components['bracket_markers']:
            vis.add_geometry(components['bracket_markers'])
        
        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
        vis.add_geometry(coord_frame)
        
        # Render settings
        opt = vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.1])
        opt.mesh_show_back_face = True
        
        print("\nControls:")
        print("- Mouse: Rotate view")
        print("- Scroll: Zoom")
        print("- Colored boxes: Bracket positions")
        print("  Blue = Incisors, Yellow = Canines, Green = Premolars/Molars")
        
        vis.run()
        vis.destroy_window()


def main():
    """Main execution function."""
    # Configuration
    stl_path = "/Users/galala/STL con/assets/AyaKhairy_LowerJaw.stl"
    
    # Create wire generator
    generator = OrthodonticWireGenerator(
        stl_path=stl_path,
        arch_type='auto',  # or 'upper'/'lower'
        wire_size='0.018'  # Standard size
    )
    
    # Generate wire
    result = generator.generate_wire()
    
    if result:
        print("\n✅ SUCCESS: Professional orthodontic wire generated")
        generator.visualize(result)
    else:
        print("\n❌ FAILED: Could not generate wire")


if __name__ == "__main__":
    main()
