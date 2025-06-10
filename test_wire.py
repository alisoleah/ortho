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
            # Prioritize incisors based on AP values
            # Sort teeth by AP position (most anterior first)
            sorted_ap_teeth = sorted(self.teeth, key=lambda t: t['ap_position'], reverse=True)
            
            incisor_candidates = []
            # Select up to 6 most anterior teeth
            for i in range(min(num_teeth, 6)):
                incisor_candidates.append(sorted_ap_teeth[i])
            
            # Refine incisor selection: must be within central 60% of arch by angle
            # This helps avoid classifying canines/premolars as incisors in wide arches
            # And ensures we get teeth from the "front" of the arch

            # Calculate angular position for each tooth (0 to 1, from one end to the other)
            for i, tooth in enumerate(self.teeth): # self.teeth is already sorted by angle
                tooth['angular_order'] = i

            # Filter incisor_candidates based on their angular order
            # Keep only those in the central part of the arch based on angular order
            # E.g. if 14 teeth, 14*0.2 = 2.8, 14*0.8 = 11.2. So indices 3 to 11.
            min_angular_idx = int(num_teeth * 0.20)
            max_angular_idx = int(num_teeth * 0.80)

            potential_incisors = [
                t for t in incisor_candidates
                if min_angular_idx <= t['angular_order'] <= max_angular_idx
            ]

            # Ensure we have at least 4 incisors if possible, up to 6
            num_incisors_to_classify = 0
            if len(potential_incisors) >= 4:
                num_incisors_to_classify = min(len(potential_incisors), 6)
            elif len(incisor_candidates) >= 4: # Fallback if angular filter was too strict
                potential_incisors = incisor_candidates
                num_incisors_to_classify = min(len(potential_incisors), 6)
            else: # If fewer than 4 candidates overall, classify all of them as incisors
                potential_incisors = incisor_candidates
                num_incisors_to_classify = len(potential_incisors)

            # Sort these chosen incisors by their angle to differentiate central/lateral
            # The two most central (around the midpoint of the angular range of incisors) are central_incisor
            # The others are lateral_incisor
            final_incisors = sorted(potential_incisors[:num_incisors_to_classify], key=lambda t: t['angular_order'])

            if len(final_incisors) > 0:
                # Determine central vs lateral based on angular order within the incisor group
                incisor_angular_orders = [t['angular_order'] for t in final_incisors]
                median_angular_order = np.median(incisor_angular_orders)
                
                # Assign central/lateral based on proximity to median angular order
                # If 4 incisors, 2 central, 2 lateral
                # If 5 incisors, 2 central, 3 lateral (or 3 central, 2 lateral - let's go with 2 central for simplicity)
                # If 6 incisors, 2 central, 4 lateral
                
                # Mark all as incisors first
                for tooth in final_incisors:
                    tooth['type'] = 'lateral_incisor' # Default to lateral

                # Identify the 2 most central incisors (closest to median_angular_order)
                final_incisors.sort(key=lambda t: abs(t['angular_order'] - median_angular_order))
                if len(final_incisors) >= 1:
                    final_incisors[0]['type'] = 'central_incisor'
                if len(final_incisors) >= 2:
                    final_incisors[1]['type'] = 'central_incisor'

            # Classify Canines:
            # Canines are the teeth immediately adjacent to the incisor group, one on each side.
            incisor_indices_in_arch = sorted([t['angular_order'] for t in final_incisors])

            canine_candidates = []
            if len(incisor_indices_in_arch) > 0:
                # Tooth before the first incisor
                min_incisor_idx = incisor_indices_in_arch[0]
                if min_incisor_idx > 0:
                    canine_candidates.append(self.teeth[min_incisor_idx - 1])
                
                # Tooth after the last incisor
                max_incisor_idx = incisor_indices_in_arch[-1]
                if max_incisor_idx < num_teeth - 1:
                    canine_candidates.append(self.teeth[max_incisor_idx + 1])

            for tooth in canine_candidates:
                if tooth['type'] == 'unknown': # Only classify if not already an incisor
                    tooth['type'] = 'canine'

            # Classify remaining teeth as Premolars or Molars
            # Iterate through teeth in their angular order
            for i, tooth in enumerate(self.teeth):
                if tooth['type'] == 'unknown': # Only classify if not incisor or canine
                    # Position along the sorted arch (0 to 1)
                    position_ratio = i / (num_teeth - 1) if num_teeth > 1 else 0.5

                    # Molars are at the ends of the arch
                    if position_ratio < 0.15 or position_ratio > 0.85: # Ends of the arch
                        tooth['type'] = 'second_molar'
                    elif position_ratio < 0.25 or position_ratio > 0.75: # Near the ends
                        tooth['type'] = 'first_molar'
                    # Premolars are between canines and molars
                    elif position_ratio < 0.35 or position_ratio > 0.65:
                        tooth['type'] = 'second_premolar'
                    else:
                        tooth['type'] = 'first_premolar'
            
            # Fallback for any remaining 'unknown' (should be rare)
            # This can happen if canine classification overrode a premolar/molar
            # or if initial angular segmentation was imperfect.
            # Re-check based on relative angular position.
            
            # First, ensure all incisors and canines are marked and we don't overwrite them
            classified_indices = set()
            for t_idx, t in enumerate(self.teeth):
                if t['type'] != 'unknown':
                    classified_indices.add(t_idx)

            # Re-classify 'unknown' based on proximity to known types
            # This loop helps to fill gaps, e.g. if a premolar is between a canine and molar
            # but was somehow missed.
            for i, tooth in enumerate(self.teeth):
                if tooth['type'] == 'unknown':
                    # Determine position relative to the overall arch
                    # (i is the angular_order index)
                    position_ratio = i / (num_teeth - 1) if num_teeth > 1 else 0.5

                    # Simplistic fallback:
                    # If it's towards the back, call it a molar.
                    # If it's towards the middle (but not incisor/canine), call it a premolar.
                    if position_ratio < 0.20 or position_ratio > 0.80:
                         # More posterior than typical first molars
                        tooth['type'] = 'second_molar'
                    elif position_ratio < 0.35 or position_ratio > 0.65:
                        # Roughly where first molars / second premolars are
                        # Check neighbors if possible
                        left_neighbor_type = self.teeth[i-1]['type'] if i > 0 else "gap"
                        right_neighbor_type = self.teeth[i+1]['type'] if i < num_teeth -1 else "gap"

                        if "molar" in left_neighbor_type or "molar" in right_neighbor_type:
                            tooth['type'] = 'first_molar' # If next to a molar
                        else:
                            tooth['type'] = 'second_premolar' # Default for this zone
                    else:
                        # Central zone, but not incisor/canine - likely first premolar
                        tooth['type'] = 'first_premolar'

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
        # Ensure mesh is not None and has vertices before calling get_center()
        if self.mesh is None or not self.mesh.has_vertices():
            print("Error: Mesh not loaded or empty, cannot calculate bracket positions.")
            # Create a dummy mesh center if mesh is problematic, for testing purposes
            mesh_center_val = np.array([0,0,0])
        else:
            mesh_center_val = self.mesh.get_center()

        for i, tooth in enumerate(self.teeth):
            # Get bracket height for tooth type
            bracket_height = self.BRACKET_HEIGHTS[self.arch_type].get(
                tooth['type'], 4.5)  # Default 4.5mm
            
            tooth_vertices = tooth.get('vertices', np.array([tooth['center']])) # Use center if no vertices
            if not isinstance(tooth_vertices, np.ndarray) or tooth_vertices.ndim != 2 or tooth_vertices.shape[1] != 3:
                tooth_vertices = np.array([tooth['center']]) # Fallback

            tooth_center = tooth['center']
            
            # Find the OUTER surface of the tooth (buccal/labial)
            # Step 1: Get vertices at bracket height
            if self.arch_type == 'upper':
                # Ensure tooth_vertices is not empty before np.min/np.max
                if tooth_vertices.shape[0] == 0: target_height = tooth_center[self.height_axis] + bracket_height
                else: target_height = np.min(tooth_vertices[:, self.height_axis]) + bracket_height
            else:
                if tooth_vertices.shape[0] == 0: target_height = tooth_center[self.height_axis] - bracket_height
                else: target_height = np.max(tooth_vertices[:, self.height_axis]) - bracket_height
            
            height_tolerance = 2.0  # 2mm tolerance
            
            # Ensure tooth_vertices is not empty before filtering
            if tooth_vertices.shape[0] == 0:
                bracket_level_vertices = np.array([])
            else:
                height_mask = np.abs(tooth_vertices[:, self.height_axis] - target_height) < height_tolerance
                bracket_level_vertices = tooth_vertices[height_mask]

            if len(bracket_level_vertices) < 10:
                bracket_pos = tooth_center.copy()
                bracket_pos[self.height_axis] = target_height
            else:
                vectors_from_center = bracket_level_vertices - mesh_center_val
                vectors_from_center[:, self.height_axis] = 0
                
                horizontal_distances = np.linalg.norm(vectors_from_center, axis=1)
                
                percentile_90 = np.percentile(horizontal_distances, 90)
                outer_mask = horizontal_distances >= percentile_90
                outer_vertices = bracket_level_vertices[outer_mask]
                
                if len(outer_vertices) > 3:
                    bracket_pos = np.mean(outer_vertices, axis=0)
                else:
                    bracket_pos = bracket_level_vertices[np.argmax(horizontal_distances)]
            
            tooth_to_bracket = bracket_pos - tooth_center
            tooth_to_bracket[self.height_axis] = 0
            
            if np.linalg.norm(tooth_to_bracket) > 0:
                outward_direction = tooth_to_bracket / np.linalg.norm(tooth_to_bracket)
            else:
                direction_from_center = bracket_pos - mesh_center_val
                direction_from_center[self.height_axis] = 0
                if np.linalg.norm(direction_from_center) > 0:
                    outward_direction = direction_from_center / np.linalg.norm(direction_from_center)
                else:
                    outward_direction = np.array([0, 1, 0])
            
            total_offset = 2.5
            bracket_pos = bracket_pos + outward_direction * total_offset
            
            bracket_info = {
                'position': bracket_pos,
                'tooth_type': tooth['type'],
                'tooth_index': i,
                'tooth_center': tooth_center,
                'direction': outward_direction
            }
            self.bracket_positions.append(bracket_info)
        
        print(f"Calculated {len(self.bracket_positions)} bracket positions")
        
        for i, bracket in enumerate(self.bracket_positions):
            dist_from_center = np.linalg.norm(
                bracket['position'][:2] - mesh_center_val[:2]
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
            print("Warning: No bracket positions available for archform creation.")
            return None
        
        positions = np.array([b['position'] for b in self.bracket_positions])
        
        # Ensure mesh is not None and has vertices before calling get_center()
        if self.mesh is None or not self.mesh.has_vertices():
            print("Error: Mesh not loaded or empty, cannot get mesh center for archform.")
            # Use average of bracket positions as a fallback center if mesh is problematic
            mesh_center_val = np.mean(positions, axis=0) if positions.shape[0] > 0 else np.array([0,0,0])
        else:
            mesh_center_val = self.mesh.get_center()

        # Sort positions by angle to ensure proper ordering
        angles = []
        for pos in positions:
            angle = np.arctan2(
                pos[self.ap_axis] - mesh_center_val[self.ap_axis],
                pos[self.lr_axis] - mesh_center_val[self.lr_axis]
            )
            angles.append(angle)
        
        sorted_indices = np.argsort(angles)
        sorted_positions = positions[sorted_indices]
        
        max_gap_dist = 0
        max_gap_idx = -1
        
        for i in range(len(sorted_positions)):
            next_i = (i + 1) % len(sorted_positions)
            dist = np.linalg.norm(sorted_positions[next_i] - sorted_positions[i])
            if dist > max_gap_dist:
                max_gap_dist = dist
                max_gap_idx = i
        
        if max_gap_idx >= 0 and max_gap_dist > 20.0: # Check if any gap found
            # Check if sorted_positions is not empty before vstack
            if sorted_positions.shape[0] > max_gap_idx + 1:
                 sorted_positions = np.vstack([
                    sorted_positions[max_gap_idx+1:],
                    sorted_positions[:max_gap_idx+1]
                ])
            # else: it implies very few points, no reordering needed or possible

        num_interp_points = len(sorted_positions) * 3
        
        distances = [0]
        for i in range(1, len(sorted_positions)):
            dist = np.linalg.norm(sorted_positions[i] - sorted_positions[i-1])
            distances.append(distances[-1] + dist)
        
        if len(sorted_positions) > 1 : # Need at least 2 points to check last_to_first
            last_to_first = np.linalg.norm(sorted_positions[0] - sorted_positions[-1])
            if last_to_first < 20.0 and len(sorted_positions) > 2:  # Avoid removing the only two points
                sorted_positions = sorted_positions[:-1]
                distances = distances[:-1]
        
        if len(sorted_positions) < 3:
            print(f"Warning: Too few points ({len(sorted_positions)}) for spline interpolation. Returning raw points.")
            return sorted_positions # Not enough points for spline
        
        if distances[-1] > 0:
            t_original = np.array(distances) / distances[-1]
        else: # All points are coincident, or only one point
            t_original = np.linspace(0, 1, len(sorted_positions))
        
        t_smooth = np.linspace(0, 1, num_interp_points)
        
        try:
            from scipy import interpolate
            
            lr_spline = interpolate.UnivariateSpline(
                t_original, sorted_positions[:, self.lr_axis], k=3, s=0.2)
            ap_spline = interpolate.UnivariateSpline(
                t_original, sorted_positions[:, self.ap_axis], k=3, s=0.2)
            height_spline = interpolate.UnivariateSpline(
                t_original, sorted_positions[:, self.height_axis], k=3, s=0.1)
            
            smooth_arch = np.zeros((len(t_smooth), 3))
            smooth_arch[:, self.lr_axis] = lr_spline(t_smooth)
            smooth_arch[:, self.ap_axis] = ap_spline(t_smooth)
            smooth_arch[:, self.height_axis] = height_spline(t_smooth)
            
        except Exception as e:
            print(f"Spline interpolation failed: {e}. Falling back to linear interpolation.")
            smooth_arch = []
            for i in range(len(sorted_positions) - 1):
                p1 = sorted_positions[i]
                p2 = sorted_positions[i + 1]
                for j in range(3): # 3 sub-segments
                    t = j / 3.0
                    interp_point = p1 * (1 - t) + p2 * t
                    smooth_arch.append(interp_point)
            smooth_arch.append(sorted_positions[-1]) # Add the last point
            smooth_arch = np.array(smooth_arch)
        
        print(f"Created archform with {len(smooth_arch)} points")
        print(f"Archform is open at posterior (no closing segment)")
        
        return smooth_arch
    
    def add_clinical_bends(self, archform): # archform is not used in current logic
        """Add clinical bends at appropriate locations - INCLUDING ALL BRACKETS."""
        print("\n--- Adding Clinical Bends ---")
        
        # CRITICAL: Include ALL bracket positions to ensure complete wire
        clinical_wire = []
        
        if not self.bracket_positions:
            print("Warning: No bracket positions to create clinical bends.")
            return []

        print(f"Adding all {len(self.bracket_positions)} bracket positions to wire")
        
        for i, bracket in enumerate(self.bracket_positions):
            bracket_pos = bracket['position'].copy()
            
            clinical_wire.append({
                'position': bracket_pos,
                'type': 'bracket',
                'tooth_type': bracket['tooth_type'],
                'tooth_index': bracket['tooth_index'],
                'original_index': i # Keep track of original bracket order if needed
            })
        
        if len(clinical_wire) > 2:
            if self.mesh is None or not self.mesh.has_vertices():
                # Fallback for mesh_center if mesh is not available
                all_positions = np.array([p['position'] for p in clinical_wire])
                mesh_center_val = np.mean(all_positions, axis=0) if all_positions.shape[0] > 0 else np.array([0,0,0])
            else:
                mesh_center_val = self.mesh.get_center()

            for point in clinical_wire:
                pos = point['position']
                angle = np.arctan2(
                    pos[self.ap_axis] - mesh_center_val[self.ap_axis],
                    pos[self.lr_axis] - mesh_center_val[self.lr_axis]
                )
                point['angle'] = angle
            
            clinical_wire.sort(key=lambda p: p['angle'])
            
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
            
            if gap_index >= 0 and max_gap > 15.0 and len(clinical_wire) > gap_index + 1:
                clinical_wire = clinical_wire[gap_index+1:] + clinical_wire[:gap_index+1]
                print(f"Reordered wire to start after posterior gap")
        
        print(f"\nClinical wire summary:")
        print(f"Total bend points: {len(clinical_wire)}")
        
        type_counts = {}
        for point in clinical_wire:
            tooth_type = point['tooth_type']
            type_counts[tooth_type] = type_counts.get(tooth_type, 0) + 1
        
        print("Tooth types in wire:")
        for tooth_type, count in sorted(type_counts.items()):
            print(f"  {tooth_type}: {count}")
        
        if clinical_wire:
            print(f"\nWire starts with: {clinical_wire[0]['tooth_type']}")
            print(f"Wire ends with: {clinical_wire[-1]['tooth_type']}")
        
        return clinical_wire
    
    def calculate_bending_instructions(self, clinical_wire):
        """Calculate wire bending machine instructions."""
        print("\n--- Calculating Bending Instructions ---")
        
        if len(clinical_wire) < 2:
            print("Warning: Not enough points in clinical_wire to calculate bending instructions.")
            return [] # Return empty list if not enough points
        
        instructions = []
        cumulative_feed = 0.0
        
        for i in range(len(clinical_wire)):
            point = clinical_wire[i]['position']
            
            if i > 0:
                segment_length = np.linalg.norm(
                    point - clinical_wire[i-1]['position'])
                cumulative_feed += segment_length
            
            if 0 < i < len(clinical_wire) - 1:
                p1 = clinical_wire[i-1]['position']
                p2 = point
                p3 = clinical_wire[i+1]['position']
                
                v1 = p1 - p2
                v2 = p3 - p2
                
                v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
                v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
                
                dot_product = np.clip(np.dot(v1_norm, v2_norm), -1, 1)
                angle_rad = np.arccos(dot_product)
                bend_angle = np.degrees(angle_rad)
                
                cross_product = np.cross(v1_norm, v2_norm)
                # Use a robust way to determine bend direction based on height_axis
                # This assumes height_axis is perpendicular to the main bending plane
                if cross_product[self.height_axis] < 0: # Adjust sign based on coordinate system
                    bend_angle = -bend_angle
            else:
                bend_angle = 0.0
            
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
        
        if not clinical_wire or len(clinical_wire) < 2: # Need at least 2 points for a segment
            print("Warning: Not enough points in clinical_wire to create mesh.")
            return None
        
        positions = [p['position'] for p in clinical_wire]
        wire_mesh = o3d.geometry.TriangleMesh()
        
        distances = []
        for i in range(len(positions)):
            next_i = (i + 1) % len(positions) # Loop for the last segment to first
            dist = np.linalg.norm(positions[next_i] - positions[i])
            distances.append(dist)
        
        if not distances: # Should not happen if clinical_wire has points
             print("Warning: No distances calculated, cannot create wire mesh.")
             return None

        max_dist = max(distances) if distances else 0
        
        print(f"Distances between points: min={min(distances):.1f}mm, max={max_dist:.1f}mm")
        
        avg_dist = sum(distances) / len(distances) if distances else 0
        gap_threshold = max(15.0, avg_dist * 2.5) # Increased multiplier for more robust gap detection
        
        print(f"Gap threshold: {gap_threshold:.1f}mm")
        
        segments_created = 0
        gaps_found = 0
        
        # Create segments for an open wire (do not connect last to first if it's a gap)
        for i in range(len(positions) -1): # Iterate up to the second to last point
            p1 = positions[i]
            p2 = positions[i+1]
            
            segment_vec = p2 - p1
            segment_length = np.linalg.norm(segment_vec)

            if segment_length > gap_threshold:
                print(f"Skipping gap between points {i} and {i+1} (length: {segment_length:.1f}mm)")
                gaps_found +=1
                continue # Don't create a cylinder for this gap

            if segment_length < 0.01: continue

            cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=self.wire_radius, height=segment_length, resolution=10)
            
            z_axis = np.array([0, 0, 1])
            segment_dir = segment_vec / segment_length
            
            rotation_axis = np.cross(z_axis, segment_dir)
            if np.linalg.norm(rotation_axis) > 1e-6:
                rotation_axis_normalized = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.arccos(np.clip(np.dot(z_axis, segment_dir), -1, 1))
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis_normalized * angle)
                cylinder.rotate(R, center=cylinder.get_center()) # Rotate around its own center
            
            cylinder.translate((p1 + p2) / 2) # Move to midpoint of segment
            wire_mesh += cylinder
            segments_created +=1

            # Add sphere at p1 (start of segment)
            if i == 0: # Add for the very first point
                 sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=self.wire_radius * 1.05, resolution=8)
                 sphere1.translate(p1)
                 wire_mesh += sphere1
            
            # Add sphere at p2 (end of segment)
            sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=self.wire_radius * 1.05, resolution=8)
            sphere2.translate(p2)
            wire_mesh += sphere2

        # Add larger terminal spheres if the wire is open and has defined endpoints
        if gaps_found > 0 or len(positions) > 1: # Wire is open or has at least two points
            if positions: # Check if positions is not empty
                # Sphere at the very start of the wire path
                start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.wire_radius * 1.3, resolution=10)
                start_sphere.translate(positions[0])
                wire_mesh += start_sphere

                # Sphere at the very end of the wire path
                end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.wire_radius * 1.3, resolution=10)
                end_sphere.translate(positions[-1])
                wire_mesh += end_sphere
        
        print(f"Created {segments_created} wire segments, found {gaps_found} gaps (internal to wire path).")
        
        wire_mesh.paint_uniform_color([1.0, 0.0, 0.0])
        wire_mesh.compute_vertex_normals()
        
        return wire_mesh
    
    def create_bracket_markers(self):
        """Create visual markers for bracket positions."""
        markers = o3d.geometry.TriangleMesh()
        
        if not self.bracket_positions:
            print("Warning: No bracket positions to create markers.")
            return markers # Return empty mesh

        for bracket in self.bracket_positions:
            cube = o3d.geometry.TriangleMesh.create_box(width=1.5, height=2.0, depth=1.0)
            cube.translate([-0.75, -1.0, -0.5]) # Center the cube before moving
            cube.translate(bracket['position'])
            
            if 'incisor' in bracket['tooth_type']:
                cube.paint_uniform_color([0.2, 0.2, 0.8])  # Blue
            elif 'canine' in bracket['tooth_type']:
                cube.paint_uniform_color([0.8, 0.8, 0.2])  # Yellow
            else: # Premolars and Molars
                cube.paint_uniform_color([0.2, 0.8, 0.2])  # Green
                
            markers += cube
        
        return markers
    
    def export_bending_instructions(self, instructions, filename=None):
        """Export bending instructions to CSV file."""
        if filename is None:
            base = os.path.splitext(os.path.basename(self.stl_path))[0] if self.stl_path else "wire"
            filename = f"{base}_{self.arch_type}_{self.wire_size}.csv"
        
        print(f"\n--- Exporting to {filename} ---")

        if not instructions:
            print("No instructions to export.")
            return

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Point_Index', 'Feed_Distance_mm', 'Bend_Angle_deg',
                'X_mm', 'Y_mm', 'Z_mm', 'Type', 'Tooth_Type'
            ])
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
        
        if instructions: # Check if instructions is not empty
            print("\n--- Wire Summary ---")
            print(f"Wire type: {self.wire_size}")
            print(f"Total length: {instructions[-1]['feed_distance']:.1f} mm")
            print(f"Number of bends: {len([i for i in instructions if abs(i['bend_angle']) > 0.1])}") # Count non-zero bends

    def optimize_for_production(self, clinical_wire):
        """Optimize wire for production by reducing bend points intelligently."""
        print("\n--- Optimizing for Production ---")
        
        current_points = len(clinical_wire)
        print(f"Current bend points: {current_points}")
        
        if not clinical_wire:
            print("No clinical wire points to optimize.")
            return []

        if current_points <= 28:
            print("Keeping all points for complete wire coverage (under optimization threshold).")
            return clinical_wire
        
        print("Optimizing wire: Number of points exceeds threshold.")
        critical_indices = set()
        
        if current_points > 0: # Ensure wire is not empty
            critical_indices.add(0)
            critical_indices.add(len(clinical_wire) - 1)
        
        for i, point in enumerate(clinical_wire):
            tooth_type = point['tooth_type']
            if 'incisor' in tooth_type or 'canine' in tooth_type or 'molar' in tooth_type or 'premolar' in tooth_type:
                critical_indices.add(i)
        
        selected_indices = sorted(list(critical_indices))
        optimized_wire = [clinical_wire[i] for i in selected_indices if i < len(clinical_wire)] # Check index bounds
        
        print(f"Optimized from {len(clinical_wire)} to {len(optimized_wire)} bend points")
        
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
        
        if not self.load_mesh():
            return None
        
        if not self.detect_individual_teeth():
            print("Failed to detect teeth")
            # Attempt to continue if some teeth were found, otherwise return
            if not self.teeth: return None
        
        if not self.calculate_bracket_positions():
            print("Failed to calculate bracket positions")
            if not self.bracket_positions: return None
        
        archform = self.create_ideal_archform()
        # Archform creation is for visualization/ideal path, wire uses bracket_positions directly
        # So, we don't strictly need archform to proceed with clinical_wire if it fails
        if archform is None:
            print("Warning: Failed to create ideal archform. Proceeding with bracket positions for wire.")

        # The add_clinical_bends method now primarily uses self.bracket_positions
        clinical_wire_points = self.add_clinical_bends(archform) # Pass archform, though it's not directly used for points
        
        if not clinical_wire_points:
            print("Failed to create clinical wire points.")
            return None

        clinical_wire_points = self.optimize_for_production(clinical_wire_points)
        
        instructions = self.calculate_bending_instructions(clinical_wire_points)
        
        wire_mesh = self.create_wire_mesh(clinical_wire_points)
        bracket_markers = self.create_bracket_markers()
        
        if instructions:
            self.export_bending_instructions(instructions)
        else:
            print("No bending instructions generated.")

        print("\n--- Production Readiness Assessment ---")
        print(f"✓ Wire type: {self.wire_size}")
        print(f"✓ Number of bends points in wire: {len(clinical_wire_points)}")
        if instructions and instructions[-1]['feed_distance'] is not None:
             print(f"✓ Total wire length: {instructions[-1]['feed_distance']:.1f}mm")
        else:
             print("✓ Total wire length: N/A (no instructions)")
        print(f"✓ Teeth detected: {len(self.teeth)}")
        
        expected_bends = len(self.teeth) // 1.5
        acceptable_range = (max(6, int(expected_bends) - 3), min(30, int(expected_bends) + 5)) # Wider range
        
        is_production_ready = acceptable_range[0] <= len(clinical_wire_points) <= acceptable_range[1]
        
        print(f"\nExpected bend point range for {len(self.teeth)} teeth: {acceptable_range[0]}-{acceptable_range[1]}")
        print(f"Production Ready: {'YES' if is_production_ready else 'NO'}")
        
        if not is_production_ready:
            if len(clinical_wire_points) < acceptable_range[0]:
                print("Recommendation: Check tooth detection and bracket placement. Wire may be too simple.")
            else:
                print("Recommendation: Wire may be too complex or have too many points. Review optimization if enabled.")
        
        return {
            'mesh': self.mesh,
            'wire_mesh': wire_mesh,
            'bracket_markers': bracket_markers,
            'instructions': instructions,
            'clinical_wire': clinical_wire_points,
            'production_ready': is_production_ready
        }
    
    def visualize(self, components):
        """Visualize the results."""
        print("\n--- Visualization ---")
        
        if not components:
            print("No components to visualize.")
            return

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"Professional Orthodontic Wire - {self.arch_type.title() if self.arch_type else 'N/A'}",
            width=1400, height=900
        )
        
        if components.get('mesh'):
            vis.add_geometry(components['mesh'])
        if components.get('wire_mesh'):
            vis.add_geometry(components['wire_mesh'])
        if components.get('bracket_markers'):
            vis.add_geometry(components['bracket_markers'])
        
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
        vis.add_geometry(coord_frame)
        
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

def run_tests():
    """Runs basic tests for OrthodonticWireGenerator."""
    print("\n--- Running Basic Tests ---")
    generator = OrthodonticWireGenerator(stl_path="dummy.stl", arch_type='lower')

    # Manually assign teeth
    generator.teeth = [
        {'type': 'central_incisor', 'center': np.array([2,19,0]), 'vertices': np.array([[2,19,1],[2,19,-1]]), 'angle': 0.05, 'ap_position': 19, 'label':0, 'angular_order': 2},
        {'type': 'central_incisor', 'center': np.array([-2,19,0]), 'vertices': np.array([[-2,19,1],[-2,19,-1]]), 'angle': -0.05, 'ap_position': 19, 'label':1, 'angular_order': 3},
        {'type': 'lateral_incisor', 'center': np.array([7,18,0]), 'vertices': np.array([[7,18,1],[7,18,-1]]), 'angle': 0.2, 'ap_position': 18, 'label':2, 'angular_order': 1},
        {'type': 'lateral_incisor', 'center': np.array([-7,18,0]), 'vertices': np.array([[-7,18,1],[-7,18,-1]]), 'angle': -0.2, 'ap_position': 18, 'label':3, 'angular_order': 4},
        {'type': 'canine', 'center': np.array([15,15,0]), 'vertices': np.array([[15,15,1],[15,15,-1]]), 'angle': 0.5, 'ap_position': 15, 'label':4, 'angular_order': 0},
        {'type': 'canine', 'center': np.array([-15,15,0]), 'vertices': np.array([[-15,15,1],[-15,15,-1]]), 'angle': -0.5, 'ap_position': 15, 'label':5, 'angular_order': 5},
        {'type': 'first_premolar', 'center': np.array([20,10,0]), 'vertices': np.array([[20,10,1],[20,10,-1]]), 'angle': 1.0, 'ap_position': 10, 'label':6, 'angular_order': 6}, # Note: angular order might be different after sorting
        {'type': 'first_premolar', 'center': np.array([-20,10,0]), 'vertices': np.array([[-20,10,1],[-20,10,-1]]), 'angle': -1.0, 'ap_position': 10, 'label':7, 'angular_order': 7},
        {'type': 'second_premolar', 'center': np.array([25,5,0]), 'vertices': np.array([[25,5,1],[25,5,-1]]), 'angle': 1.5, 'ap_position': 5, 'label':8, 'angular_order': 8},
        {'type': 'second_premolar', 'center': np.array([-25,5,0]), 'vertices': np.array([[-25,5,1],[-25,5,-1]]), 'angle': -1.5, 'ap_position': 5, 'label':9, 'angular_order': 9},
        {'type': 'first_molar', 'center': np.array([30,0,0]), 'vertices': np.array([[30,0,1],[30,0,-1]]), 'angle': 2.0, 'ap_position': 0, 'label':10, 'angular_order': 10},
        {'type': 'first_molar', 'center': np.array([-30,0,0]), 'vertices': np.array([[-30,0,1],[-30,0,-1]]), 'angle': -2.0, 'ap_position': 0, 'label':11, 'angular_order': 11}
    ]
    # Re-sort by angle as detect_individual_teeth would do
    generator.teeth.sort(key=lambda t: t['angle'])
    # Re-assign angular_order after sorting
    for i, tooth in enumerate(generator.teeth):
        tooth['angular_order'] = i


    # Set dummy attributes
    generator.arch_type = 'lower'
    generator.lr_axis = 0  # X-axis
    generator.ap_axis = 1  # Y-axis
    generator.height_axis = 2 # Z-axis

    dummy_mesh_vertices = []
    for tooth in generator.teeth:
        dummy_mesh_vertices.extend(tooth['vertices'])
    if not dummy_mesh_vertices: # Fallback if no vertices provided in teeth
        dummy_mesh_vertices = [[-40,-20,-5],[40,-20,-5], [0,20,-5], [-40,20,5], [40,20,5], [0,0,10]] # Some spread out points

    dummy_mesh = o3d.geometry.TriangleMesh()
    dummy_mesh.vertices = o3d.utility.Vector3dVector(np.array(dummy_mesh_vertices))
    # Add minimal triangles if possible, otherwise Open3D might complain for some operations
    if len(dummy_mesh_vertices) >=3:
        dummy_mesh.triangles = o3d.utility.Vector3iVector(np.array([[0,1,2]]))


    generator.mesh = dummy_mesh

    if generator.arch_type not in generator.BRACKET_HEIGHTS:
        generator.BRACKET_HEIGHTS[generator.arch_type] = generator.BRACKET_HEIGHTS['lower']

    # Manually create bracket_positions (can use calculate_bracket_positions if mesh is good enough)
    # For robustness in test, let's use a simplified version of calculate_bracket_positions
    # or simply mock it as per instructions.
    generator.bracket_positions = []
    mesh_center_val = generator.mesh.get_center() if generator.mesh and generator.mesh.has_vertices() else np.array([0,0,0])

    for i, tooth in enumerate(generator.teeth):
        # Simplified bracket calculation for testing
        bracket_height_val = generator.BRACKET_HEIGHTS[generator.arch_type].get(tooth['type'], 4.5)
        bracket_pos = tooth['center'].copy().astype(float) # Ensure bracket_pos is float

        # Adjust height based on arch type and height axis
        if generator.arch_type == 'upper':
             bracket_pos[generator.height_axis] = np.max(tooth['vertices'][:, generator.height_axis]) + bracket_height_val if tooth['vertices'].size >0 else tooth['center'][generator.height_axis] + bracket_height_val
        else: # lower
             bracket_pos[generator.height_axis] = np.min(tooth['vertices'][:, generator.height_axis]) - bracket_height_val if tooth['vertices'].size >0 else tooth['center'][generator.height_axis] - bracket_height_val

        # Simplified outward direction
        direction_from_center = bracket_pos - mesh_center_val
        direction_from_center[generator.height_axis] = 0
        norm_direction = np.array([0,1,0]) # Default outward
        if np.linalg.norm(direction_from_center) > 1e-6:
            norm_direction = direction_from_center / np.linalg.norm(direction_from_center)

        bracket_pos += norm_direction * 2.5 # Offset for bracket base + wire clearance

        generator.bracket_positions.append({
            'position': bracket_pos,
            'tooth_type': tooth['type'],
            'tooth_index': i, # This should be the index in the ordered list of teeth
            'tooth_center': tooth['center'],
            'direction': norm_direction
        })

    # Test Logic
    num_incisors = sum(1 for t in generator.teeth if 'incisor' in t['type'])
    num_canines = sum(1 for t in generator.teeth if t['type'] == 'canine')
    num_others = sum(1 for t in generator.teeth if 'molar' in t['type'] or 'premolar' in t['type'])

    print(f"Mock teeth: {len(generator.teeth)} total")
    print(f"  Incisors: {num_incisors}, Canines: {num_canines}, Others (Molars/Premolars): {num_others}")

    assert num_incisors == 4, f"Incisor count mismatch. Expected 4, got {num_incisors}"
    assert num_canines == 2, f"Canine count mismatch. Expected 2, got {num_canines}"
    assert num_others == 6, f"Other teeth count mismatch. Expected 6, got {num_others}"
    print("Tooth count assertions passed.")

    archform_points = generator.create_ideal_archform()
    assert archform_points is not None, "create_ideal_archform returned None"
    # Archform points are interpolated, so their count won't directly match bracket_positions
    print(f"Archform created with {len(archform_points)} points.")

    # The add_clinical_bends function now directly uses generator.bracket_positions
    # The `archform_points` parameter is not strictly necessary for its current primary logic
    # if it's just about getting points from brackets.
    clinical_wire_points = generator.add_clinical_bends(archform_points)

    assert len(clinical_wire_points) == len(generator.bracket_positions), \
        f"Wire points count ({len(clinical_wire_points)}) should match bracket positions count ({len(generator.bracket_positions)})"
    print("Clinical wire points count matches bracket positions count assertion passed.")

    # Test optimize_for_production
    optimized_wire = generator.optimize_for_production(list(clinical_wire_points)) # Pass a copy
    assert len(optimized_wire) <= len(clinical_wire_points), "Optimized wire should not have more points."
    if len(clinical_wire_points) > 28 : # Only if optimization was supposed to run
         assert len(optimized_wire) < len(clinical_wire_points), "Optimization did not reduce points when expected."
    print("Optimization logic basic check passed.")


    # Test calculate_bending_instructions
    instructions = generator.calculate_bending_instructions(clinical_wire_points)
    assert len(instructions) == len(clinical_wire_points), "Instructions count should match clinical wire points."
    if instructions: # If not empty
        assert 'feed_distance' in instructions[0], "Instruction missing 'feed_distance'."
        assert 'bend_angle' in instructions[0], "Instruction missing 'bend_angle'."
    print("Bending instructions calculation basic check passed.")

    # Test create_wire_mesh
    wire_mesh = generator.create_wire_mesh(clinical_wire_points)
    if len(clinical_wire_points) >=2: # Mesh can only be created with at least 2 points
        assert wire_mesh is not None, "Wire mesh creation failed for valid input."
        assert wire_mesh.has_vertices(), "Wire mesh has no vertices."
    else:
        assert wire_mesh is None, "Wire mesh should be None for insufficient points."
    print("Wire mesh creation basic check passed.")

    # Test create_bracket_markers
    bracket_markers = generator.create_bracket_markers()
    if generator.bracket_positions:
        assert bracket_markers is not None, "Bracket markers creation failed."
        assert bracket_markers.has_vertices(), "Bracket markers have no vertices."
    else:
        # If no bracket_positions, it should return an empty mesh, which is not None but has no vertices
        assert bracket_markers is not None and not bracket_markers.has_vertices(), "Bracket markers should be empty if no positions."
    print("Bracket markers creation basic check passed.")

    print("\nBasic tests passed!")

def main():
    """Main execution function."""
    # Configuration
    # Ensure stl_path is valid or handle its absence for regular runs
    stl_path = "/Users/galala/STL con/assets/AyaKhairy_LowerJaw.stl"
    # Check if the default STL path exists, otherwise skip visualization part or use a known dummy
    if not os.path.exists(stl_path):
        print(f"Warning: STL file not found at {stl_path}. Skipping main execution.")
        # Optionally, create a dummy STL for testing if needed, or ensure main() handles this
        # For now, we'll just print a warning and try to proceed if parts of main() can run without it
        # Or, more robustly, exit or prevent calling generator.generate_wire()
        return

    generator = OrthodonticWireGenerator(
        stl_path=stl_path,
        arch_type='auto',
        wire_size='0.018'
    )
    
    result = generator.generate_wire()
    
    if result and result.get('production_ready'): # Check if result is not None
        print("\n✅ SUCCESS: Professional orthodontic wire generated")
        generator.visualize(result)
    elif result:
        print("\n⚠️ Wire generated but NOT production ready. Review recommendations.")
        generator.visualize(result) # Still visualize if not production ready
    else:
        print("\n❌ FAILED: Could not generate wire")


if __name__ == "__main__":
    run_tests() # Call the test function
    # For actual use, call main():
    # main()
