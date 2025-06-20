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
        axes = [0, 1, 2]
        # Find the axis with the largest extent (likely Left-Right)
        self.lr_axis = np.argmax(extent)

        # Find the axis with the smallest extent (likely Height)
        # To ensure it's different from lr_axis if extents are very similar or degenerate:
        temp_extent = extent.copy()
        temp_extent[self.lr_axis] = np.inf # Temporarily remove lr_axis from consideration for height
        self.height_axis = np.argmin(temp_extent)

        if self.lr_axis == self.height_axis:
            possible_height_axes = [ax for ax in axes if ax != self.lr_axis]
            if possible_height_axes:
                self.height_axis = min(possible_height_axes, key=lambda ax: extent[ax])
            else:
                self.lr_axis = 0; self.height_axis = 1

        self.ap_axis = [ax for ax in axes if ax not in {self.lr_axis, self.height_axis}][0]
        
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
        if len(crown_vertices) == 0:
            print("Warning: No vertices found at crown level. Tooth detection might fail or be inaccurate.")
            # Fallback: use all vertices if no crown vertices found, this might be slow/noisy
            # crown_vertices = vertices
            # For now, let it proceed; if it fails, it fails.

        num_segments = 16
        centered_vertices = crown_vertices - center # Use crown_vertices
        if len(centered_vertices) == 0: # Handle empty centered_vertices
            print("Error: No crown vertices to process for tooth segmentation.")
            return False

        angles = np.arctan2(centered_vertices[:, self.ap_axis], centered_vertices[:, self.lr_axis])
        radii = np.sqrt(centered_vertices[:, self.lr_axis]**2 + centered_vertices[:, self.ap_axis]**2)
        
        self.teeth = []
        angle_step = 2 * np.pi / (num_segments + 2)
        start_angle = -np.pi + angle_step
        
        for i in range(num_segments):
            segment_start = start_angle + i * angle_step
            segment_end = segment_start + angle_step
            angle_mask = np.logical_and(angles >= segment_start, angles < segment_end)
            segment_vertices = crown_vertices[angle_mask] # Use crown_vertices
            
            if len(segment_vertices) > 30:
                segment_radii_points = radii[angle_mask] # Use radii from crown_vertices
                if len(segment_radii_points) > 0:
                    radius_threshold = np.percentile(segment_radii_points, 70)
                    outer_mask = segment_radii_points >= radius_threshold
                    # Apply outer_mask to segment_vertices, not crown_vertices directly
                    outer_vertices = segment_vertices[outer_mask]
                    
                    if len(outer_vertices) > 0: tooth_center = np.mean(outer_vertices, axis=0)
                    else: tooth_center = np.mean(segment_vertices, axis=0)
                else: tooth_center = np.mean(segment_vertices, axis=0)
                
                dist_from_center = np.linalg.norm(tooth_center[[self.lr_axis, self.ap_axis]] - center[[self.lr_axis, self.ap_axis]])
                if dist_from_center < 10.0: continue
                
                self.teeth.append({
                    'center': tooth_center, 'vertices': segment_vertices, 'label': len(self.teeth),
                    'type': 'unknown',
                    'angle': np.arctan2(tooth_center[self.ap_axis]-center[self.ap_axis], tooth_center[self.lr_axis]-center[self.lr_axis]),
                    'ap_position': tooth_center[self.ap_axis]
                })
        
        self.teeth.sort(key=lambda t: t['angle'])
        print("\n--- Teeth after sorting by angle: ---")
        arch_center_lr = center[self.lr_axis]
        for i, tooth in enumerate(self.teeth):
            side = "Right" if tooth['center'][self.lr_axis] < arch_center_lr else "Left"
            print(f"  Tooth {i}: Angle: {np.degrees(tooth['angle']):.1f}°, AP: {tooth['ap_position']:.1f}, LR_pos: {tooth['center'][self.lr_axis]:.1f} (Likely {side})")

        num_teeth = len(self.teeth)
        if num_teeth > 0:
            for i_tooth, t in enumerate(self.teeth): t['angular_order'] = i_tooth
            sorted_ap_teeth = sorted(self.teeth, key=lambda t: t['ap_position'], reverse=True)
            incisor_candidates = [sorted_ap_teeth[i] for i in range(min(num_teeth, 6))]
            min_angular_idx = int(num_teeth * 0.20); max_angular_idx = int(num_teeth * 0.80)
            potential_incisors = [t for t in incisor_candidates if min_angular_idx <= t['angular_order'] <= max_angular_idx]
            if len(potential_incisors) >= 4: num_incisors_to_classify = min(len(potential_incisors), 6)
            elif len(incisor_candidates) >= 4: potential_incisors = incisor_candidates; num_incisors_to_classify = min(len(potential_incisors), 6)
            else: potential_incisors = incisor_candidates; num_incisors_to_classify = len(potential_incisors)
            final_incisors = sorted(potential_incisors[:num_incisors_to_classify], key=lambda t: t['angular_order'])
            if len(final_incisors) > 0:
                median_angular_order = np.median([t['angular_order'] for t in final_incisors])
                for tooth in final_incisors: tooth['type'] = 'lateral_incisor'
                final_incisors.sort(key=lambda t: abs(t['angular_order'] - median_angular_order))
                if len(final_incisors) >= 1: final_incisors[0]['type'] = 'central_incisor'
                if len(final_incisors) >= 2: final_incisors[1]['type'] = 'central_incisor'
            incisor_indices_in_arch = sorted([t['angular_order'] for t in final_incisors])
            canine_candidates = []
            if len(incisor_indices_in_arch) > 0:
                min_idx_val = incisor_indices_in_arch[0];
                if min_idx_val > 0: canine_candidates.append(self.teeth[min_idx_val - 1])
                max_idx_val = incisor_indices_in_arch[-1];
                if max_idx_val < num_teeth - 1: canine_candidates.append(self.teeth[max_idx_val + 1])
            for tooth in canine_candidates:
                if tooth['type'] == 'unknown': tooth['type'] = 'canine'
            for i_tooth, tooth in enumerate(self.teeth):
                if tooth['type'] == 'unknown':
                    pos_ratio = i_tooth / (num_teeth - 1) if num_teeth > 1 else 0.5
                    if pos_ratio < 0.15 or pos_ratio > 0.85: tooth['type'] = 'second_molar'
                    elif pos_ratio < 0.25 or pos_ratio > 0.75: tooth['type'] = 'first_molar'
                    elif pos_ratio < 0.35 or pos_ratio > 0.65: tooth['type'] = 'second_premolar'
                    else: tooth['type'] = 'first_premolar'
            for i_tooth, tooth in enumerate(self.teeth):
                if tooth['type'] == 'unknown':
                    pos_ratio = i_tooth / (num_teeth - 1) if num_teeth > 1 else 0.5
                    if pos_ratio < 0.20 or pos_ratio > 0.80: tooth['type'] = 'second_molar'
                    elif pos_ratio < 0.35 or pos_ratio > 0.65:
                        left_type = self.teeth[i_tooth-1]['type'] if i_tooth > 0 else "gap"
                        right_type = self.teeth[i_tooth+1]['type'] if i_tooth < num_teeth-1 else "gap"
                        if "molar" in left_type or "molar" in right_type: tooth['type'] = 'first_molar'
                        else: tooth['type'] = 'second_premolar'
                    else: tooth['type'] = 'first_premolar'
        print(f"\nDetected {len(self.teeth)} teeth after classification attempt.")
        print("\n--- Teeth after final classification: ---")
        for i, tooth in enumerate(self.teeth):
            side = "Right" if tooth['center'][self.lr_axis] < center[self.lr_axis] else "Left"
            print(f"  Tooth {i}: Type: {tooth['type']}, Angle: {np.degrees(tooth['angle']):.1f}°, AP: {tooth['ap_position']:.1f}, LR_pos: {tooth['center'][self.lr_axis]:.1f} (Likely {side})")
        type_counts = {};
        for tooth in self.teeth: type_counts[tooth['type']] = type_counts.get(tooth['type'], 0) + 1
        print(f"\nFinal tooth type distribution: {type_counts}")
        return len(self.teeth) >= 6
    
    def calculate_bracket_positions(self):
        """Calculate clinical bracket positions for each tooth."""
        print("\n--- Calculating Bracket Positions ---")
        print(f"Number of teeth available for bracket calculation: {len(self.teeth)}")
        self.bracket_positions = []
        mesh_center_val = self.mesh.get_center() if self.mesh and self.mesh.has_vertices() else np.array([0,0,0])
        for i, tooth in enumerate(self.teeth):
            print(f"  Calculating bracket for Tooth {i} (Type: {tooth['type']}, Index in sorted list: {tooth.get('angular_order', 'N/A')}, Center: {tooth['center']})")
            bracket_height = self.BRACKET_HEIGHTS[self.arch_type].get(tooth['type'], 4.5)
            tooth_vertices = tooth.get('vertices', np.array([tooth['center']]))
            if not isinstance(tooth_vertices, np.ndarray) or tooth_vertices.ndim != 2 or tooth_vertices.shape[1] != 3:
                tooth_vertices = np.array([tooth['center']])
            tooth_center = tooth['center']
            if self.arch_type == 'upper':
                target_height = (np.min(tooth_vertices[:, self.height_axis]) if tooth_vertices.size > 0 else tooth_center[self.height_axis]) + bracket_height
            else:
                target_height = (np.max(tooth_vertices[:, self.height_axis]) if tooth_vertices.size > 0 else tooth_center[self.height_axis]) - bracket_height
            height_tolerance = 2.0
            bracket_level_vertices = tooth_vertices[np.abs(tooth_vertices[:, self.height_axis] - target_height) < height_tolerance] if tooth_vertices.size > 0 else np.array([])
            if len(bracket_level_vertices) < 10:
                bracket_pos = tooth_center.copy(); bracket_pos[self.height_axis] = target_height
            else:
                vectors_from_center = bracket_level_vertices - mesh_center_val; vectors_from_center[:, self.height_axis] = 0
                horizontal_distances = np.linalg.norm(vectors_from_center, axis=1)
                outer_vertices = bracket_level_vertices[horizontal_distances >= np.percentile(horizontal_distances, 90)]
                if len(outer_vertices) > 3: bracket_pos = np.mean(outer_vertices, axis=0)
                else: bracket_pos = bracket_level_vertices[np.argmax(horizontal_distances)]
            tooth_to_bracket = bracket_pos - tooth_center; tooth_to_bracket[self.height_axis] = 0
            if np.linalg.norm(tooth_to_bracket) > 0: outward_direction = tooth_to_bracket / np.linalg.norm(tooth_to_bracket)
            else:
                direction_from_center = bracket_pos - mesh_center_val; direction_from_center[self.height_axis] = 0
                outward_direction = direction_from_center / np.linalg.norm(direction_from_center) if np.linalg.norm(direction_from_center) > 0 else np.array([0,1,0])
            bracket_pos += outward_direction * 2.5
            self.bracket_positions.append({'position': bracket_pos, 'tooth_type': tooth['type'],
                                           'tooth_index': tooth.get('angular_order', i), 'tooth_center': tooth_center,
                                           'direction': outward_direction})
        print(f"Total bracket positions calculated: {len(self.bracket_positions)}")
        return len(self.bracket_positions) > 0
    
    def create_ideal_archform(self):
        print("\n--- Creating Ideal Archform ---")
        if not self.bracket_positions: print("Warning: No bracket positions for archform."); return None
        positions = np.array([b['position'] for b in self.bracket_positions])
        mesh_center_val = self.mesh.get_center() if self.mesh and self.mesh.has_vertices() else np.mean(positions, axis=0) if positions.size > 0 else np.array([0,0,0])
        angles = [np.arctan2(p[self.ap_axis]-mesh_center_val[self.ap_axis], p[self.lr_axis]-mesh_center_val[self.lr_axis]) for p in positions]
        sorted_positions = positions[np.argsort(angles)]
        max_gap_dist = 0; max_gap_idx = -1
        for i in range(len(sorted_positions)):
            dist = np.linalg.norm(sorted_positions[(i + 1) % len(sorted_positions)] - sorted_positions[i])
            if dist > max_gap_dist: max_gap_dist = dist; max_gap_idx = i
        if max_gap_idx >= 0 and max_gap_dist > 20.0 and sorted_positions.shape[0] > max_gap_idx + 1:
            sorted_positions = np.vstack([sorted_positions[max_gap_idx+1:], sorted_positions[:max_gap_idx+1]])
        num_interp_points = len(sorted_positions) * 3; distances = [0]
        for i in range(1, len(sorted_positions)): distances.append(distances[-1] + np.linalg.norm(sorted_positions[i] - sorted_positions[i-1]))
        if len(sorted_positions) > 1 and np.linalg.norm(sorted_positions[0]-sorted_positions[-1]) < 20.0 and len(sorted_positions) > 2:
            sorted_positions = sorted_positions[:-1]; distances = distances[:-1]
        if len(sorted_positions) < 3: print(f"Warning: Too few points ({len(sorted_positions)}) for spline. Raw points returned."); return sorted_positions
        t_original = np.array(distances)/distances[-1] if distances[-1]>0 else np.linspace(0,1,len(sorted_positions))
        t_smooth = np.linspace(0,1,num_interp_points)
        try:
            smooth_arch = np.zeros((len(t_smooth),3))
            for axis_idx, s_val in [(self.lr_axis, 0.2), (self.ap_axis, 0.2), (self.height_axis, 0.1)]:
                spline = interpolate.UnivariateSpline(t_original, sorted_positions[:,axis_idx], k=3, s=s_val)
                smooth_arch[:,axis_idx] = spline(t_smooth)
        except Exception as e:
            print(f"Spline failed: {e}. Linear fallback."); smooth_arch = [];
            for i in range(len(sorted_positions)-1):
                p1,p2=sorted_positions[i],sorted_positions[i+1]
                for j in range(3): smooth_arch.append(p1*(1-j/3.0)+p2*(j/3.0))
            smooth_arch.append(sorted_positions[-1]); smooth_arch=np.array(smooth_arch)
        print(f"Created archform with {len(smooth_arch)} points. Open at posterior."); return smooth_arch
    
    def add_clinical_bends(self, archform):
        print("\n--- Adding Clinical Bends ---")
        if not self.bracket_positions: print("Warning: No bracket positions."); return []
        clinical_wire = [{'position': b['position'].copy(), 'type': 'bracket', 'tooth_type': b['tooth_type'],
                          'tooth_index': b['tooth_index'], 'original_bracket_list_index': i}
                         for i,b in enumerate(self.bracket_positions)]
        print(f"Adding all {len(clinical_wire)} bracket positions to wire")
        if len(clinical_wire) > 2:
            mesh_center_val = self.mesh.get_center() if self.mesh and self.mesh.has_vertices() else np.mean([p['position'] for p in clinical_wire],axis=0)
            for p in clinical_wire: p['angle_for_sorting'] = np.arctan2(p['position'][self.ap_axis]-mesh_center_val[self.ap_axis], p['position'][self.lr_axis]-mesh_center_val[self.lr_axis])
            clinical_wire.sort(key=lambda p: p['angle_for_sorting'])
            print("\n--- Clinical wire after initial sorting by angle: ---")
            for idx, p in enumerate(clinical_wire): print(f"  Idx {idx}: Type: {p['tooth_type']}, Pos: {p['position']}, OrigBracketIdx: {p['original_bracket_list_index']}, ToothIdx (angular_order): {p['tooth_index']}")
            max_gap=0; gap_idx=-1
            for i in range(len(clinical_wire)):
                dist = np.linalg.norm(clinical_wire[(i+1)%len(clinical_wire)]['position'] - clinical_wire[i]['position'])
                if dist > max_gap: max_gap=dist; gap_idx=i
            print(f"Found posterior gap of {max_gap:.1f}mm at index {gap_idx} (before reordering).")
            if gap_idx >=0 and max_gap > 15.0 and len(clinical_wire) > gap_idx + 1 : # Ensure reordering is safe
                clinical_wire = clinical_wire[gap_idx+1:] + clinical_wire[:gap_idx+1]
                print(f"Reordered wire. New length: {len(clinical_wire)}")
        print(f"\nClinical wire summary (after potential reordering): Total points: {len(clinical_wire)}")
        type_counts = {};
        for p in clinical_wire: type_counts[p['tooth_type']] = type_counts.get(p['tooth_type'],0)+1
        print("Tooth types in wire:");
        for tt,c in sorted(type_counts.items()): print(f"  {tt}: {c}")
        if clinical_wire: print(f"\nWire starts: Idx 0, Type: {clinical_wire[0]['tooth_type']}, Pos: {clinical_wire[0]['position']}\nWire ends: Idx {len(clinical_wire)-1}, Type: {clinical_wire[-1]['tooth_type']}, Pos: {clinical_wire[-1]['position']}")
        return clinical_wire
    
    def calculate_bending_instructions(self, clinical_wire):
        print("\n--- Calculating Bending Instructions ---")
        if len(clinical_wire) < 2: print("Warning: Not enough points for bending instructions."); return []
        instructions = []; cumulative_feed = 0.0
        for i in range(len(clinical_wire)):
            point = clinical_wire[i]['position']
            if i > 0: cumulative_feed += np.linalg.norm(point - clinical_wire[i-1]['position'])
            if 0 < i < len(clinical_wire) - 1:
                p1,p2,p3 = clinical_wire[i-1]['position'], point, clinical_wire[i+1]['position']
                v1,v2 = p1 - p2, p3 - p2
                v1_n, v2_n = v1/(np.linalg.norm(v1)+1e-8), v2/(np.linalg.norm(v2)+1e-8)
                angle_rad = np.arccos(np.clip(np.dot(v1_n, v2_n), -1, 1)); bend_angle = np.degrees(angle_rad)
                if np.cross(v1_n, v2_n)[self.height_axis] < 0: bend_angle = -bend_angle
            else: bend_angle = 0.0
            instructions.append({'index':i,'feed_distance':cumulative_feed,'bend_angle':bend_angle,'position':point,
                                 'type':clinical_wire[i].get('type','standard'),'tooth_type':clinical_wire[i].get('tooth_type','unknown')})
        return instructions

    def create_wire_mesh(self, clinical_wire):
        """Create 3D mesh visualization of the wire. Assumes clinical_wire is ordered and open."""
        print("\n--- Creating Wire Mesh ---")
        if not clinical_wire or len(clinical_wire) < 2:
            print("Warning: Not enough points in clinical_wire to create mesh.")
            return None
        
        positions = np.array([p['position'] for p in clinical_wire])
        print(f"Wire mesh positions array (first 5): {positions[:5]}")
        print(f"Total positions for wire mesh: {len(positions)}")

        wire_mesh = o3d.geometry.TriangleMesh()
        segments_created = 0

        # Create segments between consecutive points, as clinical_wire is already the desired path
        for i in range(len(positions) - 1):
            p1 = positions[i]
            p2 = positions[i+1]
            
            segment_vec = p2 - p1
            segment_length = np.linalg.norm(segment_vec)

            print(f"  Segment from point {i} ({p1}) to {i+1} ({p2}): Distance: {segment_length:.2f}mm. ", end="")

            if segment_length < 0.01: # Skip very short segments
                print("Skipped (Too short).")
                continue
            print("Creating segment.")

            cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=self.wire_radius, height=segment_length, resolution=10, create_uv_map=True)
            
            z_axis = np.array([0, 0, 1])
            segment_dir = segment_vec / segment_length
            
            rotation_axis = np.cross(z_axis, segment_dir)
            dot_product = np.dot(z_axis, segment_dir)
            angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

            if np.linalg.norm(rotation_axis) > 1e-6:
                rotation_axis_normalized = rotation_axis / np.linalg.norm(rotation_axis)
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis_normalized * angle)
                cylinder.rotate(R, center=np.array([0,0,0]))
            elif dot_product < 0:
                 R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1,0,0]) * np.pi)
                 cylinder.rotate(R, center=np.array([0,0,0]))

            cylinder.translate(p1 + segment_vec / 2)
            wire_mesh += cylinder
            segments_created += 1
            
            # Add joint spheres (smaller)
            if i == 0: # First point of the first segment
                sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=self.wire_radius * 1.05, resolution=8)
                sphere1.translate(p1)
                wire_mesh += sphere1

            sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=self.wire_radius * 1.05, resolution=8)
            sphere2.translate(p2)
            wire_mesh += sphere2

        # Add larger terminal spheres at the absolute start and end of the wire
        if len(positions) > 0:
            start_terminal_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.wire_radius * 1.3, resolution=10)
            start_terminal_sphere.translate(positions[0])
            wire_mesh += start_terminal_sphere
            print(f"Added large terminal sphere at wire start: {positions[0]}")

            if len(positions) > 1: # Only add end terminal if there's more than one point
                end_terminal_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.wire_radius * 1.3, resolution=10)
                end_terminal_sphere.translate(positions[-1])
                wire_mesh += end_terminal_sphere
                print(f"Added large terminal sphere at wire end: {positions[-1]}")

        print(f"Created {segments_created} wire segments.")
        if wire_mesh.has_vertices():
            wire_mesh.paint_uniform_color([1.0, 0.0, 0.0])
            wire_mesh.compute_vertex_normals()
        return wire_mesh

    def create_bracket_markers(self):
        markers = o3d.geometry.TriangleMesh()
        if not self.bracket_positions: print("Warning: No bracket positions for markers."); return markers
        for bracket in self.bracket_positions:
            cube = o3d.geometry.TriangleMesh.create_box(width=1.5, height=2.0, depth=1.0)
            cube.translate([-0.75, -1.0, -0.5]); cube.translate(bracket['position'])
            if 'incisor' in bracket['tooth_type']: cube.paint_uniform_color([0.2, 0.2, 0.8])
            elif 'canine' in bracket['tooth_type']: cube.paint_uniform_color([0.8, 0.8, 0.2])
            else: cube.paint_uniform_color([0.2, 0.8, 0.2])
            markers += cube
        return markers
    
    def export_bending_instructions(self, instructions, filename=None):
        if filename is None:
            base = os.path.splitext(os.path.basename(self.stl_path))[0] if self.stl_path and self.stl_path != "dummy.stl" else "wire"
            filename = f"{base}_{self.arch_type}_{self.wire_size}.csv"
        print(f"\n--- Exporting to {filename} ---")
        if not instructions: print("No instructions to export."); return
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Point_Index', 'Feed_Distance_mm', 'Bend_Angle_deg', 'X_mm', 'Y_mm', 'Z_mm', 'Type', 'Tooth_Type'])
            for inst in instructions:
                writer.writerow([inst['index'], f"{inst['feed_distance']:.3f}", f"{inst['bend_angle']:.2f}",
                                 f"{inst['position'][0]:.3f}", f"{inst['position'][1]:.3f}", f"{inst['position'][2]:.3f}",
                                 inst['type'], inst['tooth_type']])
        print(f"Exported {len(instructions)} instructions")
        if instructions:
            print("\n--- Wire Summary ---"); print(f"Wire type: {self.wire_size}")
            print(f"Total length: {instructions[-1]['feed_distance']:.1f} mm")
            print(f"Number of bends: {len([i for i in instructions if abs(i['bend_angle']) > 0.1])}")

    def optimize_for_production(self, clinical_wire):
        print("\n--- Optimizing for Production ---")
        current_points = len(clinical_wire)
        print(f"Current bend points: {current_points}")
        if not clinical_wire: print("No points to optimize."); return []
        if current_points <= 28: print("Keeping all points (under optimization threshold)."); return clinical_wire
        print("Optimizing wire as points > 28.")
        critical_indices = set()
        if current_points > 0: critical_indices.add(0); critical_indices.add(len(clinical_wire) - 1)
        for i, point in enumerate(clinical_wire):
            if any(keyword in point['tooth_type'] for keyword in ['incisor','canine','molar','premolar']):
                critical_indices.add(i)
        selected_indices = sorted(list(critical_indices))
        optimized_wire = [clinical_wire[i] for i in selected_indices if i < len(clinical_wire)]
        print(f"Optimized from {len(clinical_wire)} to {len(optimized_wire)} points")
        type_counts = {};
        for p in optimized_wire: type_counts[p['tooth_type']] = type_counts.get(p['tooth_type'],0)+1
        print(f"Tooth types after optimization: {type_counts}")
        return optimized_wire
    
    def generate_wire(self):
        print("\n" + "="*70 + "\nPROFESSIONAL ORTHODONTIC WIRE GENERATION\n" + "="*70)
        if not self.load_mesh(): return None
        if not self.detect_individual_teeth(): print("Failed to detect sufficient teeth.");
        if not self.teeth: return None
        if not self.calculate_bracket_positions(): print("Failed to calculate sufficient bracket positions.");
        if not self.bracket_positions: return None
        
        archform = self.create_ideal_archform()
        if archform is None: print("Warning: Failed to create ideal archform. Wire based on brackets.")

        clinical_wire_points = self.add_clinical_bends(archform)
        if not clinical_wire_points: print("Failed to create clinical wire points."); return None

        clinical_wire_points = self.optimize_for_production(clinical_wire_points)
        instructions = self.calculate_bending_instructions(clinical_wire_points)
        wire_mesh = self.create_wire_mesh(clinical_wire_points)
        bracket_markers = self.create_bracket_markers()
        
        if instructions: self.export_bending_instructions(instructions)
        else: print("No bending instructions generated.")

        print("\n--- Production Readiness Assessment ---")
        print(f"✓ Wire type: {self.wire_size}"); print(f"✓ Number of bend points in wire: {len(clinical_wire_points)}")
        if instructions and instructions[-1]['feed_distance'] is not None: print(f"✓ Total wire length: {instructions[-1]['feed_distance']:.1f}mm")
        else: print("✓ Total wire length: N/A (no instructions)")
        print(f"✓ Teeth detected: {len(self.teeth)}")
        
        expected_bends_num = len(self.teeth) / 1.5
        acceptable_range = (max(6, int(expected_bends_num) - 3), min(30, int(expected_bends_num) + 5))
        is_production_ready = acceptable_range[0] <= len(clinical_wire_points) <= acceptable_range[1]
        
        print(f"\nExpected bend point range for {len(self.teeth)} teeth: {acceptable_range[0]}-{acceptable_range[1]}")
        print(f"Production Ready: {'YES' if is_production_ready else 'NO'}")
        if not is_production_ready:
            if len(clinical_wire_points) < acceptable_range[0]: print("Recommendation: Check tooth detection/placement. Wire may be too simple.")
            else: print("Recommendation: Wire may be too complex. Review optimization.")
        return {'mesh': self.mesh, 'wire_mesh': wire_mesh, 'bracket_markers': bracket_markers,
                'instructions': instructions, 'clinical_wire': clinical_wire_points, 'production_ready': is_production_ready}
    
    def visualize(self, components):
        print("\n--- Visualization ---")
        if not components: print("No components to visualize."); return
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"Orthodontic Wire - {self.arch_type.title() if self.arch_type else 'N/A'}", width=1400, height=900)
        if components.get('mesh'): vis.add_geometry(components['mesh'])
        if components.get('wire_mesh') and components['wire_mesh'].has_vertices(): vis.add_geometry(components['wire_mesh'])
        if components.get('bracket_markers') and components['bracket_markers'].has_vertices(): vis.add_geometry(components['bracket_markers'])
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0); vis.add_geometry(coord_frame)
        opt = vis.get_render_option(); opt.background_color = np.array([0.1, 0.1, 0.1]); opt.mesh_show_back_face = True
        print("\nControls:\n- Mouse: Rotate view\n- Scroll: Zoom\n- Blue=Incisors, Yellow=Canines, Green=Premolars/Molars")
        vis.run(); vis.destroy_window()

def run_tests():
    """Runs basic tests for OrthodonticWireGenerator."""
    print("\n--- Running Basic Tests ---")
    generator = OrthodonticWireGenerator(stl_path="dummy.stl", arch_type='lower')
    generator.teeth = [
        {'type': 'central_incisor', 'center': np.array([2,19,0]), 'vertices': np.array([[2,19,1],[2,19,-1]]), 'angle': 0.05, 'ap_position': 19, 'label':0, 'angular_order': 2},
        {'type': 'central_incisor', 'center': np.array([-2,19,0]), 'vertices': np.array([[-2,19,1],[-2,19,-1]]), 'angle': -0.05, 'ap_position': 19, 'label':1, 'angular_order': 3},
        {'type': 'lateral_incisor', 'center': np.array([7,18,0]), 'vertices': np.array([[7,18,1],[7,18,-1]]), 'angle': 0.2, 'ap_position': 18, 'label':2, 'angular_order': 1},
        {'type': 'lateral_incisor', 'center': np.array([-7,18,0]), 'vertices': np.array([[-7,18,1],[-7,18,-1]]), 'angle': -0.2, 'ap_position': 18, 'label':3, 'angular_order': 4},
        {'type': 'canine', 'center': np.array([15,15,0]), 'vertices': np.array([[15,15,1],[15,15,-1]]), 'angle': 0.5, 'ap_position': 15, 'label':4, 'angular_order': 0},
        {'type': 'canine', 'center': np.array([-15,15,0]), 'vertices': np.array([[-15,15,1],[-15,15,-1]]), 'angle': -0.5, 'ap_position': 15, 'label':5, 'angular_order': 5},
        {'type': 'first_premolar', 'center': np.array([20,10,0]), 'vertices': np.array([[20,10,1],[20,10,-1]]), 'angle': 1.0, 'ap_position': 10, 'label':6, 'angular_order': 6},
        {'type': 'first_premolar', 'center': np.array([-20,10,0]), 'vertices': np.array([[-20,10,1],[-20,10,-1]]), 'angle': -1.0, 'ap_position': 10, 'label':7, 'angular_order': 7},
        {'type': 'second_premolar', 'center': np.array([25,5,0]), 'vertices': np.array([[25,5,1],[25,5,-1]]), 'angle': 1.5, 'ap_position': 5, 'label':8, 'angular_order': 8},
        {'type': 'second_premolar', 'center': np.array([-25,5,0]), 'vertices': np.array([[-25,5,1],[-25,5,-1]]), 'angle': -1.5, 'ap_position': 5, 'label':9, 'angular_order': 9},
        {'type': 'first_molar', 'center': np.array([30,0,0]), 'vertices': np.array([[30,0,1],[30,0,-1]]), 'angle': 2.0, 'ap_position': 0, 'label':10, 'angular_order': 10},
        {'type': 'first_molar', 'center': np.array([-30,0,0]), 'vertices': np.array([[-30,0,1],[-30,0,-1]]), 'angle': -2.0, 'ap_position': 0, 'label':11, 'angular_order': 11}
    ]
    generator.teeth.sort(key=lambda t: t['angle'])
    for i, tooth in enumerate(generator.teeth): tooth['angular_order'] = i

    generator.arch_type = 'lower'; generator.lr_axis = 0; generator.ap_axis = 1; generator.height_axis = 2
    dummy_mesh_vertices = [];
    for tooth in generator.teeth: dummy_mesh_vertices.extend(tooth['vertices'])
    if not dummy_mesh_vertices: dummy_mesh_vertices = [[-40,-20,-5],[40,-20,-5], [0,20,-5], [-40,20,5], [40,20,5], [0,0,10]]
    dummy_mesh = o3d.geometry.TriangleMesh(); dummy_mesh.vertices = o3d.utility.Vector3dVector(np.array(dummy_mesh_vertices))
    if len(dummy_mesh_vertices) >=3: dummy_mesh.triangles = o3d.utility.Vector3iVector(np.array([[0,1,2]]))
    generator.mesh = dummy_mesh
    if generator.arch_type not in generator.BRACKET_HEIGHTS: generator.BRACKET_HEIGHTS[generator.arch_type] = generator.BRACKET_HEIGHTS['lower']

    generator.bracket_positions = []
    mesh_center_val = generator.mesh.get_center() if generator.mesh and generator.mesh.has_vertices() else np.array([0,0,0])
    for i, tooth in enumerate(generator.teeth):
        bracket_height_val = generator.BRACKET_HEIGHTS[generator.arch_type].get(tooth['type'], 4.5)
        bracket_pos = tooth['center'].copy().astype(float)
        if generator.arch_type == 'upper':
             bracket_pos[generator.height_axis] = (np.max(tooth['vertices'][:, generator.height_axis]) if tooth['vertices'].size >0 else tooth['center'][generator.height_axis]) + bracket_height_val
        else:
             bracket_pos[generator.height_axis] = (np.min(tooth['vertices'][:, generator.height_axis]) if tooth['vertices'].size >0 else tooth['center'][generator.height_axis]) - bracket_height_val
        direction_from_center = bracket_pos - mesh_center_val; direction_from_center[generator.height_axis] = 0
        norm_direction = np.array([0,1,0])
        if np.linalg.norm(direction_from_center) > 1e-6: norm_direction = direction_from_center / np.linalg.norm(direction_from_center)
        bracket_pos += norm_direction * 2.5
        generator.bracket_positions.append({'position': bracket_pos, 'tooth_type': tooth['type'], 'tooth_index': i,
                                           'tooth_center': tooth['center'], 'direction': norm_direction})

    num_incisors = sum(1 for t in generator.teeth if 'incisor' in t['type'])
    num_canines = sum(1 for t in generator.teeth if t['type'] == 'canine')
    num_others = sum(1 for t in generator.teeth if 'molar' in t['type'] or 'premolar' in t['type'])
    print(f"Mock teeth: {len(generator.teeth)} total\n  Incisors: {num_incisors}, Canines: {num_canines}, Others: {num_others}")
    assert num_incisors == 4, f"Incisor count. Expected 4, got {num_incisors}"; assert num_canines == 2, f"Canine count. Expected 2, got {num_canines}"; assert num_others == 6, f"Other count. Expected 6, got {num_others}"
    print("Tooth count assertions passed.")

    archform_points = generator.create_ideal_archform(); assert archform_points is not None, "create_ideal_archform is None"
    print(f"Archform created with {len(archform_points)} points.")
    clinical_wire_points = generator.add_clinical_bends(archform_points)
    assert len(clinical_wire_points) == len(generator.bracket_positions), f"Wire points ({len(clinical_wire_points)}) vs brackets ({len(generator.bracket_positions)})"
    print("Clinical wire points count matches bracket positions count assertion passed.")

    optimized_wire = generator.optimize_for_production(list(clinical_wire_points))
    assert len(optimized_wire) <= len(clinical_wire_points), "Optimized wire has more points."
    if len(clinical_wire_points) > 28 : assert len(optimized_wire) < len(clinical_wire_points), "Optimization didn't reduce points."
    print("Optimization logic basic check passed.")

    instructions = generator.calculate_bending_instructions(clinical_wire_points)
    assert len(instructions) == len(clinical_wire_points), "Instructions count mismatch."
    if instructions: assert 'feed_distance' in instructions[0] and 'bend_angle' in instructions[0], "Instruction format error."
    print("Bending instructions calculation basic check passed.")

    wire_mesh = generator.create_wire_mesh(clinical_wire_points)
    if len(clinical_wire_points) >=2: assert wire_mesh is not None and wire_mesh.has_vertices(), "Wire mesh error (valid input)."
    else: assert wire_mesh is None or (not wire_mesh.has_vertices() if wire_mesh else True), "Wire mesh error (invalid input)."
    print("Wire mesh creation basic check passed.")

    bracket_markers = generator.create_bracket_markers()
    if generator.bracket_positions: assert bracket_markers is not None and bracket_markers.has_vertices(), "Bracket markers error."
    else: assert bracket_markers is not None and not bracket_markers.has_vertices(), "Bracket markers should be empty."
    print("Bracket markers creation basic check passed.")
    print("\nBasic tests passed!")

def test_color_visualization():
    """Creates a minimal Open3D scene to test color rendering of markers."""
    print("\n--- Running Color Visualization Test ---")
    mock_brackets = [
        {'position': np.array([0, 0, 0]), 'tooth_type': 'central_incisor'},
        {'position': np.array([5, 0, 0]), 'tooth_type': 'central_incisor'},
        {'position': np.array([10, 0, 0]), 'tooth_type': 'canine'},
        {'position': np.array([15, 0, 0]), 'tooth_type': 'first_premolar'},
        {'position': np.array([-5, 0, 0]), 'tooth_type': 'lateral_incisor'}, # Another incisor
        {'position': np.array([-10, 0, 0]), 'tooth_type': 'second_molar'}, # Another green
    ]

    all_markers = o3d.geometry.TriangleMesh()

    print("Creating mock bracket markers with specified colors:")
    for bracket in mock_brackets:
        # Create small cube for bracket marker
        # Box dimensions: width (X), height (Y), depth (Z)
        marker_box = o3d.geometry.TriangleMesh.create_box(width=1.5, height=2.0, depth=1.0)

        # Color based on tooth type (logic copied from create_bracket_markers)
        color_applied = "Unknown"
        if 'incisor' in bracket['tooth_type']:
            marker_box.paint_uniform_color([0.2, 0.2, 0.8])  # Blue
            color_applied = "Blue ([0.2, 0.2, 0.8])"
        elif 'canine' in bracket['tooth_type']:
            marker_box.paint_uniform_color([0.8, 0.8, 0.2])  # Yellow
            color_applied = "Yellow ([0.8, 0.8, 0.2])"
        else: # Premolars and Molars
            marker_box.paint_uniform_color([0.2, 0.8, 0.2])  # Green
            color_applied = "Green ([0.2, 0.8, 0.2])"

        # Center the box at its origin before translating
        marker_box.translate([-0.75, -1.0, -0.5], relative=True)
        # Translate to its final position
        marker_box.translate(bracket['position'], relative=False)

        print(f"  Tooth Type: {bracket['tooth_type']}, Position: {bracket['position']}, Color: {color_applied}")
        all_markers += marker_box

    # Create a dummy arch mesh (e.g., a larger sphere at origin)
    dummy_arch_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    dummy_arch_mesh.translate([5, -5, 0]) # Offset it slightly
    dummy_arch_mesh.paint_uniform_color([0.5, 0.5, 0.5]) # Gray
    print("Added a gray sphere as a reference dummy arch.")

    # Initialize visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Test Color Visualization", width=800, height=600)

    vis.add_geometry(dummy_arch_mesh)
    if all_markers.has_vertices(): # Only add if it's not empty
        vis.add_geometry(all_markers)
    else:
        print("Warning: all_markers mesh is empty. No bracket markers to show.")

    # Add coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0,0,0])
    vis.add_geometry(coord_frame)

    # Set render options
    opt = vis.get_render_option()
    opt.background_color = np.array([0.2, 0.2, 0.2]) # Dark gray background
    opt.mesh_show_wireframe = False
    opt.point_size = 5.0

    print("\nVisualizer running. Check colors of the boxes:")
    print("- Central/Lateral Incisors should be BLUE.")
    print("- Canines should be YELLOW.")
    print("- Premolars/Molars should be GREEN.")
    print("Close the visualizer window to continue.")

    vis.run()
    vis.destroy_window()
    print("Color visualization test finished.")


def main():
    """Main execution function."""
    stl_path = "/Users/galala/STL con/assets/AyaKhairy_LowerJaw.stl"
    if not os.path.exists(stl_path):
        print(f"FATAL: STL file not found at {stl_path}. Cannot proceed with main execution.")
        print(f"Attempting to create a dummy STL file: dummy_jaw.stl for basic run.")
        points = np.array([
            [0,0,0], [10,0,0], [0,10,0], [10,10,0], # Base
            [0,0,5], [10,0,5], [0,10,5], [10,10,5], # Top
            [5,5,10] # A peak to simulate a tooth
        ])
        dummy_stl_content = "solid dummy_jaw\n"
        facets = [
            (0,1,2), (1,3,2), (4,5,6), (5,7,6), (0,1,5), (0,4,5),
            (1,3,7), (1,5,7), (2,3,7), (2,6,7), (0,2,6), (0,4,6),
            (4,5,8), (5,7,8), (6,7,8), (4,6,8)
        ]
        for f_indices in facets:
            p1,p2,p3 = points[f_indices[0]], points[f_indices[1]], points[f_indices[2]]
            normal = np.cross(p2-p1, p3-p1).astype(float)
            norm_val = np.linalg.norm(normal)
            if norm_val > 0: normal /= norm_val
            else: normal = np.array([0,0,1])
            dummy_stl_content += f"facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\nouter loop\n"
            dummy_stl_content += f"vertex {p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f}\nvertex {p2[0]:.6f} {p2[1]:.6f} {p2[2]:.6f}\nvertex {p3[0]:.6f} {p3[1]:.6f} {p3[2]:.6f}\n"
            dummy_stl_content += "endloop\nendfacet\n"
        dummy_stl_content += "endsolid dummy_jaw\n"
        stl_path = "dummy_jaw.stl"
        try:
            with open(stl_path, "w") as f_io: f_io.write(dummy_stl_content)
            print(f"Created dummy STL: {stl_path}")
        except Exception as e: print(f"Could not create dummy STL: {e}. Exiting."); return

    generator = OrthodonticWireGenerator(stl_path=stl_path, arch_type='auto', wire_size='0.018')
    result = generator.generate_wire()
    if result and result.get('production_ready'): print("\n✅ SUCCESS: Professional orthodontic wire generated"); generator.visualize(result)
    elif result: print("\n⚠️ Wire generated but NOT production ready. Review recommendations."); generator.visualize(result)
    else: print("\n❌ FAILED: Could not generate wire")

if __name__ == "__main__":
    # run_tests()
    main() # Revert to main() for user's primary use case
    # To run the color test in a graphical environment:
    # test_color_visualization()
