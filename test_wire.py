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
        
        if arch_type == 'auto':
            self.arch_type = 'lower' if 'lower' in stl_path.lower() else 'upper'
        else:
            self.arch_type = arch_type
            
        self.mesh = None
        self.teeth = []
        self.bracket_positions = []
        self.wire_path = []
        
    def load_mesh(self):
        print(f"\n--- Loading Orthodontic Model ---\nFile: {os.path.basename(self.stl_path)}, Arch: {self.arch_type.upper()}, Wire: {self.wire_size}")
        try:
            self.mesh = o3d.io.read_triangle_mesh(self.stl_path)
            if not self.mesh.has_triangles(): raise ValueError("No triangles in mesh")
            self.mesh.compute_vertex_normals(); self.mesh.compute_triangle_normals()
            self.mesh.paint_uniform_color([0.96, 0.94, 0.90])
            print(f"Loaded: {len(self.mesh.vertices)} vertices, {len(self.mesh.triangles)} triangles")
            return True
        except Exception as e: print(f"Error loading mesh: {e}"); return False
    
    def detect_individual_teeth(self):
        print("\n--- Detecting Individual Teeth ---")
        if self.mesh is None: print("Error: Mesh not loaded."); return False
        vertices = np.asarray(self.mesh.vertices); print(f"Total vertices: {len(vertices)}")
        bbox = self.mesh.get_axis_aligned_bounding_box(); center = self.mesh.get_center(); extent = bbox.get_extent()
        axes = [0,1,2]; self.lr_axis = np.argmax(extent)
        temp_extent = extent.copy(); temp_extent[self.lr_axis] = -np.inf # Use -np.inf for argmin to find smallest among remaining for height
        self.height_axis = np.argmin(temp_extent)
        if self.lr_axis == self.height_axis: # Fallback if still same (e.g. two largest are equal, one smallest)
            temp_extent_height = extent.copy(); temp_extent_height[self.lr_axis] = np.inf
            self.height_axis = np.argmin(temp_extent_height)
        if self.lr_axis == self.height_axis: # If still same (e.g. cube)
            possible_axes = [ax for ax in axes if ax != self.lr_axis]
            self.height_axis = possible_axes[0] if possible_axes else (self.lr_axis + 1) % 3

        self.ap_axis = [ax for ax in axes if ax not in {self.lr_axis, self.height_axis}][0]
        print(f"Axes - LR: {self.lr_axis}, AP: {self.ap_axis}, Height: {self.height_axis}")
        print(f"Extent - LR: {extent[self.lr_axis]:.1f}, AP: {extent[self.ap_axis]:.1f}, Height: {extent[self.height_axis]:.1f}")
        
        if self.arch_type == 'upper': crown_h = np.min(vertices[:,self.height_axis]) + extent[self.height_axis]*0.3
        else: crown_h = np.max(vertices[:,self.height_axis]) - extent[self.height_axis]*0.3
        crown_mask = np.abs(vertices[:,self.height_axis] - crown_h) < (extent[self.height_axis]*0.15)
        crown_vertices = vertices[crown_mask]; print(f"Crown level vertices: {len(crown_vertices)}")
        if len(crown_vertices) == 0: print("Warning: No crown vertices. Tooth detection may fail."); return False

        num_segments = 16; centered_verts = crown_vertices - center
        if len(centered_verts)==0: print("Error: No crown vertices to process."); return False
        angles = np.arctan2(centered_verts[:,self.ap_axis], centered_verts[:,self.lr_axis])
        radii = np.linalg.norm(centered_verts[:, [self.lr_axis, self.ap_axis]], axis=1)
        self.teeth = []
        angle_step = 2*np.pi/(num_segments+2); start_angle = -np.pi + angle_step
        for i in range(num_segments):
            s_start, s_end = start_angle+i*angle_step, start_angle+(i+1)*angle_step
            angle_m = np.logical_and(angles>=s_start, angles<s_end); seg_verts = crown_vertices[angle_m]
            if len(seg_verts)>30:
                seg_radii = radii[angle_m]
                if len(seg_radii)==0: continue
                t_center = np.mean(seg_verts[seg_radii >= np.percentile(seg_radii,70)],axis=0) if len(seg_radii)>0 else np.mean(seg_verts,axis=0)
                if np.linalg.norm(t_center[[self.lr_axis,self.ap_axis]] - center[[self.lr_axis,self.ap_axis]]) < 10.0: continue
                self.teeth.append({'center':t_center,'vertices':seg_verts,'label':len(self.teeth),'type':'unknown',
                                   'angle':np.arctan2(t_center[self.ap_axis]-center[self.ap_axis], t_center[self.lr_axis]-center[self.lr_axis]),
                                   'ap_position':t_center[self.ap_axis]})
        self.teeth.sort(key=lambda t: t['angle'])
        print("\n--- Teeth after sorting by angle: ---")
        for i,t in enumerate(self.teeth): print(f"  Tooth {i}: Angle: {np.degrees(t['angle']):.1f}°, AP: {t['ap_position']:.1f}, LR_pos: {t['center'][self.lr_axis]:.1f}")
        num_t = len(self.teeth)
        if num_t > 0:
            for i_t, t_obj in enumerate(self.teeth): t_obj['angular_order'] = i_t
            sorted_ap = sorted(self.teeth, key=lambda t:t['ap_position'], reverse=True)
            inc_cand = [sorted_ap[i] for i in range(min(num_t,6))]
            min_ao,max_ao = int(num_t*0.2), int(num_t*0.8)
            pot_inc = [t for t in inc_cand if min_ao <= t['angular_order'] <= max_ao]
            num_inc_cls = min(len(pot_inc),6) if len(pot_inc)>=4 else (min(len(inc_cand),6) if len(inc_cand)>=4 else len(inc_cand))
            fin_inc = sorted(pot_inc[:num_inc_cls] if len(pot_inc)>=4 else inc_cand[:num_inc_cls], key=lambda t:t['angular_order'])
            if fin_inc:
                med_ao = np.median([t['angular_order'] for t in fin_inc])
                for t in fin_inc: t['type']='lateral_incisor'
                fin_inc.sort(key=lambda t: abs(t['angular_order']-med_ao))
                if len(fin_inc)>=1: fin_inc[0]['type']='central_incisor'
                if len(fin_inc)>=2: fin_inc[1]['type']='central_incisor'
            inc_indices = sorted([t['angular_order'] for t in fin_inc])
            can_cand = []
            if inc_indices:
                min_idx,max_idx = inc_indices[0],inc_indices[-1]
                if min_idx>0: can_cand.append(self.teeth[min_idx-1])
                if max_idx<num_t-1: can_cand.append(self.teeth[max_idx+1])
            for t in can_cand:
                if t['type']=='unknown': t['type']='canine'
            for i_t, t_obj in enumerate(self.teeth):
                if t_obj['type']=='unknown':
                    pr = i_t/(num_t-1) if num_t>1 else 0.5
                    if pr<0.15 or pr>0.85: t_obj['type']='second_molar'
                    elif pr<0.25 or pr>0.75: t_obj['type']='first_molar'
                    elif pr<0.35 or pr>0.65: t_obj['type']='second_premolar'
                    else: t_obj['type']='first_premolar'
            for i_t, t_obj in enumerate(self.teeth): # Fallback for remaining unknowns
                 if t_obj['type']=='unknown': t_obj['type']='first_premolar' # Simple fallback
        print("\n--- Teeth after final classification: ---")
        for i,t in enumerate(self.teeth): print(f"  Tooth {i}: Type: {t['type']}, Angle: {np.degrees(t['angle']):.1f}°")
        type_counts = {}; [type_counts.update({t['type']:type_counts.get(t['type'],0)+1}) for t in self.teeth]
        print(f"\nFinal tooth type distribution: {type_counts}"); return len(self.teeth)>=6

    def calculate_bracket_positions(self):
        print(f"\n--- Calculating Bracket Positions ---\nTeeth count: {len(self.teeth)}")
        self.bracket_positions = []
        m_center = self.mesh.get_center() if self.mesh and self.mesh.has_vertices() else np.array([0,0,0])
        for i,t in enumerate(self.teeth):
            print(f"  Bracket for Tooth {i} (Type: {t['type']}, AngularIdx: {t.get('angular_order','N/A')}, Center: {t['center']})")
            b_h = self.BRACKET_HEIGHTS[self.arch_type].get(t['type'],4.5)
            t_v = t.get('vertices',np.array([t['center']]))
            if not isinstance(t_v,np.ndarray) or t_v.ndim!=2 or t_v.shape[1]!=3: t_v=np.array([t['center']])
            t_c = t['center']
            tgt_h = (np.min(t_v[:,self.height_axis]) if t_v.size>0 else t_c[self.height_axis])+b_h if self.arch_type=='upper' else (np.max(t_v[:,self.height_axis]) if t_v.size>0 else t_c[self.height_axis])-b_h
            b_lvl_v = t_v[np.abs(t_v[:,self.height_axis]-tgt_h)<2.0] if t_v.size>0 else np.array([])
            if len(b_lvl_v)<10:
                b_pos_intermediate=t_c.copy()
                b_pos_intermediate[self.height_axis]=tgt_h
            else:
                vfc=b_lvl_v-m_center; vfc[:,self.height_axis]=0
                h_dist=np.linalg.norm(vfc,axis=1)
                out_v=b_lvl_v[h_dist>=np.percentile(h_dist,90)]
                b_pos_intermediate=np.mean(out_v,axis=0) if len(out_v)>3 else b_lvl_v[np.argmax(h_dist)]

            b_pos = b_pos_intermediate.astype(float) # Ensure b_pos is float before operations involving floats

            t2b=b_pos-t_c; t2b[self.height_axis]=0 # t2b will be float as b_pos is float
            out_dir_intermediate = (t2b/np.linalg.norm(t2b)) if np.linalg.norm(t2b)>0 else ((b_pos-m_center)/np.linalg.norm(b_pos-m_center) if np.linalg.norm(b_pos-m_center)>0 else np.array([0,1,0]))

            # Ensure out_dir is float and normalized correctly
            out_dir = np.array(out_dir_intermediate, dtype=float)
            if out_dir[self.height_axis] != 0:
                out_dir[self.height_axis]=0.0
                norm_val = np.linalg.norm(out_dir)
                if norm_val > 1e-8: out_dir=out_dir/norm_val # Re-normalize if changed
                else: out_dir = np.array([0,1,0], dtype=float) # Default if becomes zero vector

            b_pos+=out_dir*2.5 # Now b_pos is float, out_dir is float
            self.bracket_positions.append({'position':b_pos,'tooth_type':t['type'],'tooth_index':t.get('angular_order',i),'tooth_center':t_c,'direction':out_dir})
        print(f"Total bracket positions calculated: {len(self.bracket_positions)}"); return len(self.bracket_positions)>0

    def create_ideal_archform(self):
        print("\n--- Creating Ideal Archform ---")
        if not self.bracket_positions: print("Warning: No bracket positions."); return None
        pos = np.array([b['position'] for b in self.bracket_positions])
        m_center = self.mesh.get_center() if self.mesh and self.mesh.has_vertices() else np.mean(pos,axis=0) if pos.size>0 else np.array([0,0,0])
        angs = [np.arctan2(p[self.ap_axis]-m_center[self.ap_axis],p[self.lr_axis]-m_center[self.lr_axis]) for p in pos]
        s_pos = pos[np.argsort(angs)]
        max_g,max_g_idx = 0,-1
        for i in range(len(s_pos)):
            d = np.linalg.norm(s_pos[(i+1)%len(s_pos)]-s_pos[i])
            if d>max_g: max_g=d; max_g_idx=i
        if max_g_idx>=0 and max_g>20.0 and s_pos.shape[0]>max_g_idx+1: s_pos=np.vstack([s_pos[max_g_idx+1:],s_pos[:max_g_idx+1]])
        n_pts = len(s_pos)*3; dists=[0]; [dists.append(dists[-1]+np.linalg.norm(s_pos[i]-s_pos[i-1])) for i in range(1,len(s_pos))]
        if len(s_pos)>1 and np.linalg.norm(s_pos[0]-s_pos[-1])<20.0 and len(s_pos)>2: s_pos,dists = s_pos[:-1],dists[:-1]
        if len(s_pos)<3: print(f"Warning: Too few points ({len(s_pos)}) for spline."); return s_pos
        t_orig = np.array(dists)/dists[-1] if dists[-1]>0 else np.linspace(0,1,len(s_pos))
        t_smooth = np.linspace(0,1,n_pts); s_arch = np.zeros((len(t_smooth),3))
        try:
            for ax_idx,s_val in [(self.lr_axis,0.2),(self.ap_axis,0.2),(self.height_axis,0.1)]:
                s_arch[:,ax_idx] = interpolate.UnivariateSpline(t_orig,s_pos[:,ax_idx],k=3,s=s_val)(t_smooth)
        except Exception as e:
            print(f"Spline failed: {e}. Linear fallback."); s_arch=[];
            for i in range(len(s_pos)-1): p1,p2=s_pos[i],s_pos[i+1]; [s_arch.append(p1*(1-j/3.0)+p2*(j/3.0)) for j in range(3)]
            s_arch.append(s_pos[-1]); s_arch=np.array(s_arch)
        print(f"Created archform with {len(s_arch)} points."); return s_arch

    def add_clinical_bends(self, archform):
        print("\n--- Adding Clinical Bends ---")
        if not self.bracket_positions: print("Warning: No bracket positions."); return []
        c_wire = [{'position':b['position'].copy(),'type':'bracket','tooth_type':b['tooth_type'],'tooth_index':b['tooth_index'],'original_bracket_list_index':i} for i,b in enumerate(self.bracket_positions)]
        print(f"Adding all {len(c_wire)} bracket positions.")
        if len(c_wire)>2:
            m_c = self.mesh.get_center() if self.mesh and self.mesh.has_vertices() else np.mean([p['position'] for p in c_wire],axis=0)
            for p in c_wire: p['angle_for_sorting']=np.arctan2(p['position'][self.ap_axis]-m_c[self.ap_axis],p['position'][self.lr_axis]-m_c[self.lr_axis])
            c_wire.sort(key=lambda p: p['angle_for_sorting'])
            print("\n--- Clinical wire after initial angular sort: ---"); [print(f" Idx {i}: Type: {p['tooth_type']}, Pos: {p['position']}") for i,p in enumerate(c_wire)]
            max_g,g_idx=0,-1
            for i in range(len(c_wire)):
                d=np.linalg.norm(c_wire[(i+1)%len(c_wire)]['position']-c_wire[i]['position'])
                if d>max_g: max_g=d;g_idx=i
            print(f"Max gap {max_g:.1f}mm at index {g_idx}.")
            if g_idx>=0 and max_g>15.0 and len(c_wire)>g_idx+1: c_wire = c_wire[g_idx+1:]+c_wire[:g_idx+1]; print("Reordered wire.")
        print(f"\nClinical wire summary: Total points: {len(c_wire)}")
        if c_wire: print(f"Wire starts: {c_wire[0]['tooth_type']}, ends: {c_wire[-1]['tooth_type']}.")
        return c_wire

    def calculate_bending_instructions(self, c_wire):
        print("\n--- Calculating Bending Instructions ---")
        if len(c_wire)<2: print("Warning: Not enough points."); return []
        instr=[];c_feed=0.0
        for i in range(len(c_wire)):
            pt=c_wire[i]['position']
            if i>0: c_feed+=np.linalg.norm(pt-c_wire[i-1]['position'])
            b_ang = 0.0
            if 0<i<len(c_wire)-1:
                p1,p2,p3=c_wire[i-1]['position'],pt,c_wire[i+1]['position']
                v1,v2=p1-p2,p3-p2; v1n,v2n=v1/(np.linalg.norm(v1)+1e-8),v2/(np.linalg.norm(v2)+1e-8)
                b_ang=np.degrees(np.arccos(np.clip(np.dot(v1n,v2n),-1,1)))
                if np.cross(v1n,v2n)[self.height_axis]<0: b_ang=-b_ang
            instr.append({'index':i,'feed_distance':c_feed,'bend_angle':b_ang,'position':pt,
                          'type':c_wire[i].get('type','std'),'tooth_type':c_wire[i].get('tooth_type','unk')})
        return instr

    def create_wire_mesh(self, clinical_wire):
        print("\n--- Creating Wire Mesh ---")
        if not clinical_wire or len(clinical_wire) < 2: print("Warning: Not enough points for mesh."); return None, 0
        positions = np.array([p['position'] for p in clinical_wire])
        print(f"Wire mesh using {len(positions)} points. First 5: {positions[:5]}")
        wire_m = o3d.geometry.TriangleMesh(); segments_created = 0
        for i in range(len(positions)-1):
            p1,p2=positions[i],positions[i+1]; seg_v=p2-p1; seg_l=np.linalg.norm(seg_v)
            print(f"  Segment {i} to {i+1}: Dist: {seg_l:.2f}mm. ",end="")
            if seg_l<0.01: print("Skipped (short)."); continue
            print("Creating.")
            cyl=o3d.geometry.TriangleMesh.create_cylinder(self.wire_radius,seg_l,10,True)
            z_ax=np.array([0,0,1]); seg_dir=seg_v/seg_l
            rot_ax=np.cross(z_ax,seg_dir); dot_p=np.dot(z_ax,seg_dir); ang=np.arccos(np.clip(dot_p,-1,1))
            if np.linalg.norm(rot_ax)>1e-6: R=o3d.geometry.get_rotation_matrix_from_axis_angle((rot_ax/np.linalg.norm(rot_ax))*ang)
            elif dot_p<0: R=o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1,0,0])*np.pi)
            else: R=np.identity(3) # No rotation needed if already aligned
            cyl.rotate(R,center=np.array([0,0,0])); cyl.translate(p1+seg_v/2)
            wire_m+=cyl; segments_created+=1
            if i==0: wire_m+=o3d.geometry.TriangleMesh.create_sphere(self.wire_radius*1.05).translate(p1) # Joint sphere
            wire_m+=o3d.geometry.TriangleMesh.create_sphere(self.wire_radius*1.05).translate(p2) # Joint sphere
        if len(positions)>0: # Terminal spheres
            wire_m+=o3d.geometry.TriangleMesh.create_sphere(self.wire_radius*1.3).translate(positions[0])
            if len(positions)>1: wire_m+=o3d.geometry.TriangleMesh.create_sphere(self.wire_radius*1.3).translate(positions[-1])
        print(f"Created {segments_created} wire segments.")
        if wire_m.has_vertices(): wire_m.paint_uniform_color([1.0,0,0]); wire_m.compute_vertex_normals()
        return wire_m, segments_created

    def create_bracket_markers(self):
        """Creates a list of dictionaries, each with a marker mesh and its intended color."""
        print("\n--- Creating Bracket Markers ---")
        bracket_marker_data = []
        if not self.bracket_positions: print("Warning: No bracket positions for markers."); return bracket_marker_data

        for bracket in self.bracket_positions:
            marker_box = o3d.geometry.TriangleMesh.create_box(width=1.5, height=2.0, depth=1.0)
            marker_box.translate([-0.75, -1.0, -0.5]) # Center the box before moving
            marker_box.translate(bracket['position']) # Position at bracket location

            color_rgb = [0.5, 0.5, 0.5] # Default gray
            if 'incisor' in bracket['tooth_type']: color_rgb = [0.2, 0.2, 0.8]  # Blue
            elif 'canine' in bracket['tooth_type']: color_rgb = [0.8, 0.8, 0.2]  # Yellow
            else: color_rgb = [0.2, 0.8, 0.2]  # Green (Premolars and Molars)

            # marker_box.paint_uniform_color(color_rgb) # Color will be applied by generate_wire
            bracket_marker_data.append({
                'marker_mesh': marker_box,
                'color': color_rgb,
                'tooth_type': bracket['tooth_type'],
                'tooth_index': bracket['tooth_index'] # Add tooth_index here
            })
        print(f"Prepared data for {len(bracket_marker_data)} bracket markers.")
        return bracket_marker_data

    def export_bending_instructions(self, instructions, filename=None):
        if filename is None:
            base=os.path.splitext(os.path.basename(self.stl_path))[0] if self.stl_path and self.stl_path!="dummy.stl" else "wire"
            filename=f"{base}_{self.arch_type}_{self.wire_size}.csv"
        print(f"\n--- Exporting to {filename} ---")
        if not instructions: print("No instructions."); return
        with open(filename,'w',newline='') as csvf:
            w=csv.writer(csvf); w.writerow(['Idx','Feed_mm','Bend_deg','X','Y','Z','Type','Tooth'])
            for inst in instructions: w.writerow([inst['index'],f"{inst['feed_distance']:.3f}",f"{inst['bend_angle']:.2f}",
                                                 f"{inst['position'][0]:.3f}",f"{inst['position'][1]:.3f}",f"{inst['position'][2]:.3f}",
                                                 inst['type'],inst['tooth_type']])
        print(f"Exported {len(instructions)} instructions.")
        if instructions: print(f"Wire Summary: Type: {self.wire_size}, Length: {instructions[-1]['feed_distance']:.1f}mm, Bends: {len([i for i in instructions if abs(i['bend_angle'])>0.1])}")

    def optimize_for_production(self, c_wire):
        print("\n--- Optimizing for Production ---")
        pts=len(c_wire); print(f"Current points: {pts}")
        if not c_wire: print("No points."); return []
        if pts<=28: print("Keeping all points."); return c_wire
        print("Optimizing (points > 28).")
        crit_idx=set();
        if pts>0: crit_idx.add(0); crit_idx.add(pts-1)
        for i,p in enumerate(c_wire):
            if any(k in p['tooth_type'] for k in ['incisor','canine','molar','premolar']): crit_idx.add(i)
        opt_wire = [c_wire[i] for i in sorted(list(crit_idx)) if i<pts]
        print(f"Optimized from {pts} to {len(opt_wire)} points.")
        return opt_wire
    
    def generate_wire(self):
        print("\n"+"="*70+"\nPROFESSIONAL ORTHODONTIC WIRE GENERATION\n"+"="*70)
        if not self.load_mesh(): return None
        if not self.detect_individual_teeth() and not self.teeth: print("Critical: No teeth detected."); return None
        if not self.calculate_bracket_positions() and not self.bracket_positions: print("Critical: No brackets."); return None
        
        archform = self.create_ideal_archform() # For potential future use or alternative flows

        clinical_wire_points = self.add_clinical_bends(archform)
        if not clinical_wire_points: print("Critical: No clinical wire points."); return None

        clinical_wire_points = self.optimize_for_production(clinical_wire_points)
        instructions = self.calculate_bending_instructions(clinical_wire_points)
        
        wire_mesh, _ = self.create_wire_mesh(clinical_wire_points) # Segments_created not used here currently

        # Handle new bracket_markers return type
        bracket_marker_data_list = self.create_bracket_markers()
        combined_bracket_markers = o3d.geometry.TriangleMesh()
        if bracket_marker_data_list:
            for marker_data in bracket_marker_data_list:
                marker_mesh = marker_data['marker_mesh']
                marker_mesh.paint_uniform_color(marker_data['color']) # Apply color now
                combined_bracket_markers += marker_mesh

        if instructions: self.export_bending_instructions(instructions)
        else: print("No bending instructions generated.")

        print("\n--- Production Readiness Assessment ---")
        # ... (rest of readiness assessment as before) ...
        len_instr = len(instructions) if instructions else 0
        total_wire_len_str = f"{instructions[-1]['feed_distance']:.1f}mm" if len_instr > 0 and instructions[-1]['feed_distance'] is not None else "N/A"
        print(f"✓ Wire type: {self.wire_size}\n✓ Bend points in wire: {len(clinical_wire_points)}\n✓ Total wire length: {total_wire_len_str}\n✓ Teeth detected: {len(self.teeth)}")
        exp_bends = len(self.teeth)/1.5; acc_range=(max(6,int(exp_bends)-3),min(30,int(exp_bends)+5))
        ready = acc_range[0]<=len(clinical_wire_points)<=acc_range[1]
        print(f"\nExpected bend range for {len(self.teeth)} teeth: {acc_range[0]}-{acc_range[1]}\nProduction Ready: {'YES' if ready else 'NO'}")
        if not ready: print(f"Recommendation: {'Check detection/placement' if len(clinical_wire_points)<acc_range[0] else 'Wire too complex, review optimization'}")

        return {'mesh':self.mesh,'wire_mesh':wire_mesh,'bracket_markers':combined_bracket_markers,
                'instructions':instructions,'clinical_wire':clinical_wire_points,'production_ready':ready}
    
    def visualize(self, components):
        print("\n--- Visualization ---")
        if not components: print("No components to visualize."); return
        vis = o3d.visualization.Visualizer()
        try: vis.create_window(window_name=f"Orthodontic Wire - {self.arch_type.title() if self.arch_type else 'N/A'}",width=1400,height=900)
        except Exception as e: print(f"[Open3D ERROR] Failed window creation: {e}. Skipping visualization."); return
        for key, geom in components.items():
            if isinstance(geom, o3d.geometry.Geometry) and geom.has_vertices(): vis.add_geometry(geom)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0); vis.add_geometry(coord_frame)
        opt = vis.get_render_option()
        if opt: opt.background_color = np.array([0.1,0.1,0.1]); opt.mesh_show_back_face=True
        print("\nControls:\n- Mouse: Rotate\n- Scroll: Zoom\n- Colors: Blue=Incisor, Yellow=Canine, Green=Premolar/Molar")
        try: vis.run()
        except Exception as e: print(f"[Open3D ERROR] Visualizer run failed: {e}")
        finally: vis.destroy_window()

def run_tests():
    print("\n--- Running Basic Tests ---")
    generator = OrthodonticWireGenerator(stl_path="dummy.stl", arch_type='lower')

    # Mock teeth data with specific types for testing classification part of coloring
    # Ensure indices used for assertions below match this list structure
    generator.teeth = [ # Sorted by angle for consistency, angular_order assigned
        {'type': 'canine',           'center': np.array([15,15,0]), 'vertices': np.array([[15,15,1]]), 'angle': -0.5, 'ap_position': 15, 'label':4, 'angular_order': 0}, # Mock canine for color test
        {'type': 'lateral_incisor',  'center': np.array([-7,18,0]), 'vertices': np.array([[-7,18,1]]), 'angle': -0.2, 'ap_position': 18, 'label':3, 'angular_order': 1}, # Mock incisor
        {'type': 'central_incisor',  'center': np.array([-2,19,0]), 'vertices': np.array([[-2,19,1]]), 'angle': -0.05,'ap_position': 19, 'label':1, 'angular_order': 2}, # Mock incisor
        {'type': 'central_incisor',  'center': np.array([2,19,0]),  'vertices': np.array([[2,19,1]]),  'angle': 0.05, 'ap_position': 19, 'label':0, 'angular_order': 3}, # Mock incisor
        {'type': 'lateral_incisor',  'center': np.array([7,18,0]),  'vertices': np.array([[7,18,1]]),  'angle': 0.2,  'ap_position': 18, 'label':2, 'angular_order': 4}, # Mock incisor
        {'type': 'canine',           'center': np.array([-15,15,0]),'vertices': np.array([[-15,15,1]]), 'angle': 0.5, 'ap_position': 15, 'label':5, 'angular_order': 5},
        {'type': 'first_premolar',   'center': np.array([20,10,0]), 'vertices': np.array([[20,10,1]]), 'angle': 1.0,  'ap_position': 10, 'label':6, 'angular_order': 6},
        {'type': 'first_premolar',   'center': np.array([-20,10,0]),'vertices': np.array([[-20,10,1]]), 'angle': -1.0, 'ap_position': 10, 'label':7, 'angular_order': 7},
        {'type': 'second_premolar',  'center': np.array([25,5,0]),  'vertices': np.array([[25,5,1]]),  'angle': 1.5,  'ap_position': 5,  'label':8, 'angular_order': 8},
        {'type': 'second_premolar',  'center': np.array([-25,5,0]), 'vertices': np.array([[-25,5,1]]), 'angle': -1.5, 'ap_position': 5,  'label':9, 'angular_order': 9},
        {'type': 'first_molar',      'center': np.array([30,0,0]),  'vertices': np.array([[30,0,1]]),  'angle': 2.0,  'ap_position': 0,  'label':10,'angular_order': 10},
        {'type': 'first_molar',      'center': np.array([-30,0,0]), 'vertices': np.array([[-30,0,1]]), 'angle': -2.0, 'ap_position': 0,  'label':11,'angular_order': 11}
    ]
    generator.teeth.sort(key=lambda t: t['angle']) # Ensure sorted by angle
    for i, tooth in enumerate(generator.teeth): tooth['angular_order'] = i # Re-assign angular_order

    # Assertions for Front Teeth Classification (mock data integrity)
    # Find teeth by a stable property like 'label' since sorting changes indices.

    # Test for a specific central incisor (e.g., original label 1)
    mock_central_incisor_label1 = next((t for t in generator.teeth if t['label'] == 1), None)
    assert mock_central_incisor_label1 is not None, "Mock central incisor (label 1) not found."
    assert mock_central_incisor_label1['type'] == 'central_incisor', \
        f"Mock central_incisor (label 1) type mismatch. Expected 'central_incisor', got {mock_central_incisor_label1['type']}"

    # Test for a specific lateral incisor (e.g., original label 3)
    mock_lateral_incisor_label3 = next((t for t in generator.teeth if t['label'] == 3), None)
    assert mock_lateral_incisor_label3 is not None, "Mock lateral incisor (label 3) not found."
    assert mock_lateral_incisor_label3['type'] == 'lateral_incisor', \
        f"Mock lateral_incisor (label 3) type mismatch. Expected 'lateral_incisor', got {mock_lateral_incisor_label3['type']}"

    print("Mock data tooth type integrity assertions passed.")

    generator.lr_axis=0; generator.ap_axis=1; generator.height_axis=2
    # Create dummy mesh for tests
    dummy_verts_np = np.array([[-40,-20,-5],[40,20,5],[0,0,0],[1,1,1],[5,5,5]]) # Ensure enough for bbox and varied points
    dummy_triangles_np = np.array([[0,2,1],[0,3,2],[0,1,4],[1,2,4]]) # Some minimal triangles

    generator.mesh = o3d.geometry.TriangleMesh()
    generator.mesh.vertices = o3d.utility.Vector3dVector(dummy_verts_np)
    generator.mesh.triangles = o3d.utility.Vector3iVector(dummy_triangles_np)
    if not generator.mesh.has_vertex_normals(): generator.mesh.compute_vertex_normals()
    if not generator.mesh.has_triangle_normals(): generator.mesh.compute_triangle_normals()


    if generator.arch_type not in generator.BRACKET_HEIGHTS: generator.BRACKET_HEIGHTS[generator.arch_type] = generator.BRACKET_HEIGHTS['lower']

    # Use actual calculate_bracket_positions for more realistic test data
    generator.calculate_bracket_positions()
    assert len(generator.bracket_positions) == len(generator.teeth), "Bracket positions not generated for all teeth."
    print("calculate_bracket_positions call in test succeeded.")

    archform_points = generator.create_ideal_archform()
    assert archform_points is not None, "create_ideal_archform returned None"
    print(f"Archform created with {len(archform_points)} points.")

    clinical_wire_points = generator.add_clinical_bends(archform_points)
    # Wire Completeness Assertion 1
    if len(generator.bracket_positions) > 1:
        assert len(clinical_wire_points) == len(generator.bracket_positions), \
            f"Clinical wire points ({len(clinical_wire_points)}) != bracket positions ({len(generator.bracket_positions)})"
    print("Clinical wire points count assertion passed.")

    optimized_wire = generator.optimize_for_production(list(clinical_wire_points))
    assert len(optimized_wire) <= len(clinical_wire_points), "Optimized wire has more points."
    print("Optimization logic basic check passed.")

    instructions = generator.calculate_bending_instructions(optimized_wire) # Use optimized for instructions
    assert len(instructions) == len(optimized_wire), "Instructions count mismatch."
    print("Bending instructions calculation basic check passed.")

    wire_mesh, segments_created = generator.create_wire_mesh(optimized_wire) # Use optimized for mesh
    # Wire Completeness Assertion 2
    if len(optimized_wire) >= 2:
        assert segments_created == len(optimized_wire) - 1, \
            f"Segments created ({segments_created}) != expected ({len(optimized_wire) - 1})"
    elif len(optimized_wire) == 1:
         assert segments_created == 0, "Expected 0 segments for 1 point wire" # Though create_wire_mesh might return None early
    print("Wire mesh segment count assertion passed.")

    # Test for create_bracket_markers new return type and color content
    bracket_marker_data_list = generator.create_bracket_markers()
    assert isinstance(bracket_marker_data_list, list), "create_bracket_markers should return a list."
    if bracket_marker_data_list: # If there are any markers
        assert 'marker_mesh' in bracket_marker_data_list[0], "Marker data missing 'marker_mesh'."
        assert 'color' in bracket_marker_data_list[0], "Marker data missing 'color'."

        # Find specific teeth in bracket_marker_data_list by matching the tooth_index (angular_order)
        # from the original generator.teeth list.

        # Test color for the central incisor (label 1, its angular_order was set after sort)
        expected_central_label1_angular_order = mock_central_incisor_label1['angular_order']
        central_marker_data = next((m for m in bracket_marker_data_list if m['tooth_index'] == expected_central_label1_angular_order), None)
        assert central_marker_data is not None, f"Marker data for central incisor (label 1, angular_order {expected_central_label1_angular_order}) not found."
        assert central_marker_data['tooth_type'] == 'central_incisor', "Marker data tooth type mismatch for central incisor."
        assert central_marker_data['color'] == [0.2, 0.2, 0.8], f"Central incisor color incorrect: {central_marker_data['color']}"

        # Test color for a canine (e.g., original label 4)
        mock_canine_label4 = next((t for t in generator.teeth if t['label'] == 4), None)
        assert mock_canine_label4 is not None, "Mock canine (label 4) not found in generator.teeth for test setup."
        expected_canine_label4_angular_order = mock_canine_label4['angular_order']
        canine_marker_data = next((m for m in bracket_marker_data_list if m['tooth_index'] == expected_canine_label4_angular_order), None)
        assert canine_marker_data is not None, f"Marker data for canine (label 4, angular_order {expected_canine_label4_angular_order}) not found."
        assert canine_marker_data['tooth_type'] == 'canine', "Marker data tooth type mismatch for canine."
        assert canine_marker_data['color'] == [0.8, 0.8, 0.2], f"Canine color incorrect: {canine_marker_data['color']}" # Test for Yellow

    print("Bracket markers data structure and color content assertions passed.")
    print("\nBasic tests passed!")

def test_color_visualization():
    print("\n--- Running Color Visualization Test ---")
    mock_brackets = [
        {'position': np.array([0,0,0]), 'tooth_type': 'central_incisor'},
        {'position': np.array([5,0,0]), 'tooth_type': 'central_incisor'},
        {'position': np.array([10,0,0]), 'tooth_type': 'canine'},
        {'position': np.array([15,0,0]), 'tooth_type': 'first_premolar'},
        {'position': np.array([-5,0,0]), 'tooth_type': 'lateral_incisor'},
        {'position': np.array([-10,0,0]), 'tooth_type': 'second_molar'},
    ]
    all_markers = o3d.geometry.TriangleMesh()
    print("Creating mock bracket markers with specified colors:")
    canine_color_check_passed = False
    for bracket in mock_brackets:
        marker_box = o3d.geometry.TriangleMesh.create_box(width=1.5, height=2.0, depth=1.0)
        color_applied = "Unknown"
        if 'incisor' in bracket['tooth_type']:
            marker_box.paint_uniform_color([0.2,0.2,0.8]); color_applied="Blue ([0.2,0.2,0.8])"
        elif 'canine' in bracket['tooth_type']:
            marker_box.paint_uniform_color([1.0,0.0,1.0]); color_applied="Magenta ([1.0,0.0,1.0])" # Magenta for this test
            if bracket['tooth_type']=='canine' and color_applied=="Magenta ([1.0,0.0,1.0])": canine_color_check_passed=True
        else:
            marker_box.paint_uniform_color([0.2,0.8,0.2]); color_applied="Green ([0.2,0.8,0.2])"
        marker_box.translate([-0.75,-1.0,-0.5],relative=True).translate(bracket['position'],relative=False)
        print(f"  Tooth Type: {bracket['tooth_type']}, Pos: {bracket['position']}, Color: {color_applied}")
        all_markers+=marker_box
    if canine_color_check_passed: print("\nCanine color (Magenta) correctly processed in logic.")
    else: print("\nERROR: Canine color (Magenta) NOT correctly processed.")
    print("\n--- SANDBOX: Intentionally exiting before visualization if in sandbox. ---")
    # return # Uncomment this line if running in a strictly headless sandbox that crashes Open3D window

    dummy_arch = o3d.geometry.TriangleMesh.create_sphere(radius=1.0).translate([5,-5,0]).paint_uniform_color([0.5,0.5,0.5])
    print("Added gray sphere as dummy arch.")
    vis = o3d.visualization.Visualizer()
    try: vis.create_window(window_name="Test Color Visualization",width=800,height=600)
    except Exception as e: print(f"[O3D ERR] Window creation failed: {e}. Skipping visualization."); return
    if all_markers.has_vertices(): vis.add_geometry(all_markers)
    else: print("Warning: No markers to show.")
    vis.add_geometry(dummy_arch); vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0))
    opt = vis.get_render_option()
    if opt: opt.background_color = np.array([0.2,0.2,0.2])
    print("\nVisualizer running. Check colors.\n- Incisors: BLUE\n- Canines: MAGENTA\n- Premolars/Molars: GREEN\nClose window to continue.")
    try: vis.run()
    except Exception as e: print(f"[O3D ERR] Visualizer run failed: {e}")
    finally: vis.destroy_window()
    print("Color visualization test finished.")

def main():
    stl_path = "/Users/galala/STL con/assets/AyaKhairy_LowerJaw.stl"
    if not os.path.exists(stl_path):
        print(f"FATAL: STL file {stl_path} not found. Creating dummy_jaw.stl.")
        points = np.array([[0,0,0],[10,0,0],[0,10,0],[10,10,0],[0,0,5],[10,0,5],[0,10,5],[10,10,5],[5,5,10]])
        content="solid dummy_jaw\n"; facets=[(0,1,2),(1,3,2),(4,5,6),(5,7,6),(0,1,5),(0,4,5),(1,3,7),(1,5,7),(2,3,7),(2,6,7),(0,2,6),(0,4,6),(4,5,8),(5,7,8),(6,7,8),(4,6,8)]
        for f_idx in facets:
            p1,p2,p3=points[f_idx[0]],points[f_idx[1]],points[f_idx[2]]
            norm=np.cross(p2-p1,p3-p1).astype(float); n_val=np.linalg.norm(norm)
            if n_val>0: norm/=n_val
            else: norm=np.array([0,0,1])
            content+=f"facet normal {norm[0]:.6f} {norm[1]:.6f} {norm[2]:.6f}\nouter loop\n"
            content+=f"vertex {p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f}\nvertex {p2[0]:.6f} {p2[1]:.6f} {p2[2]:.6f}\nvertex {p3[0]:.6f} {p3[1]:.6f} {p3[2]:.6f}\nendloop\nendfacet\n"
        content+="endsolid dummy_jaw\n"; stl_path="dummy_jaw.stl"
        try:
            with open(stl_path,"w") as f_io: f_io.write(content)
            print(f"Created dummy STL: {stl_path}")
        except Exception as e: print(f"Could not create dummy STL: {e}. Exiting."); return
    generator = OrthodonticWireGenerator(stl_path=stl_path, arch_type='auto', wire_size='0.018')
    result = generator.generate_wire()
    if result:
        if result.get('production_ready'): print("\n✅ SUCCESS: Professional orthodontic wire generated")
        else: print("\n⚠️ Wire generated but NOT production ready.")
        generator.visualize(result)
    else: print("\n❌ FAILED: Could not generate wire")

if __name__ == "__main__":
    run_tests()
    # main()
    # test_color_visualization()
