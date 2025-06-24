# orthodontic_gui.py - Professional GUI for Orthodontic Wire Design System
# Integrates with tes_stl9claude_latestin.py and orthodontic_processor.py
# Version 3.2 - Restored interactive 3D button and added sample graphic.

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkFont
import os
import sys
import threading
import time
import json
from datetime import datetime
import numpy as np
from pathlib import Path

# Try to import the processor
try:
    from orthodontic_processor import OrthodonticProcessor
    PROCESSOR_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: orthodontic_processor.py not found")
    PROCESSOR_AVAILABLE = False

# Try to import Open3D for visualization
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: Open3D not installed. 3D visualization will be limited.")
    OPEN3D_AVAILABLE = False

# Try to import matplotlib for 2D visualization
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: Matplotlib not installed. 2D/3D visualization will be limited.")
    MATPLOTLIB_AVAILABLE = False

# Try to import pyserial for ESP32 connection
try:
    import serial
    import serial.tools.list_ports
    PYSERIAL_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: PySerial not installed. Direct machine connection will be disabled.")
    PYSERIAL_AVAILABLE = False


class OrthodonticGUI:
    """Professional GUI for Orthodontic Wire Design System."""

    def __init__(self, root):
        self.root = root
        self.root.title("Dental Archwire Design System v3.2")
        self.root.geometry("1400x900")

        # Set minimum window size
        self.root.minsize(1200, 800)

        # Variables
        self.current_file = None
        self.processing_result = None
        self.processor = OrthodonticProcessor() if PROCESSOR_AVAILABLE else None
        self.gcode_frame = None
        self.analysis_frame = None
        self.external_3d_frame = None
        self.serial_connection = None
        self.is_sending_gcode = False

        # Style configuration
        self.setup_styles()

        # Create GUI components
        self.create_menu()
        self.create_toolbar()
        self.create_main_layout()
        self.create_status_bar()

        # Configure grid weights
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Bind events
        self.root.bind('<Control-o>', lambda e: self.open_file())
        self.root.bind('<Control-s>', lambda e: self.export_results())
        self.root.bind('<Control-q>', lambda e: self.on_closing())
        self.root.bind('<Control-g>', lambda e: self.generate_gcode_in_view())

        # Gracefully handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start with welcome screen
        self.show_welcome_screen()

    def setup_styles(self):
        """Configure ttk styles for modern appearance."""
        self.style = ttk.Style()

        # Configure colors
        self.colors = {
            'bg': '#f0f0f0',
            'fg': '#333333',
            'accent': '#0078d4',
            'success': '#107c10',
            'warning': '#ff8c00',
            'error': '#d13438',
            'panel_bg': '#ffffff',
            'border': '#d1d1d1'
        }

        # Configure fonts
        self.fonts = {
            'title': tkFont.Font(family='Helvetica', size=16, weight='bold'),
            'heading': tkFont.Font(family='Helvetica', size=12, weight='bold'),
            'body': tkFont.Font(family='Helvetica', size=10),
            'mono': tkFont.Font(family='Courier', size=10)
        }

        # Configure styles
        self.style.configure('Title.TLabel', font=self.fonts['title'])
        self.style.configure('Heading.TLabel', font=self.fonts['heading'])
        self.style.configure('Accent.TButton', foreground=self.colors['accent'], font=self.fonts['body'])
        self.style.configure('TButton', font=self.fonts['body'])
        self.style.configure('Connect.TButton', foreground=self.colors['success'], font=self.fonts['body'])
        self.style.configure('Disconnect.TButton', foreground=self.colors['error'], font=self.fonts['body'])


    def create_menu(self):
        """Create application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open STL File...", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Export Results...", command=self.export_results, accelerator="Ctrl+S")
        file_menu.add_command(label="Export Report...", command=self.export_report)
        file_menu.add_command(label="Export G-Code...", command=self.export_gcode, accelerator="Ctrl+G")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing, accelerator="Ctrl+Q")

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="2D Wire Path", command=lambda: self.view_notebook.select(0))
        view_menu.add_command(label="3D Preview", command=lambda: self.view_notebook.select(1))
        view_menu.add_command(label="Bending Instructions", command=lambda: self.view_notebook.select(2))
        view_menu.add_command(label="Interactive 3D", command=self.show_interactive_3d_view)
        view_menu.add_command(label="Statistical Analysis", command=self.show_analysis_in_view)
        view_menu.add_command(label="G-Code", command=self.generate_gcode_in_view)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        self.tools_menu = tools_menu # Reference for state changes
        tools_menu.add_command(label="Send G-Code to Machine", command=self.send_gcode_to_esp, state='disabled')
        tools_menu.add_separator()
        tools_menu.add_command(label="Settings...", command=self.show_settings)
        tools_menu.add_command(label="Calibration...", command=self.show_calibration)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_help)
        help_menu.add_command(label="About", command=self.show_about)

    def create_toolbar(self):
        """Create application toolbar."""
        toolbar = ttk.Frame(self.root)
        toolbar.grid(row=0, column=0, sticky='ew', padx=5, pady=5)

        # Primary Actions
        ttk.Button(toolbar, text="📁 Open STL", command=self.open_file).pack(side='left', padx=2)
        ttk.Button(toolbar, text="🔄 Process", command=self.process_file).pack(side='left', padx=2)
        ttk.Separator(toolbar, orient='vertical').pack(side='left', fill='y', padx=5, pady=2)
        
        # Application Tools
        ttk.Button(toolbar, text="⚙️ Settings", command=self.show_settings).pack(side='left', padx=2)
        ttk.Button(toolbar, text="❓ Help", command=self.show_help).pack(side='left', padx=2)

    def create_main_layout(self):
        """Create main application layout."""
        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        # Left panel - File info and controls
        self.create_left_panel(main_frame)

        # Center panel - Visualization
        self.create_center_panel(main_frame)

        # Right panel - Results
        self.create_right_panel(main_frame)

    def create_left_panel(self, parent):
        """Create left control panel."""
        left_frame = ttk.LabelFrame(parent, text="Controls", padding=10)
        left_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        left_frame.grid_columnconfigure(0, weight=1)

        # --- File Info Frame ---
        file_info_frame = ttk.LabelFrame(left_frame, text="File Information", padding=10)
        file_info_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        file_info_frame.grid_columnconfigure(0, weight=1)

        self.file_label = ttk.Label(file_info_frame, text="No file loaded", foreground='gray', wraplength=250)
        self.file_label.grid(row=0, column=0, sticky='w', pady=(0, 5))
        
        # --- Parameters Frame ---
        params_frame = ttk.LabelFrame(left_frame, text="Parameters", padding=10)
        params_frame.grid(row=1, column=0, sticky='ew', pady=(0, 10))
        params_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(params_frame, text="Arch Type:").grid(row=0, column=0, sticky='w', pady=2)
        self.arch_type_var = tk.StringVar(value="auto")
        arch_combo = ttk.Combobox(params_frame, textvariable=self.arch_type_var,
                                  values=["auto", "upper", "lower"], state='readonly', width=15)
        arch_combo.grid(row=0, column=1, sticky='ew', pady=(0, 5), padx=5)

        ttk.Label(params_frame, text="Wire Size:").grid(row=1, column=0, sticky='w', pady=2)
        self.wire_size_var = tk.StringVar(value="0.018")
        wire_sizes = ["0.012", "0.014", "0.016", "0.018", "0.020",
                      "0.016x0.022", "0.018x0.025", "0.019x0.025", "0.021x0.025"]
        wire_combo = ttk.Combobox(params_frame, textvariable=self.wire_size_var,
                                  values=wire_sizes, state='readonly', width=15)
        wire_combo.grid(row=1, column=1, sticky='ew', pady=(0, 5), padx=5)
        
        # --- Actions Frame ---
        actions_frame = ttk.Frame(left_frame)
        actions_frame.grid(row=2, column=0, sticky='ew', pady=(0,10))
        actions_frame.grid_columnconfigure(0, weight=1)

        self.process_button = ttk.Button(actions_frame, text="Generate Wire",
                                          command=self.process_file, state='disabled')
        self.process_button.grid(row=0, column=0, sticky='ew', pady=5)

        # --- Export Section ---
        export_frame = ttk.LabelFrame(left_frame, text="Export Options", padding=10)
        export_frame.grid(row=3, column=0, sticky='ew', pady=(0, 10))
        export_frame.grid_columnconfigure(0, weight=1)

        self.export_btn = ttk.Button(export_frame, text="💾 Export Results",
                                       command=self.export_results, state='disabled')
        self.export_btn.grid(row=0, column=0, sticky='ew', pady=2)
        
        self.export_gcode_btn = ttk.Button(export_frame, text="🔧 Export G-Code File",
                                           command=self.export_gcode, state='disabled')
        self.export_gcode_btn.grid(row=1, column=0, sticky='ew', pady=2)
        
        # --- Machine Connection Frame ---
        self.create_connection_panel(left_frame)
        left_frame.grid_rowconfigure(4, weight=1) # Allow connection panel to take space

        # --- Progress Bar ---
        self.progress_bar = ttk.Progressbar(left_frame, variable=tk.DoubleVar(), mode='indeterminate')
        self.progress_bar.grid(row=5, column=0, sticky='ew', pady=(10, 0))
        self.progress_bar.grid_remove()

    def create_connection_panel(self, parent):
        """Creates the ESP32 connection panel in the left sidebar."""
        conn_frame = ttk.LabelFrame(parent, text="Machine Connection", padding=10)
        conn_frame.grid(row=4, column=0, sticky='nsew', pady=(0, 10))
        conn_frame.grid_columnconfigure(0, weight=1)

        if not PYSERIAL_AVAILABLE:
            label = ttk.Label(conn_frame, text="PySerial not installed.\nConnection disabled.",
                              font=self.fonts['body'], foreground=self.colors['warning'])
            label.pack(pady=10)
            return

        # Port selection
        port_frame = ttk.Frame(conn_frame)
        port_frame.grid(row=0, column=0, sticky='ew', pady=5)
        port_frame.grid_columnconfigure(0, weight=1)

        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(port_frame, textvariable=self.port_var, state='readonly', width=15)
        self.port_combo.grid(row=0, column=0, sticky='ew', padx=(0, 5))

        self.refresh_ports_btn = ttk.Button(port_frame, text="🔄", width=3, command=self.update_serial_ports)
        self.refresh_ports_btn.grid(row=0, column=1, sticky='e')

        # Connection buttons
        button_frame = ttk.Frame(conn_frame)
        button_frame.grid(row=1, column=0, sticky='ew', pady=5)
        button_frame.grid_columnconfigure(0, weight=1)

        self.connect_btn = ttk.Button(button_frame, text="🔌 Connect", style='Connect.TButton', command=self.toggle_connection)
        self.connect_btn.grid(row=0, column=0, sticky='ew')

        self.send_gcode_btn = ttk.Button(conn_frame, text="➤ Send G-Code to Machine", command=self.send_gcode_to_esp, state='disabled')
        self.send_gcode_btn.grid(row=2, column=0, sticky='ew', pady=5)
        
        self.update_serial_ports()


    def create_center_panel(self, parent):
        """Create center visualization panel."""
        center_frame = ttk.LabelFrame(parent, text="Visualization", padding=10)
        center_frame.grid(row=0, column=1, sticky='nsew', padx=5)
        center_frame.grid_columnconfigure(0, weight=1)
        center_frame.grid_rowconfigure(1, weight=1)

        # Button frame at the top with all view buttons
        button_frame = ttk.Frame(center_frame)
        button_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        button_frame.grid_columnconfigure((0, 7), weight=1) # Center the buttons
        
        ttk.Button(button_frame, text="📈 2D Path", command=lambda: self.view_notebook.select(0)).grid(row=0, column=1, padx=2)
        ttk.Button(button_frame, text="🎲 3D Preview", command=lambda: self.view_notebook.select(1)).grid(row=0, column=2, padx=2)
        ttk.Button(button_frame, text="📋 Instructions", command=lambda: self.view_notebook.select(2)).grid(row=0, column=3, padx=2)
        
        self.interactive_3d_btn = ttk.Button(button_frame, text="👁️ Interactive 3D", command=self.show_interactive_3d_view, state='disabled')
        self.interactive_3d_btn.grid(row=0, column=4, padx=2)
        
        self.analysis_btn = ttk.Button(button_frame, text="📊 Analysis", command=self.show_analysis_in_view, state='disabled')
        self.analysis_btn.grid(row=0, column=5, padx=2)
        self.gcode_btn = ttk.Button(button_frame, text="📜 G-Code", command=self.generate_gcode_in_view, state='disabled')
        self.gcode_btn.grid(row=0, column=6, padx=2)

        # Notebook for different views (without visible tabs)
        self.view_notebook = ttk.Notebook(center_frame)
        self.view_notebook.grid(row=1, column=0, sticky='nsew')
        style = ttk.Style()
        style.layout('TNotebook.Tab', [])  # Remove tab appearance

        self.view_2d_frame = ttk.Frame(self.view_notebook)
        self.view_notebook.add(self.view_2d_frame, text="2D Wire Path")
        self.view_3d_frame = ttk.Frame(self.view_notebook)
        self.view_notebook.add(self.view_3d_frame, text="3D Preview")
        self.instructions_frame = ttk.Frame(self.view_notebook)
        self.view_notebook.add(self.instructions_frame, text="Bending Instructions")

    def create_right_panel(self, parent):
        """Create right results panel."""
        right_frame = ttk.LabelFrame(parent, text="Analysis & Results", padding=10)
        right_frame.grid(row=0, column=2, sticky='nsew', padx=(5, 0))
        right_frame.grid_columnconfigure(0, weight=1)
        right_frame.grid_rowconfigure(0, weight=1)

        # Results text widget
        self.results_text = tk.Text(right_frame, width=40, height=20, wrap='word',
                                    font=self.fonts['mono'], relief='flat')
        self.results_text.grid(row=0, column=0, sticky='nsew')
        scrollbar = ttk.Scrollbar(right_frame, orient='vertical', command=self.results_text.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.results_text.config(yscrollcommand=scrollbar.set)
        
        # Configure text tags
        self.results_text.tag_configure('heading', font=self.fonts['heading'], foreground=self.colors['accent'])
        self.results_text.tag_configure('success', foreground=self.colors['success'])
        self.results_text.tag_configure('warning', foreground=self.colors['warning'])
        self.results_text.tag_configure('error', foreground=self.colors['error'])

    def create_status_bar(self):
        """Create status bar."""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.grid(row=2, column=0, sticky='ew', padx=5, pady=(5,0))
        self.status_label = ttk.Label(self.status_bar, text="Ready", relief='sunken', anchor='w')
        self.status_label.pack(side='left', fill='x', expand=True, padx=2, pady=2)
        self.coords_label = ttk.Label(self.status_bar, text="", relief='sunken', width=30, anchor='e')
        self.coords_label.pack(side='right', padx=2, pady=2)
    
    # --- File and Processing Logic ---
    def open_file(self):
        """Open STL file dialog."""
        if self.is_sending_gcode:
            messagebox.showwarning("Busy", "Cannot open file while sending G-Code.")
            return
            
        filename = filedialog.askopenfilename(
            title="Select STL File",
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")]
        )

        if filename:
            self.current_file = filename
            self.file_label.config(text=os.path.basename(filename), foreground=self.colors['fg'])
            self.process_button.config(state='normal')
            self.update_status(f"Loaded: {os.path.basename(filename)}")
            self.clear_results()
            self.show_file_info()
    
    def show_file_info(self):
        """Display file information in results panel."""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "FILE INFORMATION\n", 'heading')
        self.results_text.insert(tk.END, "="*30 + "\n\n")
        
        if self.current_file:
            file_stat = os.stat(self.current_file)
            file_size_mb = file_stat.st_size / (1024 * 1024)
            
            self.results_text.insert(tk.END, f"Name: {os.path.basename(self.current_file)}\n")
            self.results_text.insert(tk.END, f"Size: {file_size_mb:.2f} MB\n")
            
            # Detect arch type from filename
            if 'lower' in self.current_file.lower():
                self.arch_type_var.set("lower")
            elif 'upper' in self.current_file.lower():
                self.arch_type_var.set("upper")
            else:
                self.arch_type_var.set("auto")
            
            self.results_text.insert(tk.END, f"\nDetected Arch: {self.arch_type_var.get().title()}\n", 'success')
            self.results_text.insert(tk.END, "\nReady to process...\n")
    
    def process_file(self):
        """Process the loaded STL file."""
        if not self.current_file or self.is_sending_gcode:
            if self.is_sending_gcode:
                messagebox.showwarning("Busy", "Cannot process file while sending G-Code.")
            else:
                messagebox.showerror("Error", "Please load an STL file first")
            return
        
        if not PROCESSOR_AVAILABLE:
            messagebox.showerror("Error", "Processor not available. Please ensure orthodontic_processor.py is in the same directory.")
            return
        
        self.process_button.config(state='disabled')
        self.progress_bar.grid()
        self.progress_bar.start(10)
        self.clear_results()
        
        thread = threading.Thread(target=self._process_file_thread)
        thread.daemon = True
        thread.start()
    
    def _process_file_thread(self):
        """Process file in separate thread."""
        try:
            self.update_status("Processing STL file...")
            result = self.processor.generate_wire(
                file_path=self.current_file,
                arch_type=self.arch_type_var.get(),
                wire_size=self.wire_size_var.get()
            )
            self.processing_result = result
            self.root.after(0, self._process_complete, True, result)
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, self._process_complete, False, error_msg)

    def _process_complete(self, success, result):
        """Handle processing completion."""
        self.progress_bar.stop()
        self.progress_bar.grid_remove()
        self.process_button.config(state='normal')

        if success:
            self.update_status("Processing complete!")
            self.display_results(result)
            self.visualize_results(result)

            # Enable result-dependent buttons
            self.interactive_3d_btn.config(state='normal')
            self.analysis_btn.config(state='normal')
            self.gcode_btn.config(state='normal')
            self.export_btn.config(state='normal')
            self.export_gcode_btn.config(state='normal')
            if self.serial_connection and self.serial_connection.is_open:
                self.send_gcode_btn.config(state='normal')
                self.tools_menu.entryconfig("Send G-Code to Machine", state='normal')

            messagebox.showinfo("Success", "Wire generation completed successfully!")
        else:
            self.update_status("Processing failed")
            self.results_text.insert(tk.END, f"ERROR:\n{result}", 'error')
            messagebox.showerror("Processing Error", f"Failed to process file:\n{result}")

    # --- Visualization Logic ---
    def show_interactive_3d_view(self):
        """Show a dedicated tab for launching the interactive 3D viewer."""
        if not self.processing_result:
            messagebox.showinfo("Info", "Please process a file first")
            return
        
        # Create or switch to 3D external view tab
        if not self.external_3d_frame or not self.external_3d_frame.winfo_exists():
            self.external_3d_frame = ttk.Frame(self.view_notebook)
            self.view_notebook.add(self.external_3d_frame, text="Interactive 3D")
        
        # Clear existing content
        for widget in self.external_3d_frame.winfo_children():
            widget.destroy()
        
        # Create content frame
        content_frame = ttk.Frame(self.external_3d_frame)
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(content_frame, text="Interactive 3D Visualization", 
                                font=self.fonts['heading'])
        title_label.pack(pady=(0, 10))
        
        if OPEN3D_AVAILABLE:
            # Instructions
            instructions = ttk.Label(content_frame, 
                text="Click the button below to open an interactive 3D view in a separate window.\n"
                     "You can rotate, zoom, and pan the model using your mouse.",
                font=self.fonts['body'], justify='center')
            instructions.pack(pady=10)
            
            # Launch button
            launch_btn = ttk.Button(content_frame, text="🚀 Launch 3D Viewer", 
                                  command=self.launch_3d_viewer,
                                  style='Accent.TButton')
            launch_btn.pack(pady=10)
            
            # Controls info
            controls_frame = ttk.LabelFrame(content_frame, text="3D Viewer Controls", padding=10)
            controls_frame.pack(pady=10, fill='x')
            
            controls = [
                ("Left Mouse", "Rotate view"),
                ("Right Mouse", "Pan view"),
                ("Scroll Wheel", "Zoom in/out"),
                ("Q", "Close viewer")
            ]
            
            for key, action in controls:
                control_frame = ttk.Frame(controls_frame)
                control_frame.pack(fill='x', pady=2)
                ttk.Label(control_frame, text=f"{key}:", font=self.fonts['body'], 
                          width=15).pack(side='left')
                ttk.Label(control_frame, text=action, font=self.fonts['body']).pack(side='left')
        else:
            # Open3D not available message
            msg = ttk.Label(content_frame, 
                text="Open3D is not installed.\n\n"
                     "To enable interactive 3D visualization, install Open3D:\n"
                     "pip install open3d",
                font=self.fonts['body'], justify='center')
            msg.pack(pady=20)
        
        # Switch to this tab
        self.view_notebook.select(self.external_3d_frame)
        
    def launch_3d_viewer(self):
        """Launch the Open3D viewer in the main thread (will block GUI)."""
        if not OPEN3D_AVAILABLE:
            messagebox.showerror("Error", "Open3D is not installed.")
            return
        
        raw_result = self.processing_result.get('raw_result', {})
        if not raw_result or not raw_result.get('mesh'):
            messagebox.showwarning("Warning", "No 3D data available in the processing result.")
            return
        
        geometries = []
        if raw_result.get('mesh'):
            geometries.append(raw_result['mesh'])
        if raw_result.get('wire_mesh'):
            geometries.append(raw_result['wire_mesh'])
        if raw_result.get('bracket_markers'):
            geometries.append(raw_result['bracket_markers'])
            
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15.0, origin=[0, 0, 0])
        geometries.append(coord_frame)
        
        self.update_status("Opening interactive 3D viewer... Close viewer to continue.")
        o3d.visualization.draw_geometries(geometries, window_name="Interactive 3D Wire Visualization")
        self.update_status("Ready")

    def show_analysis_in_view(self):
        """Show analysis in the view notebook."""
        if not self.processing_result:
            messagebox.showinfo("Info", "Please process a file first")
            return
        
        if self.analysis_frame and self.analysis_frame.winfo_exists():
            self.view_notebook.select(self.analysis_frame)
            return

        self.analysis_frame = ttk.Frame(self.view_notebook)
        self.view_notebook.add(self.analysis_frame, text="Statistical Analysis")
        
        if MATPLOTLIB_AVAILABLE:
            fig = Figure(figsize=(10, 6), dpi=100)
            
            # Tooth type distribution
            ax1 = fig.add_subplot(221)
            tooth_types = ['Incisors', 'Canines', 'Posterior']
            tooth_counts = [
                self.processing_result['incisors'],
                self.processing_result['canines'],
                self.processing_result['posterior']
            ]
            ax1.bar(tooth_types, tooth_counts, color=['green', 'gold', 'blue'])
            ax1.set_title('Tooth Type Distribution')
            ax1.set_ylabel('Count')
            
            # Expected vs Actual
            ax2 = fig.add_subplot(222)
            categories = ['Incisors', 'Canines']
            expected = [4, 2]
            actual = [self.processing_result['incisors'], self.processing_result['canines']]
            x = np.arange(len(categories))
            width = 0.35
            ax2.bar(x - width/2, expected, width, label='Expected', alpha=0.8)
            ax2.bar(x + width/2, actual, width, label='Actual', alpha=0.8)
            ax2.set_title('Classification Accuracy')
            ax2.set_xticks(x)
            ax2.set_xticklabels(categories)
            ax2.legend()
            
            # Wire segment lengths
            ax3 = fig.add_subplot(223)
            positions = self.processing_result['bracket_positions']
            if len(positions) > 1:
                segment_lengths = []
                for i in range(len(positions)-1):
                    p1 = np.array(positions[i]['position'])
                    p2 = np.array(positions[i+1]['position'])
                    segment_lengths.append(np.linalg.norm(p2 - p1))
                
                ax3.hist(segment_lengths, bins=10, edgecolor='black', alpha=0.7)
                ax3.set_title('Wire Segment Length Distribution')
                ax3.set_xlabel('Length (mm)')
                ax3.set_ylabel('Frequency')
            
            # Bend angles
            ax4 = fig.add_subplot(224)
            bend_angles = [inst['bend_deg'] for inst in self.processing_result['bending_instructions']]
            if bend_angles:
                ax4.plot(range(len(bend_angles)), bend_angles, 'bo-')
                ax4.set_title('Bend Angles Along Wire')
                ax4.set_xlabel('Bend Point')
                ax4.set_ylabel('Angle (degrees)')
                ax4.grid(True, alpha=0.3)
            
            fig.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=self.analysis_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            toolbar = NavigationToolbar2Tk(canvas, self.analysis_frame)
            toolbar.update()
        else:
            ttk.Label(self.analysis_frame, text="Matplotlib not installed.", font=self.fonts['body']).pack(expand=True)
        
        self.view_notebook.select(self.analysis_frame)

    # --- G-Code Logic ---
    def generate_gcode_in_view(self):
        """Generate G-code and display in view notebook."""
        if not self.processing_result:
            messagebox.showinfo("Info", "Please process a file first to generate G-code.")
            return
            
        if self.gcode_frame and self.gcode_frame.winfo_exists():
            self.view_notebook.select(self.gcode_frame)
            return

        self.gcode_frame = ttk.Frame(self.view_notebook)
        self.view_notebook.add(self.gcode_frame, text="G-Code")
        
        toolbar = ttk.Frame(self.gcode_frame)
        toolbar.pack(fill='x', padx=5, pady=5)
        ttk.Label(toolbar, text="G-Code for ESP32/Arduino Wire Bender", font=self.fonts['heading']).pack(side='left', padx=10)
        ttk.Button(toolbar, text="📋 Copy to Clipboard", command=self.copy_gcode_to_clipboard).pack(side='right', padx=2)
        ttk.Button(toolbar, text="💾 Save G-Code", command=self.export_gcode).pack(side='right', padx=2)
        
        text_frame = ttk.Frame(self.gcode_frame)
        text_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.gcode_text_widget = tk.Text(text_frame, wrap='none', font=self.fonts['mono'], relief='flat', bg='#f8f8f8')
        self.gcode_text_widget.pack(side='left', fill='both', expand=True)
        v_scrollbar = ttk.Scrollbar(text_frame, command=self.gcode_text_widget.yview)
        v_scrollbar.pack(side='right', fill='y')
        h_scrollbar = ttk.Scrollbar(self.gcode_frame, orient='horizontal', command=self.gcode_text_widget.xview)
        h_scrollbar.pack(fill='x', padx=5)
        self.gcode_text_widget.config(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        gcode_content = self.generate_gcode_content()
        self.gcode_text_widget.insert(tk.END, gcode_content)
        
        self.gcode_text_widget.tag_configure('comment', foreground='#008000')
        self.gcode_text_widget.tag_configure('command', foreground='#0000FF', font=tkFont.Font(family='Courier', size=10, weight='bold'))
        self.gcode_text_widget.tag_configure('parameter', foreground='#FF4500')
        self.highlight_gcode()
        
        self.view_notebook.select(self.gcode_frame)
        self.update_status("G-Code generated successfully")

    def generate_gcode_content(self):
        """Generate G-code content for ESP32/Arduino wire bender."""
        gcode = []
        gcode.append("; =====================================")
        gcode.append("; Orthodontic Wire Bending G-Code")
        gcode.append(f"; Generated by: Dental Archwire Design System v3.2")
        gcode.append(f"; Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        gcode.append(f"; File: {os.path.basename(self.current_file if self.current_file else 'N/A')}")
        gcode.append(f"; Arch Type: {self.arch_type_var.get().title()}")
        gcode.append(f"; Wire Size: {self.wire_size_var.get()}")
        if self.processing_result:
            gcode.append(f"; Total Length: {self.processing_result['wire_length']:.2f} mm")
        gcode.append("; =====================================")
        gcode.append("")
        gcode.append("; === INITIALIZATION ===")
        gcode.append("G90 ; Absolute positioning")
        gcode.append("G21 ; Units in millimeters")
        gcode.append("G92 X0 Y0 Z0 ; Set current position as origin")
        gcode.append("M17 ; Enable steppers")
        gcode.append("G4 P500 ; Pause 500ms for stabilization")
        gcode.append("")
        gcode.append("; === WIRE BENDING SEQUENCE ===")
        
        cumulative_feed = 0.0
        if self.processing_result:
            for i, instruction in enumerate(self.processing_result['bending_instructions']):
                gcode.append(f"; ----- Step {instruction['step']}: {instruction['location']} -----")
                if instruction['feed_mm'] > 0:
                    cumulative_feed += instruction['feed_mm']
                    gcode.append(f"G1 X{cumulative_feed:.3f} F300 ; Feed wire")
                if abs(instruction['bend_deg']) > 0.5:
                    bend_angle = instruction['bend_deg']
                    gcode.append(f"G1 Y{bend_angle:.2f} F60 ; Bend {abs(bend_angle):.1f}°")
                    gcode.append("G1 Y0 F120 ; Return bender")
                gcode.append("")

        gcode.append("; === COMPLETION SEQUENCE ===")
        gcode.append("M84 ; Disable steppers")
        gcode.append("M0 ; Program stop")
        gcode.append("; === END OF PROGRAM ===")
        return '\n'.join(gcode)

    def highlight_gcode(self):
        """Apply syntax highlighting to G-code."""
        content = self.gcode_text_widget.get('1.0', 'end-1c')
        for i, line in enumerate(content.split('\n'), 1):
            if line.strip().startswith(';'):
                self.gcode_text_widget.tag_add('comment', f'{i}.0', f'{i}.end')
            elif line.strip():
                parts = line.split(';')[0].split()
                if parts:
                    if parts[0].startswith(('G', 'M')):
                        cmd_end = len(parts[0])
                        self.gcode_text_widget.tag_add('command', f'{i}.0', f'{i}.{cmd_end}')
                    for part in parts[1:]:
                        if any(part.startswith(x) for x in ['X', 'Y', 'Z', 'F', 'P']):
                            start_idx = line.find(part)
                            end_idx = start_idx + len(part)
                            self.gcode_text_widget.tag_add('parameter', f'{i}.{start_idx}', f'{i}.{end_idx}')
        
    def copy_gcode_to_clipboard(self):
        """Copy G-code to clipboard."""
        if hasattr(self, 'gcode_text_widget'):
            gcode_content = self.gcode_text_widget.get('1.0', 'end-1c')
            self.root.clipboard_clear()
            self.root.clipboard_append(gcode_content)
            self.update_status("G-code copied to clipboard")
            messagebox.showinfo("Success", "G-code copied to clipboard!")

    # --- Serial Connection Logic ---
    def update_serial_ports(self):
        """Update the list of available serial ports."""
        if not PYSERIAL_AVAILABLE: return
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo['values'] = ports
        if ports:
            self.port_var.set(ports[0])
        else:
            self.port_var.set("")

    def toggle_connection(self):
        """Connect or disconnect from the selected serial port."""
        if self.serial_connection and self.serial_connection.is_open:
            self.disconnect_serial()
        else:
            self.connect_serial()
            
    def connect_serial(self):
        """Connect to the selected serial port."""
        if not PYSERIAL_AVAILABLE: return
        port = self.port_var.get()
        if not port:
            messagebox.showerror("Error", "No serial port selected.")
            return

        try:
            self.update_status(f"Connecting to {port}...")
            self.serial_connection = serial.Serial(port, 115200, timeout=1)
            time.sleep(2)
            self.serial_connection.write(b"\r\n\r\n")
            self.serial_connection.flushInput()
            self.connect_btn.config(text="🔌 Disconnect", style='Disconnect.TButton')
            self.update_status(f"Connected to {port}")
            self.refresh_ports_btn.config(state='disabled')
            self.port_combo.config(state='disabled')
            if self.processing_result:
                self.send_gcode_btn.config(state='normal')
                self.tools_menu.entryconfig("Send G-Code to Machine", state='normal')

        except serial.SerialException as e:
            messagebox.showerror("Connection Error", f"Failed to connect to {port}.\nError: {e}")
            self.serial_connection = None
            self.update_status("Connection failed")

    def disconnect_serial(self):
        """Disconnect from the current serial port."""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        self.serial_connection = None
        self.connect_btn.config(text="🔌 Connect", style='Connect.TButton')
        self.send_gcode_btn.config(state='disabled')
        self.tools_menu.entryconfig("Send G-Code to Machine", state='disabled')
        self.refresh_ports_btn.config(state='normal')
        self.port_combo.config(state='normal')
        self.update_status("Disconnected")
        
    def send_gcode_to_esp(self):
        """Initiate sending G-code to the connected machine."""
        if self.is_sending_gcode:
            messagebox.showwarning("Busy", "Already sending G-Code.")
            return
            
        if not (self.serial_connection and self.serial_connection.is_open):
            messagebox.showwarning("Not Connected", "Please connect to the machine first.")
            return
            
        if not self.processing_result or not hasattr(self, 'gcode_text_widget'):
            messagebox.showwarning("No G-Code", "Please generate G-Code first.")
            return

        if messagebox.askyesno("Confirm Send", "Are you sure you want to send the G-Code to the machine?"):
            self.is_sending_gcode = True
            self.send_gcode_btn.config(state='disabled')
            self.process_button.config(state='disabled')
            self.progress_bar.config(mode='determinate')
            self.progress_bar.grid()
            
            thread = threading.Thread(target=self._send_gcode_thread)
            thread.daemon = True
            thread.start()

    def _send_gcode_thread(self):
        """The actual G-code sending logic in a separate thread."""
        try:
            gcode_content = self.gcode_text_widget.get('1.0', 'end-1c')
            lines = [line.strip() for line in gcode_content.split('\n') if line.strip() and not line.strip().startswith(';')]
            total_lines = len(lines)
            self.serial_connection.flushInput()

            for i, line in enumerate(lines):
                self.root.after(0, self.update_status, f"Sending ({i+1}/{total_lines}): {line}")
                self.root.after(0, self.progress_bar.config, {'value': (i + 1) / total_lines * 100})
                
                self.serial_connection.write((line + '\n').encode('utf-8'))
                
                response = self.serial_connection.readline().decode('utf-8').strip()
                if 'ok' not in response:
                    self.root.after(0, messagebox.showerror, "Machine Error", f"Error on line {i+1}:\n{line}\n\nResponse: {response}")
                    raise serial.SerialException("Machine reported an error.")

            self.root.after(0, self.update_status, "G-Code sending complete.")
            self.root.after(0, messagebox.showinfo, "Success", "G-Code sent successfully.")

        except Exception as e:
            self.root.after(0, self.update_status, f"Error sending G-Code: {e}")
        finally:
            self.root.after(0, self.progress_bar.grid_remove)
            self.root.after(0, self.progress_bar.config, {'mode': 'indeterminate'})
            self.root.after(0, self.send_gcode_btn.config, {'state': 'normal'})
            self.root.after(0, self.process_button.config, {'state': 'normal'})
            self.is_sending_gcode = False

    # --- GUI Management & Other Methods ---
    def clear_results(self):
        """Clear all results displays."""
        self.results_text.delete(1.0, tk.END)
        self.processing_result = None
        
        for frame in [self.view_2d_frame, self.view_3d_frame, self.instructions_frame]:
            for widget in frame.winfo_children():
                widget.destroy()

        if self.gcode_frame and self.gcode_frame.winfo_exists(): self.view_notebook.forget(self.gcode_frame)
        if self.analysis_frame and self.analysis_frame.winfo_exists(): self.view_notebook.forget(self.analysis_frame)
        if self.external_3d_frame and self.external_3d_frame.winfo_exists(): self.view_notebook.forget(self.external_3d_frame)
        
        self.gcode_frame = None
        self.analysis_frame = None
        self.external_3d_frame = None

        self.interactive_3d_btn.config(state='disabled')
        self.analysis_btn.config(state='disabled')
        self.gcode_btn.config(state='disabled')
        self.export_btn.config(state='disabled')
        self.export_gcode_btn.config(state='disabled')
        if PYSERIAL_AVAILABLE:
            self.send_gcode_btn.config(state='disabled')
            self.tools_menu.entryconfig("Send G-Code to Machine", state='disabled')
        
        self.show_welcome_screen()

    def show_welcome_screen(self):
        """Show welcome screen in center panel."""
        if MATPLOTLIB_AVAILABLE:
            fig = Figure(figsize=(8, 6), dpi=100, facecolor=self.colors['panel_bg'])
            ax = fig.add_subplot(111)
            ax.set_facecolor(self.colors['panel_bg'])
            ax.text(0.5, 0.6, 'Dental Archwire Design System', ha='center', va='center', fontsize=24, weight='bold', color=self.colors['accent'])
            ax.text(0.5, 0.5, 'Version 3.2', ha='center', va='center', fontsize=16, color=self.colors['fg'])
            ax.text(0.5, 0.4, 'Lingual Wire & G-Code Generation', ha='center', va='center', fontsize=14, color=self.colors['fg'])
            ax.text(0.5, 0.2, 'Load an STL file to begin', ha='center', va='center', fontsize=12, style='italic', color='gray')
            ax.axis('off')
            canvas = FigureCanvasTkAgg(fig, master=self.view_2d_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            self.view_notebook.select(0)

    def display_results(self, result):
        """Display processing results in results panel."""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "PROCESSING RESULTS\n", 'heading')
        self.results_text.insert(tk.END, "="*30 + "\n\n")
        self.results_text.insert(tk.END, "Tooth Classification:\n", 'heading')
        self.results_text.insert(tk.END, f"Total Teeth: {result['teeth_detected']}\n")
        self.results_text.insert(tk.END, f"  • Incisors: {result['incisors']}\n", 'success' if result['incisors'] == 4 else 'warning')
        self.results_text.insert(tk.END, f"  • Canines: {result['canines']}\n", 'success' if result['canines'] == 2 else 'warning')
        self.results_text.insert(tk.END, f"  • Posterior: {result['posterior']}\n")
        self.results_text.insert(tk.END, f"\nClassification Accuracy: {result['classification_accuracy']:.1f}%\n", 'success' if result['classification_accuracy'] >= 90 else 'warning')
        self.results_text.insert(tk.END, "\nWire Specifications:\n", 'heading')
        self.results_text.insert(tk.END, f"Total Length: {result['wire_length']:.1f} mm ({result['wire_length']/25.4:.2f} in)\n")
        self.results_text.insert(tk.END, f"\nProcessing Time: {result['processing_time']:.2f} seconds\n")
        self.results_text.insert(tk.END, "\nSummary:\n", 'heading')
        if result['classification_accuracy'] >= 90: self.results_text.insert(tk.END, "✓ Excellent classification\n", 'success')
        else: self.results_text.insert(tk.END, "⚠ Classification needs review\n", 'warning')
        self.results_text.insert(tk.END, "✓ Lingual wire generated\n", 'success')
        self.results_text.insert(tk.END, f"✓ {len(result['bending_instructions'])} bending points\n", 'success')

    def visualize_results(self, result):
        """Visualize processing results."""
        for frame in [self.view_2d_frame, self.view_3d_frame, self.instructions_frame]:
            for widget in frame.winfo_children():
                widget.destroy()
        
        if MATPLOTLIB_AVAILABLE:
            self.create_2d_preview(result)
            self.create_3d_preview(result)
        
        self.show_bending_instructions_tab(result)
        self.view_notebook.select(0) 

    def create_2d_preview(self, result):
        """Create 2D wire path visualization in the GUI tab."""
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        positions = result.get('bracket_positions', [])
        if positions:
            incisors = [p for p in positions if p['type'] == 'incisor']
            canines = [p for p in positions if p['type'] == 'canine']
            posterior = [p for p in positions if p['type'] == 'posterior']
            
            if incisors: ax.scatter([p['position'][0] for p in incisors], [p['position'][1] for p in incisors], c='green', s=100, label='Incisors', marker='s')
            if canines: ax.scatter([p['position'][0] for p in canines], [p['position'][1] for p in canines], c='gold', s=100, label='Canines', marker='^')
            if posterior: ax.scatter([p['position'][0] for p in posterior], [p['position'][1] for p in posterior], c='blue', s=100, label='Posterior', marker='o')
            
            all_x = [p['position'][0] for p in positions]
            all_y = [p['position'][1] for p in positions]
            if len(all_x) > 1:
                angles = np.arctan2(all_y, all_x)
                sorted_indices = np.argsort(angles)
                sorted_x = [all_x[i] for i in sorted_indices]
                sorted_y = [all_y[i] for i in sorted_indices]
                sorted_x.append(sorted_x[0])
                sorted_y.append(sorted_y[0])
                ax.plot(sorted_x, sorted_y, 'k-', linewidth=2, alpha=0.7, label='Wire Path')
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title('Lingual Wire Path - Top View')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
            
        canvas = FigureCanvasTkAgg(fig, master=self.view_2d_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        toolbar = NavigationToolbar2Tk(canvas, self.view_2d_frame)
        toolbar.update()

    def create_3d_preview(self, result):
        """Create 3D preview with interactive launch button."""
        if not MATPLOTLIB_AVAILABLE:
            ttk.Label(self.view_3d_frame, text="Matplotlib not installed.", font=self.fonts['body']).pack(expand=True)
            return
        
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        positions = result.get('bracket_positions', [])
        if positions:
            incisors = [p for p in positions if p['type'] == 'incisor']
            canines = [p for p in positions if p['type'] == 'canine']
            posterior = [p for p in positions if p['type'] == 'posterior']

            if incisors: ax.scatter([p['position'][0] for p in incisors], [p['position'][1] for p in incisors], [p['position'][2] for p in incisors], c='green', s=100, label='Incisors', marker='s')
            if canines: ax.scatter([p['position'][0] for p in canines], [p['position'][1] for p in canines], [p['position'][2] for p in canines], c='gold', s=100, label='Canines', marker='^')
            if posterior: ax.scatter([p['position'][0] for p in posterior], [p['position'][1] for p in posterior], [p['position'][2] for p in posterior], c='blue', s=100, label='Posterior', marker='o')

            all_x = [p['position'][0] for p in positions]
            all_y = [p['position'][1] for p in positions]
            all_z = [p['position'][2] for p in positions]
            if len(all_x) > 1:
                center_x, center_y = np.mean(all_x), np.mean(all_y)
                angles = np.arctan2(np.array(all_y) - center_y, np.array(all_x) - center_x)
                sorted_indices = np.argsort(angles)
                sorted_x = [all_x[i] for i in sorted_indices]
                sorted_y = [all_y[i] for i in sorted_indices]
                sorted_z = [all_z[i] for i in sorted_indices]
                sorted_x.append(sorted_x[0]); sorted_y.append(sorted_y[0]); sorted_z.append(sorted_z[0])
                ax.plot(sorted_x, sorted_y, sorted_z, 'k-', linewidth=3, alpha=0.8, label='Wire Path')
        
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.set_title('3D Lingual Wire Preview')
        ax.legend()
        ax.view_init(elev=20, azim=45)
        
        canvas = FigureCanvasTkAgg(fig, master=self.view_3d_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        bottom_frame = ttk.Frame(self.view_3d_frame)
        bottom_frame.pack(fill='x', side='bottom')
        toolbar = NavigationToolbar2Tk(canvas, bottom_frame)
        toolbar.update()

    def show_bending_instructions_tab(self, result):
        """Display bending instructions in dedicated tab."""
        for widget in self.instructions_frame.winfo_children():
            widget.destroy()
        
        toolbar = ttk.Frame(self.instructions_frame)
        toolbar.pack(fill='x', padx=5, pady=5)
        ttk.Label(toolbar, text="Wire Bending Instructions", font=self.fonts['heading']).pack(side='left', padx=10)
        ttk.Button(toolbar, text="Export CSV", command=self.export_instructions_csv).pack(side='right', padx=5)
        
        columns = ('Step', 'Feed (mm)', 'Bend (°)', 'Location', 'Notes')
        tree = ttk.Treeview(self.instructions_frame, columns=columns, show='headings', height=15)
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100 if col != 'Notes' else 300, anchor='center')
        
        for instruction in result.get('bending_instructions', []):
            tree.insert('', 'end', values=[instruction.get(k, '') for k in ['step', 'feed_mm', 'bend_deg', 'location', 'notes']])
        
        scrollbar = ttk.Scrollbar(self.instructions_frame, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side='left', fill='both', expand=True, padx=(5,0), pady=5)
        scrollbar.pack(side='right', fill='y', pady=5)
    
    def on_closing(self):
        """Handle window closing event."""
        if self.is_sending_gcode:
            if messagebox.askokcancel("Quit", "G-Code is currently being sent. Are you sure you want to quit?"):
                self.disconnect_serial()
                self.root.destroy()
        else:
            if self.serial_connection and self.serial_connection.is_open:
                self.disconnect_serial()
            self.root.destroy()
            
    def export_results(self):
        """Export processing results with multiple format options."""
        if not self.processing_result:
            messagebox.showinfo("Info", "No results to export.")
            return
        
        export_window = tk.Toplevel(self.root)
        export_window.title("Export Results")
        export_window.geometry("400x350")
        export_window.transient(self.root)
        export_window.grab_set()
        
        export_window.update_idletasks()
        x = (export_window.winfo_screenwidth() // 2) - (export_window.winfo_width() // 2)
        y = (export_window.winfo_screenheight() // 2) - (export_window.winfo_height() // 2)
        export_window.geometry(f'+{x}+{y}')
        
        ttk.Label(export_window, text="Select Export Format:", font=self.fonts['heading']).pack(pady=20)
        
        format_var = tk.StringVar(value="json")
        formats = [
            ("JSON", "json", "Complete data in JSON format"),
            ("CSV", "csv", "Bending instructions in CSV format"),
            ("G-Code", "gcode", "G-code for ESP32/Arduino"),
            ("Report", "txt", "Comprehensive text report")
        ]
        
        for name, value, desc in formats:
            frame = ttk.Frame(export_window)
            frame.pack(fill='x', padx=30, pady=5)
            ttk.Radiobutton(frame, text=name, variable=format_var, value=value).pack(side='left')
            ttk.Label(frame, text=f" - {desc}", font=self.fonts['body']).pack(side='left', padx=(10, 0))
        
        def do_export():
            format_type = format_var.get()
            export_window.destroy()
            if format_type == "json": self.export_json()
            elif format_type == "csv": self.export_instructions_csv()
            elif format_type == "gcode": self.export_gcode()
            elif format_type == "txt": self.export_report()
        
        button_frame = ttk.Frame(export_window)
        button_frame.pack(pady=20)
        ttk.Button(button_frame, text="Export", command=do_export, style='Accent.TButton').pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=export_window.destroy).pack(side='left', padx=5)

    def export_json(self):
        """Export results as JSON."""
        filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")], title="Export as JSON")
        if filename and self.processor:
            self.processor.export_to_json(self.processing_result, filename)
            messagebox.showinfo("Success", f"Results exported to {filename}")

    def export_instructions_csv(self):
        """Export bending instructions to CSV."""
        filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Export as CSV")
        if filename and self.processor:
            self.processor.export_to_csv(self.processing_result, filename)
            messagebox.showinfo("Success", f"Instructions exported to {filename}")

    def export_gcode(self):
        """Export G-code file."""
        filename = filedialog.asksaveasfilename(defaultextension=".gcode", filetypes=[("G-code files", "*.gcode")], title="Export G-Code")
        if filename:
            with open(filename, 'w') as f:
                f.write(self.generate_gcode_content())
            messagebox.showinfo("Success", f"G-code exported to {filename}")

    def export_report(self):
        """Export comprehensive report."""
        filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")], title="Export Report")
        if filename:
            with open(filename, 'w') as f:
                f.write("DENTAL ARCHWIRE DESIGN REPORT\n" + "="*60 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Software: Dental Archwire Design System v3.2\n\n")
                
                f.write("FILE INFORMATION\n")
                f.write("-"*30 + "\n")
                f.write(f"File: {os.path.basename(self.current_file)}\n")
                f.write(f"Arch Type: {self.arch_type_var.get().title()}\n")
                f.write(f"Wire Size: {self.wire_size_var.get()}\n")
                f.write(f"Surface: Lingual\n\n")
                
                f.write("CLASSIFICATION RESULTS\n")
                f.write("-"*30 + "\n")
                f.write(f"Total Teeth: {self.processing_result['teeth_detected']}\n")
                f.write(f"Incisors: {self.processing_result['incisors']}\n")
                f.write(f"Canines: {self.processing_result['canines']}\n")
                f.write(f"Posterior: {self.processing_result['posterior']}\n")
                f.write(f"Accuracy: {self.processing_result['classification_accuracy']:.1f}%\n\n")
                
                f.write("WIRE SPECIFICATIONS\n")
                f.write("-"*30 + "\n")
                f.write(f"Total Length: {self.processing_result['wire_length']:.1f} mm\n")
                f.write(f"Total Length: {self.processing_result['wire_length']/25.4:.2f} inches\n")
                f.write(f"Total Bends: {len(self.processing_result['bending_instructions'])-1}\n\n")
                
                f.write("BENDING INSTRUCTIONS\n")
                f.write("-"*60 + "\n")
                f.write("Step | Feed(mm) | Bend(°) | Location        | Notes\n")
                f.write("-"*60 + "\n")
                
                for inst in self.processing_result['bending_instructions']:
                    f.write(f"{inst['step']:4d} | {inst['feed_mm']:8.1f} | {inst['bend_deg']:7.1f} | "
                            f"{inst['location']:14s} | {inst['notes']}\n")
                
                f.write("\n" + "="*60 + "\n")
                f.write("End of Report\n")
            messagebox.showinfo("Success", f"Report exported to {filename}")

    def show_settings(self):
        """Show settings dialog."""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("500x400")
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        settings_window.update_idletasks()
        x = (settings_window.winfo_screenwidth() // 2) - (settings_window.winfo_width() // 2)
        y = (settings_window.winfo_screenheight() // 2) - (settings_window.winfo_height() // 2)
        settings_window.geometry(f'+{x}+{y}')
        
        notebook = ttk.Notebook(settings_window)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        general_frame = ttk.Frame(notebook)
        notebook.add(general_frame, text="General")
        
        general_content = ttk.Frame(general_frame, padding=20)
        general_content.pack(fill='both', expand=True)
        
        ttk.Label(general_content, text="Default Wire Size:").grid(row=0, column=0, sticky='w', pady=5)
        wire_combo = ttk.Combobox(general_content, values=["0.012", "0.014", "0.016", "0.018", "0.020", "0.016x0.022", "0.018x0.025", "0.019x0.025", "0.021x0.025"], state='readonly', width=15)
        wire_combo.set(self.wire_size_var.get())
        wire_combo.grid(row=0, column=1, padx=10, pady=5)
        
        button_frame = ttk.Frame(settings_window)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="OK", command=settings_window.destroy, style='Accent.TButton').pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=settings_window.destroy).pack(side='left', padx=5)

    def show_calibration(self):
        """Show calibration dialog."""
        calib_window = tk.Toplevel(self.root)
        calib_window.title("Machine Calibration")
        calib_window.geometry("600x400")
        calib_window.transient(self.root)
        calib_window.grab_set()
        
        calib_window.update_idletasks()
        x = (calib_window.winfo_screenwidth() // 2) - (calib_window.winfo_width() // 2)
        y = (calib_window.winfo_screenheight() // 2) - (calib_window.winfo_height() // 2)
        calib_window.geometry(f'+{x}+{y}')
        
        content = ttk.Frame(calib_window, padding=20)
        content.pack(fill='both', expand=True)
        
        ttk.Label(content, text="Wire Bender Calibration", font=self.fonts['heading']).pack(pady=(0, 20))
        
        info_text = """This wizard is a placeholder for a future calibration routine."""
        ttk.Label(content, text=info_text, font=self.fonts['body'], justify='left', wraplength=500).pack(pady=20)
        
        ttk.Button(content, text="Close", command=calib_window.destroy).pack()

    def show_help(self):
        """Show help dialog."""
        help_window = tk.Toplevel(self.root)
        help_window.title("User Guide")
        help_window.geometry("700x600")
        help_window.transient(self.root)
        
        help_window.update_idletasks()
        x = (help_window.winfo_screenwidth() // 2) - (help_window.winfo_width() // 2)
        y = (help_window.winfo_screenheight() // 2) - (help_window.winfo_height() // 2)
        help_window.geometry(f'+{x}+{y}')
        
        text_frame = ttk.Frame(help_window)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        text = tk.Text(text_frame, wrap='word', font=self.fonts['body'], relief='flat', bg='#f8f8f8')
        text.pack(side='left', fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame, command=text.yview)
        scrollbar.pack(side='right', fill='y')
        text.config(yscrollcommand=scrollbar.set)
        
        help_content = """DENTAL ARCHWIRE DESIGN SYSTEM - USER GUIDE

GETTING STARTED
===============
1. Click "Open STL" or use File > Open to load a dental arch STL file
2. The system will automatically detect if it's an upper or lower arch
3. Select wire size (default is 0.018)
4. Click "Generate Wire" to process the file

WORKFLOW
========
1. Load STL → 2. Process → 3. View Results → 4. Export Data/G-Code

VISUALIZATION OPTIONS
====================
Use the buttons above the visualization area:

📈 2D Path - Top-down view of the wire path
🎲 3D Preview - Static 3D view within the application.
👁️ Interactive 3D - Opens a new tab to launch a movable 3D view (requires Open3D).
📋 Instructions - Step-by-step bending instructions
📊 Analysis - Statistical analysis and charts
📜 G-Code - View and copy the generated G-Code.

MACHINE CONNECTION
==================
The panel on the left allows direct connection to an ESP32 machine:
1. Select the correct COM port from the dropdown.
2. Click "Connect". The button will turn green.
3. Once connected and G-Code is generated, click "Send G-Code to Machine".
4. Disconnect when finished.

KEYBOARD SHORTCUTS
==================
• Ctrl+O: Open file
• Ctrl+S: Export results
• Ctrl+G: Generate G-code
• Ctrl+Q: Exit application

TROUBLESHOOTING
===============
• For machine connection, install PySerial: pip install pyserial
• If 3D visualization doesn't work, install Open3D: pip install open3d
• For charts and graphs, install Matplotlib: pip install matplotlib
• Ensure orthodontic_processor.py is in the same directory.
• Check that your STL file is not corrupted."""
        text.insert(1.0, help_content)
        text.config(state='disabled')
        
        ttk.Button(help_window, text="Close", command=help_window.destroy).pack(pady=10)

    def show_about(self):
        """Show about dialog with a sample graphic."""
        about_window = tk.Toplevel(self.root)
        about_window.title("About")
        about_window.geometry("500x450")
        about_window.transient(self.root)
        about_window.grab_set()
        
        # Center the window
        about_window.update_idletasks()
        x = (about_window.winfo_screenwidth() // 2) - (about_window.winfo_width() // 2)
        y = (about_window.winfo_screenheight() // 2) - (about_window.winfo_height() // 2)
        about_window.geometry(f'+{x}+{y}')
        
        # Content
        content = ttk.Frame(about_window, padding=20)
        content.pack(fill='both', expand=True)
        
        title_label = ttk.Label(content, text="Dental Archwire Design System", font=self.fonts['title'], foreground=self.colors['accent'])
        title_label.pack(pady=(0, 5))
        
        version_label = ttk.Label(content, text="Version 3.2", font=self.fonts['heading'])
        version_label.pack()

        # Add a canvas for the picture
        canvas = tk.Canvas(content, width=200, height=100, bg=self.colors['panel_bg'], highlightthickness=0)
        canvas.pack(pady=10)
        
        # Draw a simple representation of a jaw arch and wire
        canvas.create_arc(20, 20, 180, 150, start=180, extent=180, style=tk.ARC, outline=self.colors['fg'], width=2)
        canvas.create_arc(30, 30, 170, 130, start=180, extent=180, style=tk.ARC, outline=self.colors['accent'], width=3)
        for i in range(7):
            x = 35 + i * 22
            canvas.create_rectangle(x, 28, x+5, 33, fill=self.colors['fg'])
        
        desc_text = """Professional software for lingual archwire design."""
        desc_label = ttk.Label(content, text=desc_text, font=self.fonts['body'], justify='center', wraplength=400)
        desc_label.pack(pady=10)

        # Close button
        ttk.Button(content, text="Close", command=about_window.destroy, style='Accent.TButton').pack()


    def update_status(self, message):
        """Update status bar message."""
        self.status_label.config(text=message)
        self.root.update_idletasks()


def main():
    """Main application entry point."""
    print("="*60)
    print("DENTAL ARCHWIRE DESIGN SYSTEM v3.2")
    print("="*60)

    print("\nChecking dependencies:")
    print(f"• Processor: {'✓' if PROCESSOR_AVAILABLE else '✗'} orthodontic_processor.py")
    print(f"• Open3D: {'✓' if OPEN3D_AVAILABLE else '✗'} (for interactive 3D visualization)")
    print(f"• Matplotlib: {'✓' if MATPLOTLIB_AVAILABLE else '✗'} (for embedded 2D/3D plots)")
    print(f"• PySerial: {'✓' if PYSERIAL_AVAILABLE else '✗'} (for direct machine connection)")

    if not all([PROCESSOR_AVAILABLE, MATPLOTLIB_AVAILABLE]):
        print("\n⚠️ WARNING: Core functionality may be limited due to missing dependencies.")

    print("\nStarting GUI...")

    root = tk.Tk()
    app = OrthodonticGUI(root)
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f'+{x}+{y}')
    root.mainloop()


if __name__ == "__main__":
    main()
