Dental Archwire Design System
This project is a comprehensive system for designing and manufacturing custom orthodontic archwires. It processes 3D dental models from STL files, generates a precise lingual wire path, creates visualizations, and produces G-code for an automated wire bending machine.
🌟 Key Features
STL File Processing: Loads and analyzes 3D dental arch models from .stl files.
Automatic Tooth Detection: Identifies and classifies teeth (incisors, canines, posterior).
Lingual Wire Design: Calculates optimal bracket positions and generates a smooth, precise lingual wire path.
Multi-View Visualization:
Interactive 3D view of the dental model, brackets, and wire.
2D top-down plot of the wire path.
Static 3D preview.
G-Code Generation: Creates machine-readable G-code for an ESP32-based wire bending machine.
Direct Machine Control: Connects directly to the wire bender via a serial port to send G-code commands.
Data Export: Exports results to various formats, including JSON, CSV (bending instructions), and comprehensive text reports.
User-Friendly GUI: An intuitive graphical interface built with Tkinter for easy operation.
🏗️ System Architecture
The system is composed of three main Python modules that work together:
tes_stl9claude_latestin.py (Core Wire Generator): This is the foundational engine. It takes an STL file, performs the heavy lifting of mesh analysis, tooth detection, and calculates the raw geometry for the lingual wire path and bracket placement.
orthodontic_processor.py (Backend Logic): This module acts as a bridge between the core generator and the user interface. It takes the raw data from the generator, post-processes it into a user-friendly format (e.g., calculating wire length, generating bending instructions), and handles data export logic. It can also run in a "mock" mode if the core generator is unavailable.
orthodontic_guiv4.py (Frontend GUI): This is the user-facing application. It provides all the controls, displays visualizations, and manages the connection to the wire bending machine. It communicates with the orthodontic_processor to run the analysis and display the results.
graph TD
    A[User loads STL in GUI] --> B{orthodontic_guiv4.py};
    B --> C{orthodontic_processor.py};
    C --> D{tes_stl9claude_latestin.py};
    D -- Raw Geometric Data --> C;
    C -- Processed Results & Instructions --> B;
    B -- Displays Visuals & G-Code --> E[User];
    B -- Sends G-Code --> F[ESP32 Wire Bender];


⚙️ Installation & Requirements
To run this application, you need Python 3 and the following libraries. You can install them using pip:
pip install numpy open3d scipy matplotlib pyserial


🚀 How to Run
Ensure all three Python files (orthodontic_guiv4.py, orthodontic_processor.py, tes_stl9claude_latestin.py) are in the same directory.
Run the main GUI application from your terminal:
python orthodontic_guiv4.py


Workflow:
Click "Open STL" to load a dental model.
Adjust parameters like Arch Type and Wire Size if needed.
Click "Generate Wire" to start the analysis.
Once processing is complete, use the view tabs (2D Path, 3D Preview, Instructions, etc.) to inspect the results.
To connect to a wire bender, select the correct COM port and click "Connect".
Click "Send G-Code to Machine" to start the bending process.
Use the File > Export menu to save results, reports, or the G-code file.
📄 Full File & Function Documentation
1. orthodontic_guiv4.py
This file contains the main OrthodonticGUI class which builds and manages the entire graphical user interface using Tkinter.
class OrthodonticGUI
The main class for the application's user interface.


Method
Description
__init__(self, root)
Initializes the main application window, styles, and lays out all GUI components.
setup_styles(self)
Configures the visual style (colors, fonts) for the ttk widgets to give the application a modern look.
create_menu(self)
Creates the main application menu bar (File, View, Tools, Help).
create_toolbar(self)
Creates the quick-access toolbar with buttons for common actions like Open and Process.
create_main_layout(self)
Sets up the main three-panel layout (left: controls, center: visualization, right: results).
create_left_panel(self, parent)
Creates the left-side panel containing file info, processing parameters, action buttons, and the machine connection interface.
create_connection_panel(self, parent)
Creates the specific frame for connecting to the ESP32 wire bender, including port selection and connect/disconnect buttons.
create_center_panel(self, parent)
Creates the central panel that holds the notebook for switching between different visualizations (2D, 3D, G-code, etc.).
create_right_panel(self, parent)
Creates the right-side panel for displaying text-based analysis and processing results.
create_status_bar(self)
Creates the status bar at the bottom of the window to display status messages.
open_file(self)
Opens a file dialog to allow the user to select an STL file.
show_file_info(self)
Displays basic information about the loaded STL file in the results panel.
process_file(self)
Initiates the wire generation process by calling the backend processor in a separate thread to keep the GUI responsive.
_process_file_thread(self)
The actual processing function that runs in a background thread.
_process_complete(self, success, result)
Callback function executed on the main thread after processing is finished. Updates the GUI with results or an error message.
show_interactive_3d_view(self)
Creates a new tab with a button to launch the fully interactive Open3D visualization window.
launch_3d_viewer(self)
Launches the external Open3D window to display the model, wire, and brackets.
show_analysis_in_view(self)
Generates and displays statistical charts (e.g., tooth distribution, segment lengths) in a new view tab using Matplotlib.
generate_gcode_in_view(self)
Generates the G-code from the results and displays it in a new, dedicated view tab with syntax highlighting.
generate_gcode_content(self)
Constructs the complete G-code string based on the bending instructions from the processor.
highlight_gcode(self)
Applies color syntax highlighting to the G-code text widget for better readability.
copy_gcode_to_clipboard(self)
Copies the generated G-code to the system clipboard.
update_serial_ports(self)
Refreshes the list of available COM ports for the machine connection.
toggle_connection(self)
Connects to or disconnects from the selected serial port.
connect_serial(self)
Handles the logic for establishing a serial connection to the ESP32.
disconnect_serial(self)
Closes the active serial connection.
send_gcode_to_esp(self)
Initiates sending the generated G-code line-by-line to the connected machine in a background thread.
_send_gcode_thread(self)
The function running in a background thread that sends G-code and waits for an 'ok' response from the machine after each line.
clear_results(self)
Clears all previous results, visualizations, and generated data from the GUI.
show_welcome_screen(self)
Displays the initial welcome screen in the visualization panel.
display_results(self, result)
Populates the right-side results panel with the processed data (tooth counts, wire length, etc.).
visualize_results(self, result)
Calls the functions to create and display the 2D plot, 3D preview, and bending instructions table.
create_2d_preview(self, result)
Creates the 2D top-down plot of the wire path using Matplotlib and embeds it in the GUI.
create_3d_preview(self, result)
Creates a static 3D scatter plot of the wire path using Matplotlib and embeds it in the GUI.
show_bending_instructions_tab(self, result)
Creates a new view tab and displays the step-by-step bending instructions in a clear, sortable table.
on_closing(self)
Handles the window-closing event, ensuring the serial connection is safely terminated.
export_results(self)
Opens a dialog window allowing the user to choose an export format (JSON, CSV, G-Code, Report).
export_json(self)
Handles exporting the full results to a JSON file.
export_instructions_csv(self)
Handles exporting the bending instructions to a CSV file.
export_gcode(self)
Handles exporting the G-code to a .gcode file.
export_report(self)
Handles exporting a comprehensive summary report to a text file.
show_settings(self)
Displays a settings dialog (placeholder).
show_calibration(self)
Displays a machine calibration dialog (placeholder).
show_help(self)
Displays a helpful user guide in a new window.
show_about(self)
Displays the 'About' window with application information.
update_status(self, message)
Updates the text in the bottom status bar.

2. orthodontic_processor.py
This file acts as the backend "brain" of the application, connecting the core wire generation logic with the GUI. It takes raw data and turns it into meaningful, processed information.
class OrthodonticProcessor
Manages the wire generation workflow and data post-processing.
Method
Arguments
Returns
Description
__init__(self)
-
None
Initializes the processor, checking if the real wire generator (CleanLingualWireGenerator) is available.
generate_wire(...)
file_path, arch_type, wire_size
Dict
The main public method. It orchestrates the entire wire generation process, either by calling the real generator or a mock version, and then post-processes the results.
_process_with_real_generator(...)
file_path, arch_type, wire_size
Dict
(Private) Calls the CleanLingualWireGenerator from tes_stl9claude_latestin.py to perform the core analysis.
_process_with_mock_generator(...)
file_path, arch_type, wire_size
Dict
(Private) A fallback method used for testing or when the main generator is missing. It produces realistic-looking mock data.
_generate_mock_bracket_positions(self)
-
List[Dict]
(Private) Generates a list of mock bracket positions for use by the mock generator.
_post_process_results(...)
raw_result, start_time
Dict
(Private) Takes the raw output from the generator and enriches it by calculating wire length, accuracy, and generating human-readable bending instructions.
_get_tooth_color(...)
tooth_type
str
(Private) Returns a color string based on the tooth type for visualization purposes.
_calculate_wire_length(...)
bracket_positions
float
(Private) Calculates the total length of the wire by summing the distances between consecutive bracket points.
_calculate_classification_accuracy(...)
tooth_counts
float
(Private) Calculates a simple accuracy score based on whether the expected number of incisors (4) and canines (2) were detected.
_generate_bending_instructions(...)
bracket_positions
List[Dict]
(Private) A key function that converts the 3D bracket positions into a sequence of machine instructions (feed distance, bend angle).
_get_clinical_note(...)
tooth_type, bend_angle, etc.
str
(Private) Generates a descriptive note for each bending step (e.g., "Major bend at Canine").
_validate_results(...)
result
None
(Private) Performs sanity checks on the final processed result to warn about unusual values (e.g., very low tooth count or strange wire length).
get_processing_status(self)
-
str
Returns the current status of the processor (e.g., "Ready", "Processing", "Complete").
get_last_result(self)
-
Optional[Dict]
Returns the most recent processing result.
export_to_json(...)
result, filename
None
Formats and saves the processing result as a .json file.
export_to_csv(...)
result, filename
None
Formats and saves the bending instructions as a .csv file.

3. tes_stl9claude_latestin.py
This is the core scientific and geometric processing engine of the system. It handles the complex task of analyzing the 3D mesh to identify anatomical features and generate the wire path.
class CleanLingualWireGenerator
Performs the 3D analysis of the dental mesh.
Method
Arguments
Returns
Description
__init__(...)
stl_path, arch_type, wire_size
None
Initializes the generator with the path to the STL file and other parameters.
load_mesh(self)
-
bool
Loads the STL file into an Open3D mesh object, cleans it (removes duplicates/degenerate triangles), and computes normals.
detect_teeth_simple(self)
-
bool
A simplified tooth detection algorithm that uses angular segmentation based on the crown-level vertices of the mesh to identify individual teeth.
_simple_angular_segmentation(...)
crown_vertices, center
List
(Private) The core logic for the angular segmentation of teeth.
_apply_simple_classification(self)
-
None
(Private) A simple classification scheme that identifies the 6 most anterior teeth and classifies them as 4 incisors and 2 canines based on their relative positions.
_fallback_classification(self)
-
None
(Private) A fallback method used if the primary classification fails to find the correct number of incisors and canines. It uses a "centrality" score to make a best-guess assignment.
calculate_lingual_bracket_positions(self)
-
None
Calculates the precise 3D coordinates for bracket placement on the lingual (inner) surface of each tooth, taking into account height and an inward offset.
create_smooth_wire_path(self)
-
List
Creates a path connecting the calculated bracket positions. The path is sorted by angle to ensure the correct tooth order.
create_clean_visualization_meshes(self)
-
Tuple[TriangleMesh, TriangleMesh]
Creates the 3D geometry for visualization. This includes creating cylinder meshes for the wire segments and box meshes for the brackets, coloring them appropriately (e.g., blue for posterior).
visualize_clean_system(...)
components
None
Launches an Open3D visualization window to display the final generated system (dental model, wire, brackets).
generate_clean_lingual_wire(self)
-
Optional[Dict]
The main public method of this class. It executes the entire pipeline—load, detect, position, create path, and create visuals—and returns a dictionary containing all the generated geometric data.

G-Code Commands
The system generates G-code tailored for a simple 2-axis wire bender (one axis for feeding, one for bending).
G90: Set to absolute positioning.
G21: Set units to millimeters.
G1 X[distance] F300: Feed the wire forward by [distance] millimeters.
G1 Y[angle] F60: Bend the wire to the specified [angle].
G1 Y0 F120: Return the bender to its neutral (0-degree) position.
M17: Enable stepper motors.
M84: Disable stepper motors to conserve power.
M0: Program stop.
; [comment]: Comments are used to describe each step.
License
This project is open-source. Please add your preferred license (e.g., MIT, GPL) here.
