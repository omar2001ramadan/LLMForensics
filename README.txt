--------------------FORENSIC ANALYSIS.py-------------------


-> Basic Usage (Visualization only): 
python forensic_analysis.py /path/to/output.json

-> With a Query:
python forensic_analysis.py /path/to/output.json --query "What is the telephone number of the device?" --output forensic_graph.html



Additional Options: {

-> Specify Output File:
python forensic_analysis.py /path/to/output.json --query "..." --output custom_graph.html

-> Set Maximum Path Depth:
python forensic_analysis.py /path/to/output.json --query "..." --max_depth 3
}

------------------------TREE FINDER.CPP (exe) ----------------------
README
XML to JSON Graph Converter
This program processes an XML file to extract strings and their associations through file paths, building a graph where:

Nodes represent unique strings.
Edges represent pairs of strings that co-occur in the same file paths.
File Paths are associated with edges to indicate where the strings co-occur.
The program is optimized for efficiency and can handle large datasets by minimizing memory usage and leveraging parallel processing.

Features
Efficient Parsing: Utilizes TinyXML-2 for fast and efficient XML parsing.
Memory Optimization: Maps strings and file paths to integer IDs to reduce memory consumption.
Parallel Processing: Leverages OpenMP to process XML data in parallel, speeding up execution.
Edge Storage: Writes edges directly to disk to avoid high memory usage.
JSON Output: Generates JSON files for string nodes and file paths for easy integration.
Scalability: Capable of handling large XML files without excessive memory requirements.
Dependencies
C++11 or higher
TinyXML-2: XML parsing library.
nlohmann/json: JSON library for C++.
OpenMP: For parallel processing support.
Installation
1. Install TinyXML-2
Linux (using package manager):
bash
Copy code
sudo apt-get install libtinyxml2-dev
MacOS (using Homebrew):
bash
Copy code
brew install tinyxml2
Windows:
Download and build from TinyXML-2 GitHub.
2. Install nlohmann/json
Download the single-header file from the GitHub repository.
Place json.hpp in your include path.
Compilation
Compile the program using a C++ compiler with OpenMP support. Here's an example using g++:

bash
Copy code
g++ -std=c++11 -fopenmp -o process_xml process_xml.cpp -ltinyxml2
Note: Ensure that the compiler can find the tinyxml2 and nlohmann/json headers.

Usage
bash
Copy code
./process_xml <XML_File_Path> [--output_json <JSON_File_Path>] [--edge_file <Edge_File_Path>]
Arguments:

<XML_File_Path>: Path to the input XML file.
--output_json <JSON_File_Path>: (Optional) Path to save the JSON nodes file. Default is string_dictionary.json.
--edge_file <Edge_File_Path>: (Optional) Path to save the edges file. Default is edges.csv.
Example:

bash
Copy code
./process_xml sample.xml --output_json nodes.json --edge_file edges.csv
Output Files
1. Nodes JSON (string_dictionary.json or specified name):
Contains a list of unique strings with their assigned IDs and labels.

Structure:

json
Copy code
{
  "nodes": [
    {
      "id": 0,
      "label": "stringa"
    },
    {
      "id": 1,
      "label": "stringb"
    },
    ...
  ]
}
2. Edges CSV (edges.csv or specified name):
Contains edges between strings, associated with file path IDs.

Format:

Copy code
string_id1,string_id2,file_path_id
3. File Paths JSON (file_path_dictionary.json):
Contains a list of file paths with their assigned IDs.

Structure:

json
Copy code
{
  "file_paths": [
    {
      "id": 0,
      "path": "/path/to/file1"
    },
    {
      "id": 1,
      "path": "/path/to/file2"
    },
    ...
  ]
}
Processing the Output
Edges:

Edges are stored in edges.csv to avoid loading all edges into memory.
Each line represents an edge between two string IDs and the file path ID where they co-occur.
Associations:

Use the string_dictionary.json and file_path_dictionary.json to map IDs to actual strings and file paths.
This allows you to interpret the edges and understand the relationships.
Example Workflow
1. Run the Program:
bash
Copy code
./process_xml data.xml
2. Parse the Nodes:
Use a JSON parser in your preferred language to read string_dictionary.json.

Example in Python:

python
Copy code
import json

with open('string_dictionary.json', 'r') as f:
    data = json.load(f)
    id_to_string = {node['id']: node['label'] for node in data['nodes']}
3. Parse the File Paths:
python
Copy code
with open('file_path_dictionary.json', 'r') as f:
    data = json.load(f)
    id_to_file_path = {fp['id']: fp['path'] for fp in data['file_paths']}
4. Process Edges:
Read edges.csv line by line to analyze the relationships.

Example in Python:

python
Copy code
edges = []
with open('edges.csv', 'r') as f:
    for line in f:
        s1_id, s2_id, fp_id = map(int, line.strip().split(','))
        s1 = id_to_string[s1_id]
        s2 = id_to_string[s2_id]
        file_path = id_to_file_path[fp_id]
        edges.append((s1, s2, file_path))