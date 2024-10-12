import os
import argparse
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import sys
import chardet

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate XML report of directory contents.")
    parser.add_argument("input_dir", help="Path to the input directory.")
    parser.add_argument("output_xml", help="Path to the output XML file.")
    return parser.parse_args()

def detect_encoding(file_path):
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB for detection
        result = chardet.detect(raw_data)
        return result['encoding'] if result['encoding'] else 'utf-8'
    except Exception as e:
        print(f"Error detecting encoding for {file_path}: {e}")
        return 'utf-8'

def extract_ascii_content(file_path):
    try:
        encoding = detect_encoding(file_path)
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            content = f.read()
        # Filter out non-ASCII characters
        ascii_content = ''.join([c if ord(c) < 128 else '?' for c in content])
        return ascii_content
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def create_xml_structure():
    # Create root element with attributes
    root = ET.Element("iefExport", {
        "version": "1.0",
        "iefVersion": "8.2.0.40565"
    })

    # Create Artifacts element
    artifacts = ET.SubElement(root, "Artifacts")

    # Create Artifact for File System Information
    fs_artifact = ET.SubElement(artifacts, "Artifact", {"name": "File System Information"})
    
    # Define Fragments for File System Information
    fs_fragments = ET.SubElement(fs_artifact, "Fragments")
    fs_fragment_names = [
        ("Tags", "Bookmark"),
        ("Comments", "Bookmark,Comment"),
        ("ID", "None"),
        ("Volume Serial Number", "DeviceIdentifier"),
        ("Full Volume Serial Number", "DeviceIdentifier"),
        ("File System", "None"),
        ("Sectors per cluster", "None"),
        ("Bytes per sector", "None"),
        ("Starting Sector", "None"),
        ("Ending Sector", "None"),
        ("Total Sectors", "None"),
        ("Total Clusters", "None"),
        ("Free Clusters", "None"),
        ("Total Capacity (Bytes)", "None"),
        ("Unallocated Area (Bytes)", "None"),
        ("Allocated Area (Bytes)", "None"),
        ("Volume Name", "DeviceIdentifier"),
        ("Volume Offset (Bytes)", "None"),
        ("Drive Type", "None"),
        ("Source", "Source"),
        ("Location", "Source"),
        ("Evidence number", "Source"),
        ("Deleted source", "Source"),
        ("Recovery method", "Source"),
        ("Item ID", "None")
    ]
    for name, category in fs_fragment_names:
        ET.SubElement(fs_fragments, "Fragment", {"name": name, "category": category})

    # Create Hits element
    fs_hits = ET.SubElement(fs_artifact, "Hits")

    return root, fs_hits

def add_file_hit(fs_hits, sequence_number, file_info):
    hit = ET.SubElement(fs_hits, "Hit", {"sequenceNumber": str(sequence_number)})

    # Populate Fragments based on user-defined Fragment names
    fragments = {
        "File System": file_info.get("file_system", "None"),
        "Source": file_info.get("source", "None"),
        "Location": file_info.get("location", "None"),
        "Evidence number": file_info.get("evidence_number", "None"),
        "Recovery method": file_info.get("recovery_method", "None"),
        "Item ID": str(file_info.get("item_id", sequence_number)),
        # Add more fragments as needed
    }

    for name, value in fragments.items():
        ET.SubElement(hit, "Fragment", {"name": name}).text = value

def prettify_xml(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def main():
    args = parse_arguments()
    input_dir = args.input_dir
    output_xml = args.output_xml

    if not os.path.isdir(input_dir):
        print(f"Error: The input path '{input_dir}' is not a directory or does not exist.")
        sys.exit(1)

    root, fs_hits = create_xml_structure()
    sequence_number = 1

    # Traverse the directory
    for dirpath, dirnames, filenames in os.walk(input_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            abs_path = os.path.abspath(file_path)
            rel_path = os.path.relpath(file_path, input_dir)

            file_info = {
                "file_system": "ZIP" if filename.lower().endswith('.zip') else "None",
                "source": abs_path,
                "location": rel_path,
                "evidence_number": "PhysicalDrive0",
                "recovery_method": "Parsing",
                "item_id": sequence_number
            }

            # If the file is an XML, extract ASCII content
            if filename.lower().endswith('.xml'):
                ascii_content = extract_ascii_content(file_path)
                file_info["ascii_content"] = ascii_content
                # You can add ascii_content to a specific Fragment if needed

            add_file_hit(fs_hits, sequence_number, file_info)
            sequence_number += 1

    # Write the XML to file
    pretty_xml = prettify_xml(root)
    try:
        with open(output_xml, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
        print(f"XML report successfully written to '{output_xml}'.")
    except Exception as e:
        print(f"Error writing XML to file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Check if chardet is installed
    try:
        import chardet
    except ImportError:
        print("The 'chardet' library is required to run this script.")
        print("You can install it using 'pip install chardet'.")
        sys.exit(1)
    main()
