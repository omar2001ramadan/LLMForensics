// process_xml.cpp

#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <sstream>
#include <chrono>
#include <iomanip>

// Include TinyXML-2 for XML parsing
#include "tinyxml2.h"

// Include nlohmann/json for JSON handling
#include "json.hpp"

// Include OpenMP for parallel processing
#include <omp.h>

using json = nlohmann::json;
using namespace tinyxml2;
using namespace std;
using namespace std::chrono;

// Function to trim whitespace from both ends of a string
string trim(const string& str) {
    size_t first = str.find_first_not_of(" \n\r\t");
    if (first == string::npos)
        return "";
    size_t last = str.find_last_not_of(" \n\r\t");
    return str.substr(first, (last - first + 1));
}

// Function to check if a string looks like a file path
bool isFilePath(const string& str) {
    return str.find('/') != string::npos || str.find('\\') != string::npos;
}

// Function to display a progress bar
void displayProgress(size_t current, size_t total, steady_clock::time_point start_time) {
    float progress = static_cast<float>(current) / total;
    int barWidth = 50;
    cout << "\r[";
    int pos = static_cast<int>(barWidth * progress);
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) cout << "=";
        else if (i == pos) cout << ">";
        else cout << " ";
    }
    cout << "] " << int(progress * 100.0) << "%";

    // Estimate remaining time
    auto now = steady_clock::now();
    double elapsed_seconds = duration_cast<duration<double>>(now - start_time).count();
    double estimated_total_time = elapsed_seconds / progress;
    double remaining_time = estimated_total_time - elapsed_seconds;

    cout << " | Elapsed: " << fixed << setprecision(1) << elapsed_seconds << "s";
    cout << " | Remaining: " << fixed << setprecision(1) << max(0.0, remaining_time) << "s";
    cout.flush();
}

// Function to parse XML and populate the data structures
bool parseXML(const string& xmlFilePath,
              unordered_map<int, string>& id_to_string,
              unordered_map<int, string>& id_to_file_path,
              atomic<int>& current_string_id,
              atomic<int>& current_file_path_id,
              const string& edge_output_file) {
    cout << "[INFO] Loading XML file: " << xmlFilePath << endl;
    // Load XML file
    XMLDocument doc;
    XMLError eResult = doc.LoadFile(xmlFilePath.c_str());
    if (eResult != XML_SUCCESS) {
        cerr << "[ERROR] Failed to load XML file: " << xmlFilePath << endl;
        return false;
    }
    cout << "[INFO] XML file loaded successfully." << endl;

    // Navigate to the root <iefExport> element
    cout << "[INFO] Parsing XML structure..." << endl;
    XMLNode* root = doc.FirstChildElement("iefExport");
    if (root == nullptr) {
        cerr << "[ERROR] No <iefExport> root element in XML." << endl;
        return false;
    }

    // Navigate to the <Artifacts> element within <iefExport>
    XMLElement* artifactsElement = root->FirstChildElement("Artifacts");
    if (artifactsElement == nullptr) {
        cerr << "[ERROR] No <Artifacts> element in XML." << endl;
        return false;
    }

    // Mutex for thread-safe access to shared data structures
    mutex mtx;

    // Collect all <Artifact> elements
    vector<XMLElement*> artifacts;
    XMLElement* artifactElement = artifactsElement->FirstChildElement("Artifact");
    while (artifactElement != nullptr) {
        artifacts.push_back(artifactElement);
        artifactElement = artifactElement->NextSiblingElement("Artifact");
    }

    size_t total_artifacts = artifacts.size();
    cout << "[INFO] Total Artifacts: " << total_artifacts << endl;

    // Open the edge output file
    ofstream edge_file(edge_output_file, ios::out | ios::binary);
    if (!edge_file.is_open()) {
        cerr << "[ERROR] Unable to open edge output file: " << edge_output_file << endl;
        return false;
    }

    cout << "[INFO] Starting parallel processing of artifacts..." << endl;

    // Start time for progress bar and time estimates
    auto start_time = steady_clock::now();

    // Atomic counter for progress tracking
    atomic<size_t> artifacts_processed(0);

    // Parallel processing of artifacts
    #pragma omp parallel
    {
        // Thread-local data structures
        unordered_map<string, int> local_string_to_id;
        unordered_map<string, int> local_file_path_to_id;
        stringstream local_edge_stream;

        #pragma omp for schedule(dynamic)
        for (size_t idx = 0; idx < total_artifacts; ++idx) {
            XMLElement* artifactElement = artifacts[idx];

            // Iterate through <Hits> within each <Artifact>
            XMLElement* hitsElement = artifactElement->FirstChildElement("Hits");
            if (hitsElement != nullptr) {
                XMLElement* hitElement = hitsElement->FirstChildElement("Hit");
                while (hitElement != nullptr) {
                    vector<int> strings_in_hit;
                    vector<int> file_paths_in_hit;

                    // Iterate through all <Fragment> elements within <Hit> to extract strings and file paths
                    XMLElement* fragmentElement = hitElement->FirstChildElement("Fragment");
                    while (fragmentElement != nullptr) {
                        const char* fragmentText = fragmentElement->GetText();
                        if (fragmentText) {
                            string s = trim(fragmentText);

                            if (isFilePath(s)) {
                                // Map file path to ID
                                int fp_id;
                                auto it_fp = local_file_path_to_id.find(s);
                                if (it_fp == local_file_path_to_id.end()) {
                                    // Assign a new ID atomically
                                    fp_id = current_file_path_id.fetch_add(1);
                                    local_file_path_to_id[s] = fp_id;

                                    // Thread-safe insertion into shared id_to_file_path map
                                    {
                                        lock_guard<mutex> lock(mtx);
                                        id_to_file_path[fp_id] = s;
                                    }
                                } else {
                                    fp_id = it_fp->second;
                                }
                                file_paths_in_hit.push_back(fp_id);
                            } else {
                                // Convert to lowercase for consistency
                                transform(s.begin(), s.end(), s.begin(), ::tolower);

                                // Map string to ID
                                int s_id;
                                auto it = local_string_to_id.find(s);
                                if (it == local_string_to_id.end()) {
                                    // Assign a new ID atomically
                                    s_id = current_string_id.fetch_add(1);
                                    local_string_to_id[s] = s_id;

                                    // Thread-safe insertion into shared id_to_string map
                                    {
                                        lock_guard<mutex> lock(mtx);
                                        id_to_string[s_id] = s;
                                    }
                                } else {
                                    s_id = it->second;
                                }
                                strings_in_hit.push_back(s_id);
                            }
                        }
                        fragmentElement = fragmentElement->NextSiblingElement("Fragment");
                    }

                    // Associate strings with file paths by creating edges
                    for (const int& fp_id : file_paths_in_hit) {
                        for (size_t i = 0; i < strings_in_hit.size(); ++i) {
                            for (size_t j = i + 1; j < strings_in_hit.size(); ++j) {
                                int s1_id = strings_in_hit[i];
                                int s2_id = strings_in_hit[j];

                                // Ensure consistent ordering
                                int min_id = min(s1_id, s2_id);
                                int max_id = max(s1_id, s2_id);

                                // Write edge with associated file path ID to the local stream
                                local_edge_stream << min_id << "," << max_id << "," << fp_id << "\n";
                            }
                        }
                    }

                    hitElement = hitElement->NextSiblingElement("Hit");
                }
            }

            // Update progress
            size_t processed = ++artifacts_processed;
            if (omp_get_thread_num() == 0 && processed % 10 == 0) {
                displayProgress(processed, total_artifacts, start_time);
            }
        }

        // Thread-safe writing of local edges to the edge file
        {
            lock_guard<mutex> lock(mtx);
            edge_file << local_edge_stream.str();
        }
    }

    // Ensure progress bar reaches 100%
    displayProgress(total_artifacts, total_artifacts, start_time);
    cout << endl;

    edge_file.close();
    cout << "[INFO] Parallel processing completed." << endl;
    return true;
}

int main(int argc, char* argv[]) {
    // Check for minimum required arguments
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <XML_File_Path> [--output_json <JSON_File_Path>] [--edge_file <Edge_File_Path>]" << endl;
        return 1;
    }

    // Default parameters
    string xmlFilePath = argv[1];
    string output_json = "string_dictionary.json";
    string edge_output_file = "edges.csv";

    // Parse command-line arguments
    for (int i = 2; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--output_json" && i + 1 < argc) {
            output_json = argv[++i];
        } else if (arg == "--edge_file" && i + 1 < argc) {
            edge_output_file = argv[++i];
        } else {
            cerr << "[WARNING] Unknown or incomplete argument: " << arg << endl;
        }
    }

    // Shared data structures
    unordered_map<int, string> id_to_string;
    unordered_map<int, string> id_to_file_path;
    atomic<int> current_string_id(0);
    atomic<int> current_file_path_id(0);

    // Start total execution time
    auto total_start_time = steady_clock::now();

    // Parse the XML file
    if (!parseXML(xmlFilePath, id_to_string, id_to_file_path, current_string_id, current_file_path_id, edge_output_file)) {
        cerr << "[ERROR] Failed to parse XML file." << endl;
        return 1;
    }

    // Total execution time
    auto total_end_time = steady_clock::now();
    double total_elapsed_seconds = duration_cast<duration<double>>(total_end_time - total_start_time).count();

    cout << "[INFO] Completed parsing in " << fixed << setprecision(1) << total_elapsed_seconds << " seconds." << endl;
    cout << "[INFO] Total unique strings: " << id_to_string.size() << endl;
    cout << "[INFO] Total unique file paths: " << id_to_file_path.size() << endl;

    // Serialize the string dictionary to JSON
    json graph_json;
    json nodes_json = json::array();

    // Add string nodes
    for (const auto& pair : id_to_string) {
        json node;
        node["id"] = pair.first;
        node["label"] = pair.second;
        nodes_json.push_back(node);
    }

    graph_json["nodes"] = nodes_json;

    // Write JSON nodes to file
    ofstream json_file(output_json);
    if (!json_file.is_open()) {
        cerr << "[ERROR] Unable to open JSON file for writing: " << output_json << endl;
        return 1;
    }

    json_file << graph_json.dump(4); // Pretty print with 4-space indentation
    json_file.close();

    // Serialize the file path dictionary to JSON
    string file_path_output = "file_path_dictionary.json";
    json file_paths_json;
    json file_paths_array = json::array();

    for (const auto& pair : id_to_file_path) {
        json fp;
        fp["id"] = pair.first;
        fp["path"] = pair.second;
        file_paths_array.push_back(fp);
    }

    file_paths_json["file_paths"] = file_paths_array;

    ofstream fp_json_file(file_path_output);
    if (!fp_json_file.is_open()) {
        cerr << "[ERROR] Unable to open file for writing: " << file_path_output << endl;
        return 1;
    }

    fp_json_file << file_paths_json.dump(4);
    fp_json_file.close();

    cout << "[INFO] JSON nodes saved to '" << output_json << "'." << endl;
    cout << "[INFO] File paths saved to '" << file_path_output << "'." << endl;
    cout << "[INFO] Edges with file paths saved to '" << edge_output_file << "'." << endl;

    return 0;
}
