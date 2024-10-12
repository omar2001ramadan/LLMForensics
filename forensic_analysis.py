import json
import csv
import networkx as nx
from pyvis.network import Network
from openai import OpenAI  # Updated import
import argparse
import os
import sys
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import re
import concurrent.futures
from tqdm import tqdm
import logging
import time

from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for rate limiting
CALLS_PER_MINUTE = 5000
RATE_LIMIT_PERIOD = 60  # in seconds

# Define rate limit decorator
@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=RATE_LIMIT_PERIOD)
def rate_limited():
    """Placeholder function to enforce rate limiting."""
    pass

# Define retry decorator with exponential backoff
def retry_on_exception():
    return retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(Exception)
    )

# Helper function to make OpenAI API calls with rate limiting and retry
@retry_on_exception()
def safe_create_chat_completion(client: OpenAI, model: str, messages: list, max_completion_tokens: int):
    """
    Makes a rate-limited and retried OpenAI chat completion request.
    """
    try:
        # Enforce rate limiting
        rate_limited()
        
        # Make the API call
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
        )
        
        # Check for empty response
        content = response.choices[0].message.content.strip()
        if not content:
            raise ValueError("Received empty response from OpenAI API.")
        
        # Remove Markdown code fences if present
        if content.startswith("```") and content.endswith("```"):
            content = "\n".join(content.split("\n")[1:-1])
        
        return content

    except Exception as e:
        logging.error(f"API call failed: {str(e)}")
        raise  # Trigger the retry mechanism

def is_mapping_label(label: str) -> bool:
    """
    Determines if the label is a mapping-only identifier.
    Returns True if the label is purely numeric or a hash-like string.
    """
    # Check if the label is purely numeric
    if label.isdigit():
        return True
    # Check if the label is a hash-like string (e.g., hexadecimal with length >= 32)
    if re.fullmatch(r'[a-fA-F0-9]{32,}', label):
        return True
    return False

def is_image_file(label: str, file_path_map: Dict[int, str], G: nx.DiGraph) -> bool:
    """
    Determines if the label corresponds to an image file based on associated file paths.
    Returns True if any associated file path ends with a common image extension.
    """
    # Common image file extensions
    image_extensions = ('.png', '.gif', '.jpeg', '.jpg', '.bmp', '.tiff', '.svg', '.webp')
    
    # Find the node ID based on the label
    node_id = None
    for nid, data in G.nodes(data=True):
        if data.get('label') == label:
            node_id = nid
            break
    if node_id is None:
        return False  # Node not found
    
    # Gather all file paths associated with edges connected to this node
    connected_file_paths = []
    for _, target, edge_data in G.edges(node_id, data=True):
        connected_file_paths.extend(edge_data.get('files', []))
    for source, _, edge_data in G.in_edges(node_id, data=True):
        connected_file_paths.extend(edge_data.get('files', []))
    
    # Check if any file path ends with an image extension
    for file_path in connected_file_paths:
        if isinstance(file_path, str) and any(file_path.lower().endswith(ext) for ext in image_extensions):
            logging.debug(f"Label '{label}' is associated with an image file: '{file_path}'.")
            return True
    return False

def load_graph_data(string_dict_path: str, edge_csv_path: str, file_path_dict_path: str) -> Dict:
    """
    Load graph data from the specified files.

    Returns a dictionary with 'nodes' and 'edges'.
    """
    # Load nodes from string_dictionary.json
    try:
        with open(string_dict_path, 'r', encoding='utf-8') as f:
            string_dict = json.load(f)
            nodes = string_dict.get('nodes', [])
            logging.info(f"Loaded {len(nodes)} nodes from '{string_dict_path}'.")
    except FileNotFoundError:
        logging.error(f"The file '{string_dict_path}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from '{string_dict_path}'. {str(e)}")
        sys.exit(1)

    # Load file paths from file_path_dictionary.json
    try:
        with open(file_path_dict_path, 'r', encoding='utf-8') as f:
            file_path_dict = json.load(f)
            file_paths = file_path_dict.get('file_paths', [])
            file_path_map = {fp['id']: fp['path'] for fp in file_paths}
            logging.info(f"Loaded {len(file_paths)} file paths from '{file_path_dict_path}'.")
    except FileNotFoundError:
        logging.error(f"The file '{file_path_dict_path}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from '{file_path_dict_path}'. {str(e)}")
        sys.exit(1)

    # Load edges from edges.csv
    edges = []
    try:
        # First, count total number of edges for progress bar
        with open(edge_csv_path, 'r', encoding='utf-8') as f:
            total_edges = sum(1 for _ in f)
        with open(edge_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            logging.info(f"Total edges to load: {total_edges}")
            for row in tqdm(reader, total=total_edges, desc="Loading edges"):
                if len(row) != 3:
                    continue  # Skip invalid rows
                source_id, target_id, file_path_id = row
                edges.append({
                    'source': int(source_id),
                    'target': int(target_id),
                    'file_path_id': int(file_path_id)
                })
    except FileNotFoundError:
        logging.error(f"The file '{edge_csv_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error reading edges from '{edge_csv_path}': {str(e)}")
        sys.exit(1)

    # Map file_path_id to actual file paths in edges
    logging.info("Mapping file paths to edges...")
    for edge in tqdm(edges, desc="Mapping file paths"):
        file_path_id = edge.get('file_path_id')
        edge['files'] = [file_path_map.get(file_path_id, 'Unknown')]
        # Remove 'file_path_id' as it's no longer needed
        del edge['file_path_id']

    return {'nodes': nodes, 'edges': edges, 'file_path_map': file_path_map}

def create_graph(data: dict) -> nx.DiGraph:
    """
    Create a NetworkX directed graph from data.
    """
    G = nx.DiGraph()

    # Add nodes
    logging.info("Adding nodes to the graph...")
    for node in tqdm(data.get('nodes', []), desc="Adding nodes"):
        node_id = node.get('id')
        label = node.get('label', node_id)
        G.add_node(node_id, label=label, data=node.get('data', {}))

    # Add edges
    logging.info("Adding edges to the graph...")
    for edge in tqdm(data.get('edges', []), desc="Adding edges"):
        source = edge.get('source')
        target = edge.get('target')
        files = edge.get('files', [])
        if source is not None and target is not None:
            G.add_edge(source, target, files=files)
        else:
            logging.warning(f"Edge with missing source or target: {edge}")

    logging.info(f"Graph creation complete. Total nodes: {G.number_of_nodes()}, Total edges: {G.number_of_edges()}")
    return G

def visualize_graph(G: nx.DiGraph, output_file: str = "graph.html") -> None:
    """
    Generate an interactive HTML visualization of the graph using PyVis.
    """
    logging.info("Visualizing the graph...")
    net = Network(notebook=False, directed=True, height="750px", width="100%")
    net.from_nx(G)

    # Create a mapping from (source, target) to files for quick lookup
    edge_files = {(source, target): data.get('files', []) for source, target, data in G.edges(data=True)}

    # Iterate through net.edges and set the 'title' attribute
    for edge in net.edges:
        source = edge['from']
        target = edge['to']
        files = edge_files.get((source, target), [])
        if files:
            edge['title'] = ', '.join(files)
        else:
            edge['title'] = "No associated files"

    # Configure the physics for better layout
    net.force_atlas_2based()

    # Generate and save the HTML file
    net.show(output_file)
    logging.info(f"Graph has been visualized and saved to '{output_file}'")

def get_openai_api_key() -> str:
    """
    Retrieve the OpenAI API key from environment variables.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logging.error("The OpenAI API key is not set. Please set it in the '.env' file or as an environment variable 'OPENAI_API_KEY'.")
        sys.exit(1)
    return api_key

def filter_irrelevant_nodes(client: OpenAI, G: nx.DiGraph, file_path_map: Dict[int, str]) -> None:
    """
    Filter out forensically irrelevant nodes from the graph based on specified criteria.
    Excludes mapping-only labels and labels corresponding to image files.
    """
    nodes = list(G.nodes(data=True))
    nodes_to_remove = []
    batch_size = 100  # Adjust based on the context window and model limits
    cache = {}  # Cache to store previous results

    logging.info("Evaluating nodes for relevancy...")
    for i in tqdm(range(0, len(nodes), batch_size), desc="Filtering nodes"):
        batch = nodes[i:i+batch_size]
        node_labels = [data.get('label', '') for _, data in batch]

        # Initialize lists for labels to evaluate
        labels_to_evaluate = []
        node_ids_to_evaluate = []

        for idx, label in enumerate(node_labels):
            node_id = batch[idx][0]
            # Skip mapping-only labels
            if is_mapping_label(label):
                logging.debug(f"Skipping mapping-only node '{label}' (ID: {node_id}).")
                continue
            # Skip labels corresponding to image files
            if is_image_file(label, file_path_map, G):
                logging.debug(f"Skipping image file node '{label}' (ID: {node_id}).")
                nodes_to_remove.append(node_id)
                continue
            # Proceed to evaluate the label
            if label in cache:
                if cache[label]['is_irrelevant']:
                    nodes_to_remove.append(node_id)
                continue
            else:
                labels_to_evaluate.append(label)
                node_ids_to_evaluate.append(node_id)

        if not labels_to_evaluate:
            continue  # No meaningful labels to evaluate in this batch

        # Build the prompt
        prompt = (
            "You are a forensic analysis assistant.\n"
            "Determine if the following data items are forensically irrelevant based on criteria such as system and application binaries, default system files, common libraries, known good files, and non-user generated content.\n"
            "Respond with a JSON array of objects with keys 'label', 'is_irrelevant' (true or false), and 'reason'.\n\nData Items:\n"
        )
        for label in labels_to_evaluate:
            prompt += f"- {label}\n"
        prompt += "\nEnsure the JSON is properly formatted."

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        text = ""  # Initialize text variable
        try:
            # Make the API call with rate limiting and retry
            text = safe_create_chat_completion(client, "gpt-4o", messages, 2000)
            logging.debug(f"Raw response text: {text}")  # Log the raw response
            results = json.loads(text)
            # Update cache and nodes_to_remove
            for result in results:
                label = result.get('label')
                is_irrelevant = result.get('is_irrelevant', False)
                reason = result.get('reason', '')
                cache[label] = {'is_irrelevant': is_irrelevant, 'reason': reason}
                if is_irrelevant:
                    try:
                        idx = labels_to_evaluate.index(label)
                        node_id = node_ids_to_evaluate[idx]
                        nodes_to_remove.append(node_id)
                        logging.info(f"Removing node '{label}' (ID: {node_id}): {reason}")
                    except ValueError:
                        logging.warning(f"Label '{label}' not found in the current batch.")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {e.msg}")
            logging.error(f"Raw response text: {text}")  # Log the raw response
            logging.debug("Exception details:", exc_info=True)  # Log stack trace
            continue
        except Exception as e:
            logging.error(f"Error evaluating nodes: {str(e)}")
            logging.debug("Exception details:", exc_info=True)  # Log stack trace
            continue

    # Remove irrelevant nodes from the graph
    G.remove_nodes_from(nodes_to_remove)
    logging.info(f"Removed {len(nodes_to_remove)} irrelevant nodes from the graph.")

def interpret_question(client: OpenAI, question: str) -> Dict:
    """
    Use the AI model to interpret the user's question and extract entity descriptions and intent.
    """
    logging.info("Interpreting the user's question...")

    prompt = f"""You are a helpful assistant that extracts information from questions. Please analyze the following question and extract the relevant entity descriptions and intent. Provide the answer in JSON format with the following keys:
- 'entity_descriptions': a list of descriptions of entities mentioned in the question.
- 'intent': a brief description of what the user wants to find out.
- 'direction': 'bidirectional', 'forward', or 'backward' indicating the search direction.

Question: "{question}"

Ensure the JSON is properly formatted."""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = ""  # Initialize text variable
    try:
        # Make the API call with rate limiting and retry
        text = safe_create_chat_completion(client, "o1-mini-2024-09-12", messages, 500)
        logging.debug(f"Raw response text: {text}")  # Log the raw response
        if not text.strip():
            logging.error("Received empty response from OpenAI API.")
            sys.exit(1)
        result = json.loads(text)
        logging.info("Question interpreted successfully.")
        return result
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON response: {e.msg}")
        logging.error(f"Raw response text: {text}")  # Log the raw response
        logging.debug("Exception details:", exc_info=True)  # Log stack trace
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error interpreting the question: {str(e)}")
        logging.debug("Exception details:", exc_info=True)  # Log stack trace
        sys.exit(1)

def find_matching_nodes(client: OpenAI, G: nx.DiGraph, descriptions: List[str]) -> List[str]:
    """
    Use the AI model to find nodes whose labels match the given descriptions or can be used to derive the desired information.
    """
    node_labels = [data['label'] for _, data in G.nodes(data=True)]
    node_ids = [node for node, _ in G.nodes(data=True)]

    matching_nodes = set()
    batch_size = 250  # Adjust batch size as needed
    cache = {}  # Cache to store previous matches

    for description in descriptions:
        logging.info(f"Finding matching nodes for description: '{description}'")
        labels_to_evaluate = []
        labels_indices = []

        # Check cache first
        for idx, label in enumerate(node_labels):
            if (description, label) in cache:
                if cache[(description, label)]:
                    matching_nodes.add(node_ids[idx])
                continue
            else:
                labels_to_evaluate.append(label)
                labels_indices.append(idx)

        if not labels_to_evaluate:
            continue  # All labels have been evaluated

        for i in tqdm(range(0, len(labels_to_evaluate), batch_size), desc=f"Processing batches for '{description}'"):
            batch_labels = labels_to_evaluate[i:i+batch_size]
            batch_indices = labels_indices[i:i+batch_size]
            labels_str = "\n".join(batch_labels)

            prompt = f"""You are a helpful assistant that matches descriptions to labels, considering direct matches and related data that can help derive the desired information. Given the following list of labels:
{labels_str}

Identify all labels that match or are closely related to the description: '{description}'. Consider data that can be directly used to derive or obtain the desired information.

Provide a list of matching labels."""

            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            text = ""  # Initialize text variable
            try:
                # Make the API call with rate limiting and retry
                text = safe_create_chat_completion(client, "o1-mini-2024-09-12", messages, 1000)
                logging.debug(f"Raw response text: {text}")  # Log the raw response
                if not text.strip():
                    logging.error("Received empty response from OpenAI API.")
                    continue
                matched_labels = [line.strip('- ').strip() for line in text.strip().split('\n') if line.strip()]
                for label in matched_labels:
                    if label in batch_labels:
                        idx = batch_labels.index(label)
                        node_id = node_ids[batch_indices[idx]]
                        matching_nodes.add(node_id)
                        cache[(description, label)] = True
                # Update cache for non-matched labels
                unmatched_labels = set(batch_labels) - set(matched_labels)
                for label in unmatched_labels:
                    cache[(description, label)] = False
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON response: {e.msg}")
                logging.error(f"Raw response text: {text}")  # Log the raw response
                logging.debug("Exception details:", exc_info=True)  # Log stack trace
                continue
            except Exception as e:
                logging.error(f"Error matching labels: {str(e)}")
                logging.debug("Exception details:", exc_info=True)  # Log stack trace
                continue
    logging.info(f"Found {len(matching_nodes)} matching nodes.")
    return list(matching_nodes)

def evaluate_node_relevance_batch(client: OpenAI, node_labels: List[str], contexts: List[str]) -> List[Tuple[bool, float]]:
    """
    Use the AI model to evaluate the forensic relevance of a batch of nodes.
    Returns a list of tuples (is_relevant, confidence_score).
    """
    results = []
    batch_size = 50  # Adjust based on context window
    cache = {}  # Cache to store previous evaluations

    for i in tqdm(range(0, len(node_labels), batch_size), desc="Evaluating node relevance"):
        batch_labels = node_labels[i:i+batch_size]
        batch_contexts = contexts[i:i+batch_size]
        labels_to_evaluate = []
        contexts_to_evaluate = []
        indices = []

        for idx, label in enumerate(batch_labels):
            if label in cache:
                results.append(cache[label])
            else:
                labels_to_evaluate.append(label)
                contexts_to_evaluate.append(batch_contexts[idx])
                indices.append(idx)

        if not labels_to_evaluate:
            continue

        # Build the prompt
        prompt = (
            "You are a forensic analyst assessing the relevance of data in investigations.\n"
            "Assess the following data items in the context of a forensic investigation. For each item, provide a JSON object with keys 'label', 'is_relevant' (true or false), and 'confidence' (integer between 0 and 100).\n\nData Items:\n"
        )
        for label, context in zip(labels_to_evaluate, contexts_to_evaluate):
            prompt += f"- Label: {label}\n  Context: {context}\n"

        prompt += "\nEnsure the JSON is properly formatted."

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        text = ""  # Initialize text variable
        try:
            # Make the API call with rate limiting and retry
            text = safe_create_chat_completion(client, "o1-mini-2024-09-12", messages, 2000)
            logging.debug(f"Raw response text: {text}")  # Log the raw response
            if not text.strip():
                logging.error("Received empty response from OpenAI API.")
                # In case of error, assume not relevant
                for label in labels_to_evaluate:
                    cache[label] = (False, 0)
                    results.append((False, 0))
                continue
            batch_results = json.loads(text)
            for result in batch_results:
                label = result.get('label')
                is_relevant = result.get('is_relevant', False)
                confidence = result.get('confidence', 0)
                cache[label] = (is_relevant, confidence)
                results.append((is_relevant, confidence))
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {e.msg}")
            logging.error(f"Raw response text: {text}")  # Log the raw response
            logging.debug("Exception details:", exc_info=True)  # Log stack trace
            # In case of error, assume not relevant
            for label in labels_to_evaluate:
                cache[label] = (False, 0)
                results.append((False, 0))
        except Exception as e:
            logging.error(f"Error evaluating node relevance: {str(e)}")
            logging.debug("Exception details:", exc_info=True)  # Log stack trace
            # In case of error, assume not relevant
            for label in labels_to_evaluate:
                cache[label] = (False, 0)
                results.append((False, 0))

    return results

def evaluate_path_relevance(client: OpenAI, G: nx.DiGraph, path: List[str]) -> Tuple[float, bool]:
    """
    Evaluate the relevance of a path and calculate the aggregate confidence.
    Returns the aggregate confidence and a boolean indicating if the path is valid.
    """
    node_labels = []
    contexts = []
    for node in path:
        node_label = G.nodes[node]['label']
        # Include context (adjacent nodes' labels and edge files)
        neighbors = list(G.successors(node)) + list(G.predecessors(node))
        neighbor_labels = [G.nodes[n]['label'] for n in neighbors if n != node]
        # Get edge file paths
        edges = list(G.edges(node)) + list(G.in_edges(node))
        edge_files = []
        for edge in edges:
            data = G.get_edge_data(*edge)
            edge_files.extend(data.get('files', []))
        context = f"Neighbors: {', '.join(neighbor_labels)}; Files: {', '.join(edge_files)}"
        node_labels.append(node_label)
        contexts.append(context)

    # Evaluate nodes in batch
    relevancies = evaluate_node_relevance_batch(client, node_labels, contexts)
    aggregate_confidence = 100
    for is_relevant, confidence in relevancies:
        if not is_relevant or confidence < 50:
            return 0, False  # Discard the path if any node is not relevant enough
        aggregate_confidence = min(aggregate_confidence, confidence)
    return aggregate_confidence, True

def depth_limited_search(G: nx.DiGraph, start_node: str, max_depth: int = 5, reverse: bool = False) -> List[List[str]]:
    """
    Perform a depth-limited search from the start_node.
    If reverse is True, search predecessors instead of successors.
    """
    paths = []
    visited = set()
    stack = [(start_node, [start_node], 0)]
    while stack:
        (vertex, path, depth) = stack.pop()
        if depth >= max_depth:
            continue
        if reverse:
            neighbors = G.predecessors(vertex)
        else:
            neighbors = G.successors(vertex)
        for next_node in neighbors:
            if next_node not in visited:
                visited.add(next_node)
                new_path = path + [next_node]
                stack.append((next_node, new_path, depth + 1))
                paths.append(new_path)
    return paths

def search_graph(client: OpenAI, G: nx.DiGraph, descriptions: List[str], direction: str, max_depth: int = 5) -> List[Dict]:
    """
    Search the graph based on the entity descriptions and direction provided.
    """
    # Find matching nodes
    matching_nodes = find_matching_nodes(client, G, descriptions)
    if not matching_nodes:
        logging.warning("No matching nodes found for the given descriptions.")
        return []

    # Depending on the direction, set search parameters
    if direction == 'forward':
        reverse = False
    elif direction == 'backward':
        reverse = True
    else:
        reverse = None  # Bidirectional

    results = []
    # If bidirectional, search in both directions
    directions = [False, True] if reverse is None else [reverse]

    for dir in directions:
        dir_str = 'backward' if dir else 'forward'
        logging.info(f"Performing {dir_str} search...")
        for start_node in matching_nodes:
            paths = depth_limited_search(G, start_node, max_depth=max_depth, reverse=dir)
            logging.info(f"Evaluating {len(paths)} paths from node {start_node}...")
            # Evaluate paths in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_path = {executor.submit(evaluate_path_relevance, client, G, path): path for path in paths}
                for future in concurrent.futures.as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        confidence, is_valid = future.result()
                        if is_valid:
                            result = {
                                'path': path,
                                'start_label': G.nodes[start_node]['label'],
                                'end_label': G.nodes[path[-1]]['label'],
                                'confidence': confidence,
                                'direction': dir_str
                            }
                            results.append(result)
                    except Exception as e:
                        logging.error(f"Error evaluating path: {str(e)}")
                        logging.debug("Exception details:", exc_info=True)  # Log stack trace
    logging.info(f"Search complete. Found {len(results)} valid paths.")
    return results

def output_results(results: List[Dict], G: nx.DiGraph) -> None:
    """
    Output the results to JSON files.
    """
    logging.info("Outputting results...")
    for idx, result in enumerate(results, 1):
        output_data = {
            'path': result['path'],
            'nodes': [G.nodes[node]['label'] for node in result['path']],
            'confidence': result['confidence'],
            'direction': result['direction']
        }
        start_label = re.sub(r'\W+', '_', result['start_label'])
        end_label = re.sub(r'\W+', '_', result['end_label'])
        filename = f"Json_{idx}_{start_label}_to_{end_label}_{result['confidence']}%.json"
        try:
            with open(filename, 'w') as f:
                json.dump(output_data, f, indent=4)
            logging.info(f"Result saved to {filename}")
        except Exception as e:
            logging.error(f"Failed to save result to {filename}: {str(e)}")
            logging.debug("Exception details:", exc_info=True)  # Log stack trace

def main():
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Visualize graph data and query with AI.")
    parser.add_argument(
        '--string_dict',
        type=str,
        default='string_dictionary.json',
        help='Path to the string dictionary JSON file (default: string_dictionary.json)'
    )
    parser.add_argument(
        '--edge_csv',
        type=str,
        default='edges.csv',
        help='Path to the edges CSV file (default: edges.csv)'
    )
    parser.add_argument(
        '--file_path_dict',
        type=str,
        default='file_path_dictionary.json',
        help='Path to the file path dictionary JSON file (default: file_path_dictionary.json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='graph.html',
        help='Output HTML file for the graph visualization (default: graph.html)'
    )
    args = parser.parse_args()

    # Load graph data
    logging.info("Loading graph data...")
    data = load_graph_data(args.string_dict, args.edge_csv, args.file_path_dict)
    G = create_graph(data)
    file_path_map = data.get('file_path_map', {})

    # Retrieve OpenAI API key
    api_key = get_openai_api_key()
    # Instantiate the OpenAI client as per your original code
    client = OpenAI(api_key=api_key)

    # Filter out irrelevant nodes
    logging.info("Filtering out forensically irrelevant nodes...")
    filter_irrelevant_nodes(client, G, file_path_map)

    # Visualize the graph and save to HTML
    visualize_graph(G, output_file=args.output)

    # Prompt the user for their question
    user_question = input("Please enter your question: ")

    # Proceed if a question is provided
    if not user_question.strip():
        logging.info("No question provided. Exiting after graph visualization.")
        sys.exit(0)

    # Interpret the user's question
    interpretation = interpret_question(client, user_question)
    descriptions = interpretation.get('entity_descriptions', [])
    intent = interpretation.get('intent', '')
    direction = interpretation.get('direction', 'bidirectional')

    if not descriptions:
        logging.warning("Could not extract any entity descriptions from the question.")
        sys.exit(1)

    logging.info(f"Entity Descriptions extracted: {descriptions}")
    logging.info(f"Intent: {intent}")
    logging.info(f"Direction: {direction}")

    # Search the graph based on the interpretation
    results = search_graph(client, G, descriptions, direction)

    # Filter results with confidence >= 50%
    valid_results = [res for res in results if res['confidence'] >= 50]

    # Output the results
    if not valid_results:
        logging.warning("No valid connections found with confidence >= 50%.")
    else:
        output_results(valid_results, G)

        # Print summary
        logging.info("\n--- Results Summary ---\n")
        for idx, result in enumerate(valid_results, 1):
            logging.info(f"Json {idx}: {result['start_label']} to {result['end_label']} ({result['confidence']}% confidence)")

if __name__ == "__main__":
    main()
