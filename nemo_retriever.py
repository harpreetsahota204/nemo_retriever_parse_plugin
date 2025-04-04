import fiftyone as fo
from fiftyone.core.labels import Detections, Detection
import json
import requests
from typing import Dict
from getpass import getpass
from tqdm import tqdm

def create_headers(api_key: str, asset_id: str = None) -> Dict[str, str]:
    """Create headers for NVIDIA API requests
    
    Args:
        api_key: NVIDIA API authentication key
        asset_id: Optional ID of uploaded asset to reference in headers
        
    Returns:
        Dictionary containing required headers for NVIDIA API requests
    """
    # Set base authentication headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }
    
    # Add asset-specific headers if an asset_id is provided
    if asset_id:
        headers.update({
            "Content-Type": "application/json",
            "NVCF-INPUT-ASSET-REFERENCES": asset_id,  # Reference the uploaded asset
            "NVCF-FUNCTION-ASSET-IDS": asset_id       # Specify asset for function call
        })
    return headers

def upload_asset(image_data: bytes, description: str, api_key: str) -> str:
    """Upload image asset to NVIDIA's API
    
    Args:
        image_data: Raw bytes of the image to upload
        description: Text description of the image asset
        api_key: NVIDIA API authentication key
        
    Returns:
        Asset ID string assigned by NVIDIA API
        
    Raises:
        requests.exceptions.HTTPError: If either API request fails
    """
    # First request to get upload URL and asset ID
    auth_response = requests.post(
        "https://api.nvcf.nvidia.com/v2/nvcf/assets",
        headers=create_headers(api_key),
        json={"contentType": "image/jpeg", "description": description},
        timeout=30
    )
    auth_response.raise_for_status()
    auth_data = auth_response.json()
    
    # Second request to upload actual image data to provided URL
    upload_response = requests.put(
        auth_data["uploadUrl"],
        data=image_data,
        headers={
            "x-amz-meta-nvcf-asset-description": description,
            "content-type": "image/jpeg"
        },
        timeout=300  # Longer timeout for file upload
    )
    upload_response.raise_for_status()
    
    return str(auth_data["assetId"])

def process_image(image_path: str, api_key: str) -> Dict:
    """Send image to NeMo API and get response
    
    Args:
        image_path: Path to image file on disk
        api_key: NVIDIA API authentication key
        
    Returns:
        JSON response from NeMo API containing detection results
        
    Raises:
        IOError: If image file cannot be read
        requests.exceptions.HTTPError: If API request fails
    """
    # Upload image file to NVIDIA's asset storage
    with open(image_path, "rb") as f:
        asset_id = upload_asset(f.read(), "Test Image", api_key)
    
    # Construct content array with image reference
    content = [{
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;asset_id,{asset_id}"}
    }]
    
    # Specify markdown_bbox tool for detection
    tool = [{
        "type": "function",
        "function": {"name": "markdown_bbox"}
    }]
    
    # Build complete API request payload
    payload = {
        "tools": tool,
        "model": "nvidia/nemoretriever-parse",
        "messages": [{"role": "user", "content": content}]
    }
    
    # Make API request to process image
    response = requests.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers=create_headers(api_key, asset_id),
        json=payload
    )
    response.raise_for_status()
    
    return response.json()

def parse_nemo_response_to_detections(response: Dict) -> Detections:
    """Convert NeMo API response to FiftyOne detections format
    
    Args:
        response: JSON response from NeMo API
        
    Returns:
        FiftyOne Detections object containing all detected regions
    """
    detections = []
    try:
        # Extract bounding box data from nested response structure
        bbox_data = response['choices'][0]['message']['tool_calls'][0]['function']['arguments']
        elements = json.loads(bbox_data)[0]
        
        # Process each detected element
        for elem in elements:
            # Convert bounding box coordinates to [x, y, width, height] format
            bbox = [
                float(elem['bbox']['xmin']),
                float(elem['bbox']['ymin']),
                float(elem['bbox']['xmax']) - float(elem['bbox']['xmin']),  # Convert to width
                float(elem['bbox']['ymax']) - float(elem['bbox']['ymin'])   # Convert to height
            ]
            
            # Create Detection object with element data
            detection = Detection(
                label=elem['type'],
                bounding_box=bbox,
                text=elem['text']
            )
            detections.append(detection)
            
    except Exception as e:
        print(f"Error parsing response: {e}")
    
    return Detections(detections=detections)

def run_nemo_parse(dataset: fo.Dataset, api_key: str):
    """Process dataset with NeMo Retriever Parse and add detections and token usage fields.
    
    This function processes each image in the dataset through the NeMo API and adds
    the detection results and token usage statistics as new fields to each sample.
    
    Args:
        dataset: FiftyOne dataset to process
        api_key: NVIDIA API authentication key
        
    The following fields will be added to each sample:
        - nemo_detections: Detected regions with bounding boxes and text
        - nemo_prompt_tokens: Number of prompt tokens used
        - nemo_completion_tokens: Number of completion tokens used  
        - nemo_total_tokens: Total tokens used
        
    Note:
        If processing fails for any image, empty/zero values will be added to maintain
        alignment with the dataset.
    """
    # Get all image filepaths from dataset
    filepaths = dataset.values("filepath")
    
    # Initialize lists to store results
    all_detections = []
    prompt_tokens = []
    completion_tokens = []
    total_tokens = []
    
    # Process each image in the dataset
    for filepath in tqdm(filepaths, desc="Processing images"):
        try:
            # Get API response for current image
            response = process_image(filepath, api_key)
            
            # Extract and store detections
            detections = parse_nemo_response_to_detections(response)
            all_detections.append(detections)
            
            # Extract and store token usage statistics
            usage = response.get('usage', {})
            prompt_tokens.append(usage.get('prompt_tokens', 0))
            completion_tokens.append(usage.get('completion_tokens', 0))
            total_tokens.append(usage.get('total_tokens', 0))
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            # Add empty/zero values to maintain list alignment
            all_detections.append(Detections())
            prompt_tokens.append(0)
            completion_tokens.append(0)
            total_tokens.append(0)
    
    # Add all computed fields to the dataset
    dataset.set_values("nemo_detections", all_detections)
    dataset.set_values("nemo_prompt_tokens", prompt_tokens)
    dataset.set_values("nemo_completion_tokens", completion_tokens)
    dataset.set_values("nemo_total_tokens", total_tokens)