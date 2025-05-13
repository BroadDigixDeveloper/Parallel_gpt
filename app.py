from flask import Flask, request, jsonify, send_file
import json
import os
import time
import requests
import random
from pathlib import Path
import threading
import queue
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging system
def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # File handler with rotation (10MB max size, keep 5 backup files)
    file_handler = RotatingFileHandler(
        log_dir / "gpt_processor.log", 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

app = Flask(__name__)

# Configuration
API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-4o"
MAX_CONCURRENT_REQUESTS = 10
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0
MIN_COMPLETE_SIZE_KB = 2  # Minimum size in KB to consider a result complete

# File system paths
RESULTS_DIR = Path("gpt_results")
RESULTS_DIR.mkdir(exist_ok=True)

# In-memory processing queue
processing_jobs = {}  # Jobs being processed
request_queue = queue.Queue()
worker_running = False

def prepare_gpt_request(json_data):
    """Prepare the GPT request from the JSON data"""
    start_time = time.time()
    logger.info(f"Preparing GPT request for URL: {json_data.get('url', 'Unknown URL')}")
    
    try:
        # Default system message as per client example
        system_message = {
            "role": "system",
            "content": (
                "Format for a light editing environment with no bolding, indenting, or markdown. "
                "Use only standard text characters.\n\n"
                "Target User: The target user is the leader who is leading this change initiative.\n\n"
                "The goal is for them to feel, think, and do the following:\n\n"
                "Feel - I want them to feel clear and supported in completing the instructions.\n\n"
                "Think - I want them to think that completing these instructions is crucial to the success of the initiative.\n\n"
                "Do - I want them to promptly complete this action according to the instructions provided, ensuring accuracy and attention to detail."
            )
        }

        # Extract the website data from the scrapped JSON
        website_data = json_data.get("result", {})
        
        # Extract key information
        site_title = website_data.get("page_info", {}).get("title", "Unknown Site")
        site_url = json_data.get("url", "Unknown URL")
        logger.info(f"Processing website: {site_title} - {site_url}")
        
        # Extract main content
        paragraphs = website_data.get("main_content", {}).get("paragraphs", [])
        headings = website_data.get("main_content", {}).get("headings", {})
        h1s = headings.get("h1", [])
        h2s = headings.get("h2", [])
        
        # Log content statistics
        logger.info(f"Content stats - Paragraphs: {len(paragraphs)}, H1s: {len(h1s)}, H2s: {len(h2s)}")
        
        # Extract navigation links
        navigation = website_data.get("header", {}).get("navigation", [])
        nav_items = [f"{item.get('text', '')}: {item.get('url', '')}" for item in navigation if item.get('text')]
        
        # Extract forms
        forms = website_data.get("forms", [])
        form_fields = []
        for form in forms:
            for field in form.get("fields", []):
                field_info = f"{field.get('type', 'field')}: {field.get('name', '')} - {field.get('placeholder', '')}"
                form_fields.append(field_info)
        
        logger.info(f"Additional content - Navigation items: {len(nav_items)}, Forms: {len(forms)}, Form fields: {len(form_fields)}")
        
        # Construct content to analyze
        content_sections = []
        
        if h1s:
            content_sections.append("# Main Headings:\n" + "\n".join(h1s))
        
        if h2s:
            content_sections.append("# Sub Headings:\n" + "\n".join(h2s))
        
        if paragraphs:
            content_sections.append("# Main Content:\n" + "\n".join(paragraphs))  
        
        if nav_items:
            content_sections.append("# Navigation Links:\n" + "\n".join(nav_items))
        
        if form_fields:
            content_sections.append("# Form Fields:\n" + "\n".join(form_fields))  # All form fields
        
        content_text = "\n\n".join(content_sections)
        # Log content text size
        content_size = len(content_text)
        logger.info(f"Prepared content text size: {content_size} characters")
        
        # Create user message with the content to be analyzed using client's preferred format
        user_message = {
            "role": "user",
            "content": (
                f"[Style]\nUnspecified\n\n"
                f"[Writing Perspective]\n"
                f"Third-person Objective: Ensure the responses are neutral and objective, avoiding personal interpretations or perspectives. "
                f"Maintain a professional tone suitable for formal documentation.\n\n"
                f"[Output Format]\nDefault: Unspecified\n\n"
                f"[Actions]\n"
                f"Here is a list of actions for ChatGPT to perform:\n\n"
                f"Objective:\nGenerate a detailed analysis of the website.\n\n"
                f"Instructions:\n\n"
                f"1. Present the header as follows:\nWebsite Analysis Summary for {site_title}\n\n"
                f"2. Analyze the website at {site_url} based on the scraped content.\n\n"
                f"3. Address these topics:\n"
                f"- Primary Purpose of the Website\n"
                f"- Target Audience\n"
                f"- Key Products or Services\n"
                f"- Content Organization\n"
                f"- Main Value Propositions\n"
                f"- Call to Actions\n"
                f"- Overall User Experience\n\n"
                f"4. Proceed topic by topic. For each include 2 to 5 paragraphs with key insights.\n\n"
                f"5. Based on the scraped content provided below:\n\n"
                f"{content_text}"
            )
        }
        
        # Create the full request payload
        request_data = {
            "model": MODEL,
            "messages": [system_message, user_message],
            "temperature": 0.2,
            "max_tokens": 2000,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.1,
            "response_format": {"type": "text"}
        }
        
        # Log request size
        request_size = len(json.dumps(request_data))
        logger.info(f"GPT request prepared successfully. Size: {request_size} bytes. Time taken: {time.time() - start_time:.2f}s")
        
        return request_data
    
    except Exception as e:
        logger.error(f"Error preparing GPT request: {str(e)}", exc_info=True)
        return None

def call_openai_with_retry(request_data, job_id, max_retries=MAX_RETRIES, initial_backoff=INITIAL_BACKOFF):
    """Call OpenAI API with exponential backoff retry logic"""
    retry_count = 0
    backoff = initial_backoff
    
    while retry_count <= max_retries:
        try:
            # If not the first attempt, log retry
            if retry_count > 0:
                logger.info(f"Retry attempt {retry_count}/{max_retries} for job {job_id} with backoff {backoff:.2f}s")
                time.sleep(backoff)
                
            response = requests.post(
                OPENAI_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}"
                },
                json=request_data,
                timeout=60  # Add timeout to prevent hanging requests
            )
            
            # Return successful responses and non-retriable errors immediately
            if response.status_code == 200 or (400 <= response.status_code < 500 and response.status_code != 429):
                return response
                
            # For 5xx errors or rate limits (429), retry with backoff
            if response.status_code >= 500 or response.status_code == 429:
                logger.warning(f"Job {job_id}: OpenAI API returned status {response.status_code}, retrying...")
            else:
                # Other errors, don't retry
                return response
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Job {job_id}: Request exception: {str(e)}, retrying...")
        
        # Update retry count and backoff (exponential backoff with jitter)
        retry_count += 1
        # Add jitter to prevent thundering herd
        jitter = 0.8 + 0.4 * random.random()  # Random value between 0.8 and 1.2
        backoff = backoff * 2 * jitter
    
    # If we get here, we've exhausted all retries
    logger.error(f"Job {job_id}: Failed after {max_retries} retry attempts")
    
    # Create a mock response object for the error case
    class MockResponse:
        def __init__(self):
            self.status_code = 500
            self.text = f"Failed after {max_retries} retry attempts"
            
        def json(self):
            return {"error": {"message": self.text}}
    
    return MockResponse()

def process_job(job_id):
    """Process a single job"""
    start_time = time.time()
    logger.info(f"Starting to process job {job_id}")
    
    try:
        if job_id not in processing_jobs:
            logger.warning(f"Job {job_id} not found in processing queue")
            return
            
        job = processing_jobs[job_id]
        job["status"] = "processing"
        url = job.get("json_data", {}).get("url", "Unknown URL")
        logger.info(f"Processing job {job_id} for URL: {url}")
        
        # Log request time
        api_start_time = time.time()
        logger.info(f"Sending request to OpenAI API for job {job_id}")
        
        # Call the OpenAI API with retry logic
        response = call_openai_with_retry(job["gpt_request"], job_id)
        
        api_time = time.time() - api_start_time
        logger.info(f"OpenAI API response received for job {job_id}. Time taken: {api_time:.2f}s. Status code: {response.status_code}")
        
        # Check if the request was successful
        if response.status_code == 200:
            try:
                response_json = response.json()
                content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                token_usage = response_json.get("usage", {})
                
                logger.info(f"Job {job_id} completed successfully. Tokens used - Prompt: {token_usage.get('prompt_tokens', 'N/A')}, "
                            f"Completion: {token_usage.get('completion_tokens', 'N/A')}, "
                            f"Total: {token_usage.get('total_tokens', 'N/A')}")
                
                # Create result JSON
                result = {
                    "job_id": job_id,
                    "status": "completed",
                    "submitted_at": job["submitted_at"],
                    "completed_at": time.time(),
                    "processing_time": time.time() - job["submitted_at"],
                    "api_time": api_time,
                    "token_usage": token_usage,
                    "analysis": content
                }
                
                # Save to file
                result_path = RESULTS_DIR / f"{job_id}.json"
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2)
                
                # Update job status
                job["status"] = "completed"
            
            except Exception as e:
                logger.error(f"Error parsing OpenAI API response for job {job_id}: {str(e)}", exc_info=True)
                raise
                
        else:
            # Handle error - IMPROVED ERROR HANDLING
            error_message = "Unknown error"
            retriable_error = response.status_code >= 500 or response.status_code == 429
            
            # Safely try to parse error message from response
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", "Unknown error")
            except Exception as e:
                # If response is not valid JSON, use the response text or status code
                if response.text:
                    error_message = f"API Error: {response.status_code} - {response.text[:200]}"
                else:
                    error_message = f"API Error: {response.status_code}"
                
                logger.error(f"Could not parse error response from OpenAI API for job {job_id}: {str(e)}")
            
            logger.error(f"OpenAI API error for job {job_id}: {error_message}")
            
            # Create error JSON
            result = {
                "job_id": job_id,
                "status": "error",
                "submitted_at": job["submitted_at"],
                "completed_at": time.time(),
                "processing_time": time.time() - job["submitted_at"],
                "api_time": api_time,
                "error": error_message,
                "retriable": retriable_error
            }
            
            # Save to file
            result_path = RESULTS_DIR / f"{job_id}.json"
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Update job status
            job["status"] = "error"
            
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Error processing job {job_id}: {str(e)}", exc_info=True)
        
        # Create error JSON
        result = {
            "job_id": job_id,
            "status": "error",
            "submitted_at": job["submitted_at"] if job_id in processing_jobs and "submitted_at" in processing_jobs[job_id] else time.time(),
            "completed_at": time.time(),
            "processing_time": time.time() - (processing_jobs[job_id]["submitted_at"] if job_id in processing_jobs and "submitted_at" in processing_jobs[job_id] else time.time()),
            "error": str(e)
        }
        
        # Save to file
        result_path = RESULTS_DIR / f"{job_id}.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Update job status if it exists
        if job_id in processing_jobs:
            processing_jobs[job_id]["status"] = "error"
    
    finally:
        # Remove from processing queue when done
        if job_id in processing_jobs:
            del processing_jobs[job_id]
        
        total_time = time.time() - start_time
        logger.info(f"Finished processing job {job_id}. Total processing time: {total_time:.2f}s")

def worker():
    """Background worker to process queued requests"""
    global worker_running
    worker_running = True
    
    logger.info("Background worker thread started")
    
    while worker_running:
        try:
            # Get up to MAX_CONCURRENT_REQUESTS jobs from the queue
            queue_size = request_queue.qsize()
            if queue_size > 0:
                logger.info(f"Worker found {queue_size} jobs in queue")
            
            active_threads = []
            for _ in range(min(MAX_CONCURRENT_REQUESTS, queue_size)):
                try:
                    job_id = request_queue.get(block=False)
                    logger.info(f"Starting new thread for job {job_id}")
                    # Start a thread for each job
                    thread = threading.Thread(target=process_job, args=(job_id,), name=f"Job-{job_id}")
                    thread.start()
                    active_threads.append(thread)
                    request_queue.task_done()
                except queue.Empty:
                    break  # No more jobs in queue
            
            # Wait for all threads to complete
            for thread in active_threads:
                thread.join()
                logger.info(f"Thread for {thread.name} completed")
                
            # Sleep briefly if no jobs were found
            if not active_threads:
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Worker error: {str(e)}", exc_info=True)
            time.sleep(1)

@app.route('/api/submit', methods=['POST'])
def submit_job():
    """Submit a new job with a JSON file"""
    start_time = time.time()
    
    try:
        # Get job_id from request
        job_id = request.args.get('job_id')
        
        # Generate a job_id if not provided
        if not job_id:
            job_id = f"job_{int(time.time() * 1000)}"
        
        client_ip = request.remote_addr
        logger.info(f"New job submission from {client_ip}. Job ID: {job_id}")
        
        # Check request size
        content_length = request.content_length
        logger.info(f"Job {job_id} request size: {content_length} bytes")
        
        # Check if job already exists
        if job_id in processing_jobs or (RESULTS_DIR / f"{job_id}.json").exists():
            logger.warning(f"Job ID '{job_id}' already exists. Request rejected.")
            return jsonify({"error": f"Job ID '{job_id}' already exists"}), 400
        
        # Get JSON data from request
        if not request.is_json:
            logger.warning(f"Job {job_id} rejected: Request must contain JSON data")
            return jsonify({"error": "Request must contain JSON data"}), 400
        
        json_data = request.json
        logger.info(f"Job {job_id} data received for URL: {json_data.get('url', 'Unknown URL')}")
        
        # Prepare GPT request
        logger.info(f"Preparing GPT request for job {job_id}")
        gpt_request = prepare_gpt_request(json_data)
        if not gpt_request:
            logger.error(f"Failed to prepare GPT request for job {job_id}")
            return jsonify({"error": "Failed to prepare GPT request from JSON data"}), 400
        
        # Create job
        processing_jobs[job_id] = {
            "status": "queued",
            "submitted_at": time.time(),
            "json_data": json_data,
            "gpt_request": gpt_request
        }
        
        # Add to queue
        request_queue.put(job_id)
        logger.info(f"Job {job_id} added to processing queue. Queue size: {request_queue.qsize()}")
        
        processing_time = time.time() - start_time
        logger.info(f"Job {job_id} submission processing completed in {processing_time:.2f}s")
        
        return jsonify({
            "job_id": job_id,
            "status": "queued",
            "message": "Job submitted successfully"
        })
        
    except Exception as e:
        logger.error(f"Error submitting job: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error submitting job: {str(e)}"}), 500

@app.route('/api/status', methods=['GET'])
def get_job_status():
    """Get the status of a job"""
    start_time = time.time()
    job_id = request.args.get('job_id')
    client_ip = request.remote_addr
    
    logger.info(f"Status check from {client_ip} for job {job_id}")
    
    if not job_id:
        logger.warning("Status check rejected: No job_id provided")
        return jsonify({"error": "No job_id provided"}), 400
    
    # Check if job is in processing
    if job_id in processing_jobs:
        status = processing_jobs[job_id]["status"]
        logger.info(f"Status for job {job_id}: {status} (in memory)")
        processing_time = time.time() - start_time
        logger.info(f"Status check completed in {processing_time:.2f}s")
        return jsonify({
            "job_id": job_id,
            "status": status
        })
    
    # Check if result file exists
    result_path = RESULTS_DIR / f"{job_id}.json"
    if result_path.exists():
        try:
            with open(result_path, 'r') as f:
                result = json.load(f)
            status = result.get("status", "unknown")
            logger.info(f"Status for job {job_id}: {status} (from file)")
            processing_time = time.time() - start_time
            logger.info(f"Status check completed in {processing_time:.2f}s")
            return jsonify({
                "job_id": job_id,
                "status": status
            })
        except Exception as e:
            logger.error(f"Error reading result file for job {job_id}: {str(e)}", exc_info=True)
            return jsonify({"error": f"Error reading result file: {str(e)}"}), 500
    
    logger.warning(f"Job {job_id} not found")
    processing_time = time.time() - start_time
    logger.info(f"Status check completed in {processing_time:.2f}s")
    return jsonify({"error": f"Job {job_id} not found"}), 404

@app.route('/api/retrieve', methods=['GET'])
def retrieve_job():
    """Retrieve the result of a job"""
    start_time = time.time()
    job_id = request.args.get('job_id')
    client_ip = request.remote_addr
    
    logger.info(f"Result retrieval request from {client_ip} for job {job_id}")
    
    if not job_id:
        logger.warning("Retrieval rejected: No job_id provided")
        return jsonify({"error": "No job_id provided"}), 400
    
    # Check if result file exists
    result_path = RESULTS_DIR / f"{job_id}.json"
    if result_path.exists():
        try:
            # Get file size
            file_size = result_path.stat().st_size
            file_size_kb = file_size / 1024
            logger.info(f"Retrieving result file for job {job_id}. File size: {file_size_kb:.2f}KB")
            
            # Read the result file
            with open(result_path, 'r') as f:
                result = json.load(f)
            
            # Check if we should return this as a completed result
            if file_size_kb < MIN_COMPLETE_SIZE_KB and result.get("status") != "completed":
                logger.info(f"Job {job_id} result file is smaller than {MIN_COMPLETE_SIZE_KB}KB, treating as still processing")
                processing_time = time.time() - start_time
                
                # Return minimal response for jobs that are not complete
                minimal_response = {
                    "job_id": job_id,
                    "status": "processing",
                    "submitted_at": result.get("submitted_at", time.time()),
                    "message": f"Job is still processing (result size: {file_size_kb:.2f}KB)"
                }
                
                logger.info(f"Retrieval request completed in {processing_time:.2f}s")
                return jsonify(minimal_response)
            
            # Return the complete result
            processing_time = time.time() - start_time
            logger.info(f"Result retrieval completed in {processing_time:.2f}s")
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error reading result file for job {job_id}: {str(e)}", exc_info=True)
            return jsonify({"error": f"Error reading result file: {str(e)}"}), 500
    
    # Check if job is still processing
    if job_id in processing_jobs:
        status = processing_jobs[job_id]["status"]
        
        # Create a minimal response for jobs that are still processing
        minimal_response = {
            "job_id": job_id,
            "status": status,
            "submitted_at": processing_jobs[job_id].get("submitted_at", time.time()),
            "message": "Job is still processing"
        }
        
        logger.info(f"Retrieval for job {job_id} - Still processing with status: {status}")
        processing_time = time.time() - start_time
        logger.info(f"Retrieval request completed in {processing_time:.2f}s")
        
        # Return the minimal response
        return jsonify(minimal_response)
    
    logger.warning(f"Job {job_id} not found for retrieval")
    processing_time = time.time() - start_time
    logger.info(f"Retrieval request completed in {processing_time:.2f}s")
    return jsonify({"error": f"Job {job_id} not found"}), 404

@app.route('/api/delete', methods=['GET'])
def delete_job():
    """Delete a job result"""
    start_time = time.time()
    job_id = request.args.get('job_id')
    client_ip = request.remote_addr
    
    logger.info(f"Delete request from {client_ip} for job {job_id}")
    
    if not job_id:
        logger.warning("Delete rejected: No job_id provided")
        return jsonify({"error": "No job_id provided"}), 400
    
    # Check if result file exists
    result_path = RESULTS_DIR / f"{job_id}.json"
    if result_path.exists():
        try:
            result_path.unlink()
            logger.info(f"Job {job_id} result file deleted successfully")
            processing_time = time.time() - start_time
            logger.info(f"Delete request completed in {processing_time:.2f}s")
            return jsonify({
                "job_id": job_id,
                "status": "deleted",
                "message": "Job deleted successfully"
            })
        except Exception as e:
            logger.error(f"Error deleting result file for job {job_id}: {str(e)}", exc_info=True)
            return jsonify({"error": f"Error deleting job: {str(e)}"}), 500
    
    # Check if job is still processing
    if job_id in processing_jobs:
        del processing_jobs[job_id]
        logger.info(f"Job {job_id} cancelled while in processing queue")
        processing_time = time.time() - start_time
        logger.info(f"Delete request completed in {processing_time:.2f}s")
        return jsonify({
            "job_id": job_id,
            "status": "cancelled",
            "message": "Job cancelled successfully"
        })
    
    logger.warning(f"Job {job_id} not found for deletion")
    processing_time = time.time() - start_time
    logger.info(f"Delete request completed in {processing_time:.2f}s")
    return jsonify({"error": f"Job {job_id} not found"}), 404

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    logger.info("Health check request received")
    
    # Get system status
    queue_size = request_queue.qsize()
    active_jobs = len(processing_jobs)
    
    response = {
        "status": "healthy",
        "queue_size": queue_size,
        "active_jobs": active_jobs,
        "worker_running": worker_running,
        "model": MODEL,
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "max_retries": MAX_RETRIES,
        "min_complete_size_kb": MIN_COMPLETE_SIZE_KB
    }
    
    logger.info(f"Health check response: Queue size: {queue_size}, Active jobs: {active_jobs}")
    return jsonify(response)

if __name__ == '__main__':
    # Setup logging
    logger.info("Starting GPT Website Analysis Service")
    logger.info(f"Using model: {MODEL}")
    logger.info(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    logger.info(f"Max retries: {MAX_RETRIES}")
    logger.info(f"Results directory: {RESULTS_DIR}")
    
    # Start the worker thread
    logger.info("Starting background worker thread")
    worker_thread = threading.Thread(target=worker, daemon=True, name="WorkerThread")
    worker_thread.start()
    
    # Run the app
    logger.info("Starting Flask web server on port 8081")
    app.run(host='0.0.0.0', port=8081, debug=False)