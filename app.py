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
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.lib.colors import black, blue, darkblue
import re
from html import unescape
from io import BytesIO

# Load environment variables
load_dotenv()

# Configure logging system
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

app = Flask(__name__)

# Configuration
API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o"
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9
DEFAULT_FREQUENCY_PENALTY = 0.0
DEFAULT_PRESENCE_PENALTY = 0.1
MAX_CONCURRENT_REQUESTS = 10
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0
MIN_COMPLETE_SIZE_KB = 2

# File system paths
RESULTS_DIR = Path("gpt_results")
PDF_DIR = Path("gpt_pdfs")
RESULTS_DIR.mkdir(exist_ok=True)
PDF_DIR.mkdir(exist_ok=True)

# In-memory processing queue
processing_jobs = {}  # Child jobs being processed
parent_jobs = {}  # Parent job tracking
request_queue = queue.Queue()
worker_running = False

def prepare_child_gpt_request(system_prompt, user_prompt, request_params):
    """Prepare the GPT request for a single child job with formatting instructions"""
    try:
        # Enhanced system prompt for better formatting
        enhanced_system_prompt = f"""{system_prompt}

Please format your response with proper structure using:
- **Bold text** for headings and important points
- Use clear headings and subheadings
- Organize content with proper paragraphs
- Use bullet points or numbered lists where appropriate
- Ensure the content is well-structured and professional
"""
        
        messages = [
            {"role": "system", "content": enhanced_system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        request_data = {
            "model": request_params.get("model", DEFAULT_MODEL),
            "messages": messages,
            "temperature": float(request_params.get("temperature", DEFAULT_TEMPERATURE)),
            "max_tokens": int(request_params.get("max_tokens", DEFAULT_MAX_TOKENS)),
            "top_p": float(request_params.get("top_p", DEFAULT_TOP_P)),
            "frequency_penalty": float(request_params.get("frequency_penalty", DEFAULT_FREQUENCY_PENALTY)),
            "presence_penalty": float(request_params.get("presence_penalty", DEFAULT_PRESENCE_PENALTY)),
            "response_format": {"type": "text"}
        }
        
        return request_data
    except Exception as e:
        logger.error(f"Error preparing GPT request: {str(e)}", exc_info=True)
        return None

def create_pdf_from_content(content, job_id, user_prompt=""):
    """Create a PDF from the GPT response content with improved formatting"""
    try:
        pdf_path = PDF_DIR / f"{job_id}.pdf"
        
        # Create the PDF document
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        
        # Custom styles - ALL TEXT IN BLACK
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=20,
            spaceBefore=10,
            alignment=TA_LEFT,
            textColor=black,
            leftIndent=0
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=15,
            spaceBefore=12,
            textColor=black,
            leftIndent=0
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            spaceBefore=4,
            alignment=TA_JUSTIFY,
            leftIndent=0,
            rightIndent=0,
            textColor=black
        )
        
        # Bullet point style (for clean indented text without bullets)
        bullet_style = ParagraphStyle(
            'CustomBullet',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            spaceBefore=2,
            alignment=TA_LEFT,
            leftIndent=20,  # Indent for former bullet points
            rightIndent=0,
            textColor=black
        )
        
        # Process the content to convert markdown-like formatting
        formatted_content = format_content_for_pdf(content)
        
        # Split content into paragraphs and process each
        paragraphs = formatted_content.split('\n\n')
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Skip empty lines or lines with just dashes
            if para in ['---', '===', '───'] or re.match(r'^[-=─]+$', para):
                continue
                
            # Check if it's a heading (starts with ###, ##, or #)
            if para.startswith('#'):
                # Handle markdown headings
                if para.startswith('###'):
                    heading_text = para.replace('###', '').strip()
                    heading = Paragraph(f"<b>{heading_text}</b>", subtitle_style)
                elif para.startswith('##'):
                    heading_text = para.replace('##', '').strip()
                    heading = Paragraph(f"<b>{heading_text}</b>", title_style)
                elif para.startswith('#'):
                    heading_text = para.replace('#', '').strip()
                    heading = Paragraph(f"<b>{heading_text}</b>", title_style)
                else:
                    heading = Paragraph(para, normal_style)
                elements.append(heading)
                elements.append(Spacer(1, 8))
            else:
                # Check if this looks like it was originally a bullet point (indented content)
                lines = para.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # If line looks like it was a bullet point (common patterns), use bullet style
                    if (line.startswith(('Product', 'The', 'A', 'An', '•')) or 
                        any(keyword in line.lower() for keyword in ['increase', 'decrease', 'growth', 'sales', 'market', 'customer'])):
                        p = Paragraph(line, bullet_style)
                    else:
                        # Regular paragraph
                        p = Paragraph(line, normal_style)
                    
                    elements.append(p)
                    elements.append(Spacer(1, 4))
        
        # Build the PDF
        doc.build(elements)
        
        logger.info(f"PDF created successfully for job {job_id}: {pdf_path}")
        return True, str(pdf_path)
        
    except Exception as e:
        logger.error(f"Error creating PDF for job {job_id}: {str(e)}", exc_info=True)
        return False, str(e)


def format_content_for_pdf(content):
    """Format content for better PDF rendering - improved version"""
    try:
        # Remove horizontal lines (--- or similar patterns)
        content = re.sub(r'^---+\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^═+\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^─+\s*$', '', content, flags=re.MULTILINE)
        
        # Convert markdown formatting to HTML with BLACK color
        content = re.sub(r'\*\*(.*?)\*\*', r'<b><font color="black">\1</font></b>', content)  # Bold in black
        content = re.sub(r'\*(.*?)\*', r'<i><font color="black">\1</font></i>', content)      # Italic in black
        content = re.sub(r'`(.*?)`', r'<font name="Courier" color="black">\1</font>', content)  # Code in black
        
        # Handle bullet points - REMOVE bullets entirely, just keep the text
        content = re.sub(r'^[-•*]\s+(.*?)$', r'\1', content, flags=re.MULTILINE)
        
        # Handle numbered lists - keep numbers but clean formatting
        content = re.sub(r'^(\d+)\.\s+(.*?)$', r'\1. \2', content, flags=re.MULTILINE)
        
        # Clean up multiple consecutive newlines
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        # Clean up any HTML entities
        content = unescape(content)
        
        return content.strip()
        
    except Exception as e:
        logger.error(f"Error formatting content: {str(e)}", exc_info=True)
        return content
def call_openai_with_retry(request_data, job_id, max_retries=MAX_RETRIES, initial_backoff=INITIAL_BACKOFF):
    """Call OpenAI API with exponential backoff retry logic"""
    retry_count = 0
    backoff = initial_backoff
    
    while retry_count <= max_retries:
        try:
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
                timeout=60
            )
            
            if response.status_code == 200 or (400 <= response.status_code < 500 and response.status_code != 429):
                return response
                
            if response.status_code >= 500 or response.status_code == 429:
                logger.warning(f"Job {job_id}: OpenAI API returned status {response.status_code}, retrying...")
            else:
                return response
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Job {job_id}: Request exception: {str(e)}, retrying...")
        
        retry_count += 1
        jitter = 0.8 + 0.4 * random.random()
        backoff = backoff * 2 * jitter
    
    logger.error(f"Job {job_id}: Failed after {max_retries} retry attempts")
    
    class MockResponse:
        def __init__(self):
            self.status_code = 500
            self.text = f"Failed after {max_retries} retry attempts"
            
        def json(self):
            return {"error": {"message": self.text}}
    
    return MockResponse()

def process_job(job_id):
    """Process a single child job and conditionally create PDF"""
    start_time = time.time()
    logger.info(f"Starting to process child job {job_id}")
    
    try:
        if job_id not in processing_jobs:
            logger.warning(f"Child job {job_id} not found in processing queue")
            return
            
        job = processing_jobs[job_id]
        job["status"] = "processing"
        pdf_enabled = job.get("pdf_enabled", True)  # Default to True for backward compatibility
        
        logger.info(f"Processing child job {job_id}, PDF generation: {pdf_enabled}")
        
        api_start_time = time.time()
        logger.info(f"Sending request to OpenAI API for child job {job_id}")
        
        response = call_openai_with_retry(job["gpt_request"], job_id)
        
        api_time = time.time() - api_start_time
        logger.info(f"OpenAI API response received for child job {job_id}. Time taken: {api_time:.2f}s. Status code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                response_json = response.json()
                content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                token_usage = response_json.get("usage", {})
                
                logger.info(f"Child job {job_id} completed successfully. Tokens used - Prompt: {token_usage.get('prompt_tokens', 'N/A')}, "
                            f"Completion: {token_usage.get('completion_tokens', 'N/A')}, "
                            f"Total: {token_usage.get('total_tokens', 'N/A')}")
                
                # CONDITIONALLY CREATE PDF
                pdf_success = False
                pdf_path_or_error = None
                
                if pdf_enabled:
                    pdf_success, pdf_path_or_error = create_pdf_from_content(
                        content, 
                        job_id, 
                        job.get("user_prompt", "")
                    )
                    if pdf_success:
                        logger.info(f"PDF created successfully for job {job_id}")
                    else:
                        logger.error(f"PDF creation failed for job {job_id}: {pdf_path_or_error}")
                else:
                    logger.info(f"PDF generation disabled for job {job_id}")
                
                result = {
                    "job_id": job_id,
                    "parent_job_id": job.get("parent_job_id"),
                    "child_index": job.get("child_index"),
                    "status": "completed",
                    "submitted_at": job["submitted_at"],
                    "completed_at": time.time(),
                    "processing_time": time.time() - job["submitted_at"],
                    "api_time": api_time,
                    "token_usage": token_usage,
                    "user_prompt": job.get("user_prompt", ""),
                    "response": content,
                    "pdf_enabled": pdf_enabled,  # ADD THIS LINE
                    "pdf_generated": pdf_success,
                    "pdf_path": pdf_path_or_error if pdf_success else None,
                    "pdf_error": pdf_path_or_error if pdf_enabled and not pdf_success else None
                }
                
                # Save child job result
                result_path = RESULTS_DIR / f"{job_id}.json"
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2)
                
                job["status"] = "completed"
                
                # Update parent job status
                parent_job_id = job.get("parent_job_id")
                if parent_job_id:
                    update_parent_job_status(parent_job_id)
            
            except Exception as e:
                logger.error(f"Error parsing OpenAI API response for child job {job_id}: {str(e)}", exc_info=True)
                raise
                
        else:
            error_message = "Unknown error"
            retriable_error = response.status_code >= 500 or response.status_code == 429
            
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", "Unknown error")
            except Exception as e:
                if response.text:
                    error_message = f"API Error: {response.status_code} - {response.text[:200]}"
                else:
                    error_message = f"API Error: {response.status_code}"
                
                logger.error(f"Could not parse error response from OpenAI API for child job {job_id}: {str(e)}")
            
            logger.error(f"OpenAI API error for child job {job_id}: {error_message}")
            
            result = {
                "job_id": job_id,
                "parent_job_id": job.get("parent_job_id"),
                "child_index": job.get("child_index"),
                "status": "error",
                "submitted_at": job["submitted_at"],
                "completed_at": time.time(),
                "processing_time": time.time() - job["submitted_at"],
                "api_time": api_time,
                "error": error_message,
                "retriable": retriable_error,
                "user_prompt": job.get("user_prompt", ""),
                "pdf_enabled": job.get("pdf_enabled", True),  # ADD THIS LINE
                "pdf_generated": False
            }
            
            result_path = RESULTS_DIR / f"{job_id}.json"
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            job["status"] = "error"
            
            # Update parent job status
            parent_job_id = job.get("parent_job_id")
            if parent_job_id:
                update_parent_job_status(parent_job_id)
            
    except Exception as e:
        logger.error(f"Error processing child job {job_id}: {str(e)}", exc_info=True)
        
        result = {
            "job_id": job_id,
            "parent_job_id": job.get("parent_job_id") if job_id in processing_jobs else None,
            "child_index": job.get("child_index") if job_id in processing_jobs else None,
            "status": "error",
            "submitted_at": processing_jobs[job_id]["submitted_at"] if job_id in processing_jobs and "submitted_at" in processing_jobs[job_id] else time.time(),
            "completed_at": time.time(),
            "processing_time": time.time() - (processing_jobs[job_id]["submitted_at"] if job_id in processing_jobs and "submitted_at" in processing_jobs[job_id] else time.time()),
            "error": str(e),
            "pdf_enabled": processing_jobs[job_id].get("pdf_enabled", True) if job_id in processing_jobs else True,  # ADD THIS LINE
            "pdf_generated": False
        }
        
        result_path = RESULTS_DIR / f"{job_id}.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        if job_id in processing_jobs:
            processing_jobs[job_id]["status"] = "error"
            
            parent_job_id = processing_jobs[job_id].get("parent_job_id")
            if parent_job_id:
                update_parent_job_status(parent_job_id)
    
    finally:
        if job_id in processing_jobs:
            del processing_jobs[job_id]
        
        total_time = time.time() - start_time
        logger.info(f"Finished processing child job {job_id}. Total processing time: {total_time:.2f}s")

def update_parent_job_status(parent_job_id):
    """Update parent job status based on child job statuses"""
    try:
        if parent_job_id not in parent_jobs:
            return
            
        parent_job = parent_jobs[parent_job_id]
        child_job_ids = parent_job["child_job_ids"]
        
        # Check status of all child jobs
        completed_children = 0
        error_children = 0
        processing_children = 0
        
        for child_id in child_job_ids:
            if child_id in processing_jobs:
                processing_children += 1
            else:
                # Check file
                result_path = RESULTS_DIR / f"{child_id}.json"
                if result_path.exists():
                    try:
                        with open(result_path, 'r') as f:
                            child_result = json.load(f)
                        if child_result.get("status") == "completed":
                            completed_children += 1
                        elif child_result.get("status") == "error":
                            error_children += 1
                    except:
                        error_children += 1
                else:
                    processing_children += 1
        
        # Update parent status
        total_children = len(child_job_ids)
        if completed_children == total_children:
            parent_job["status"] = "completed"
        elif error_children + completed_children == total_children:
            parent_job["status"] = "partially_completed"
        else:
            parent_job["status"] = "processing"
            
        # Save parent job status - ADD PDF_ENABLED INFO
        parent_result = {
            "job_id": parent_job_id,
            "type": "parent",
            "status": parent_job["status"],
            "submitted_at": parent_job["submitted_at"],
            "total_children": total_children,
            "completed_children": completed_children,
            "error_children": error_children,
            "processing_children": processing_children,
            "child_job_ids": child_job_ids,
            "pdf_generation_enabled": parent_job.get("pdf_enabled", True),  # ADD THIS LINE
            "updated_at": time.time()
        }
        
        parent_result_path = RESULTS_DIR / f"{parent_job_id}_parent.json"
        with open(parent_result_path, 'w') as f:
            json.dump(parent_result, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error updating parent job {parent_job_id}: {str(e)}", exc_info=True)

def worker():
    """Background worker to process queued requests"""
    global worker_running
    worker_running = True
    
    logger.info("Background worker thread started with thread ID: %s", threading.get_ident())
    
    # Add proper interval tracking for conditional logging
    last_status_log = time.time()
    status_log_interval = 30  # Check every 30 seconds
    
    while worker_running:
        try:
            batch = []
            for _ in range(MAX_CONCURRENT_REQUESTS):
                try:
                    job_id = request_queue.get(block=True, timeout=1)
                    if job_id in processing_jobs:
                        batch.append(job_id)
                        logger.info(f"Added child job {job_id} to current processing batch")
                    request_queue.task_done()
                except queue.Empty:
                    break
            
            if batch:
                logger.info(f"Worker found {len(batch)} child jobs to process: {batch}")
                
                with ThreadPoolExecutor(max_workers=max(1, len(batch))) as executor:
                    futures = {
                        executor.submit(process_job, job_id): job_id for job_id in batch
                    }
                    
                    for future in futures:
                        try:
                            future.result()
                        except Exception as e:
                            job_id = futures[future]
                            logger.error(f"Error in thread for child job {job_id}: {str(e)}", exc_info=True)
                            if job_id in processing_jobs:
                                result = {
                                    "job_id": job_id,
                                    "status": "error",
                                    "submitted_at": processing_jobs[job_id]["submitted_at"],
                                    "completed_at": time.time(),
                                    "error": str(e),
                                    "pdf_generated": False
                                }
                                
                                result_path = RESULTS_DIR / f"{job_id}.json"
                                with open(result_path, 'w') as f:
                                    json.dump(result, f, indent=2)
                                
                                del processing_jobs[job_id]
            
            # Conditional status logging - only log when there's activity
            current_time = time.time()
            if current_time - last_status_log >= status_log_interval:
                queue_size = request_queue.qsize()
                active_jobs = len(processing_jobs)
                
                # Only log when there are active jobs or items in queue
                if queue_size > 0 or active_jobs > 0:
                    logger.info(f"Queue status: Size={queue_size}, Active child jobs={active_jobs}")
                
                last_status_log = current_time
                
            if not batch:
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Worker error: {str(e)}", exc_info=True)
            time.sleep(1)

@app.route('/api/submit', methods=['POST'])
def submit_job():
    """Submit a new job with multiple prompts in user_prompts array"""
    start_time = time.time()
    
    try:
        job_id = request.args.get('job_id')
        
        # Add PDF generation control parameter
        generate_pdf = request.args.get('generate_pdf', 'true').lower() in ['true', '1', 'yes', 'on']
        
        if not job_id:
            job_id = f"job_{int(time.time() * 1000)}"
        
        client_ip = request.remote_addr
        logger.info(f"New parent job submission from {client_ip}. Job ID: {job_id}, PDF Generation: {generate_pdf}")
        
        request_params = {
            "model": request.args.get('model', DEFAULT_MODEL),
            "max_tokens": request.args.get('max_tokens', DEFAULT_MAX_TOKENS),
            "temperature": request.args.get('temperature', DEFAULT_TEMPERATURE),
            "top_p": request.args.get('top_p', DEFAULT_TOP_P),
            "frequency_penalty": request.args.get('frequency_penalty', DEFAULT_FREQUENCY_PENALTY),
            "presence_penalty": request.args.get('presence_penalty', DEFAULT_PRESENCE_PENALTY)
        }
        
        # Check if parent job already exists
        if (job_id in parent_jobs or 
            (RESULTS_DIR / f"{job_id}_parent.json").exists() or
            any(child_id.startswith(f"{job_id}_child_") for child_id in processing_jobs)):
            logger.warning(f"Parent job ID '{job_id}' already exists. Request rejected.")
            return jsonify({"error": f"Parent job ID '{job_id}' already exists"}), 400
        
        # Get JSON data
        json_data = {}
        
        if 'json_file' in request.files:
            json_file = request.files['json_file']
            try:
                json_content = json_file.read().decode('utf-8')
                json_data = json.loads(json_content)
                logger.info(f"Parent job {job_id} data received from JSON file")
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON file for parent job {job_id}: {str(e)}", exc_info=True)
                return jsonify({"error": f"Error parsing JSON file: {str(e)}"}), 400
        elif request.is_json:
            json_data = request.json
            logger.info(f"Parent job {job_id} data received from JSON body")
        else:
            return jsonify({"error": "Request must contain either a JSON file or JSON data"}), 400
        
        # Validate JSON structure
        if "user_prompts" not in json_data:
            return jsonify({
                "error": "JSON must contain 'user_prompts' array. Format: {\"system\": \"...\", \"user_prompts\": [\"prompt1\", \"prompt2\", ...]}"
            }), 400
        
        system_prompt = json_data.get("system", "You are a helpful assistant.")
        user_prompts = json_data["user_prompts"]
        
        if not isinstance(user_prompts, list) or len(user_prompts) == 0:
            return jsonify({"error": "user_prompts must be a non-empty array"}), 400
        
        logger.info(f"Parent job {job_id}: Found {len(user_prompts)} user prompts")
        
        # Create parent job tracking - ADD PDF_ENABLED FLAG
        parent_jobs[job_id] = {
            "status": "processing",
            "submitted_at": time.time(),
            "child_job_ids": [],
            "total_prompts": len(user_prompts),
            "pdf_enabled": generate_pdf  # ADD THIS LINE
        }
        
        # Create child jobs
        child_job_ids = []
        for i, user_prompt in enumerate(user_prompts, 1):
            child_job_id = f"{job_id}_child_{i}"
            child_job_ids.append(child_job_id)
            
            # Prepare GPT request for this child
            gpt_request = prepare_child_gpt_request(system_prompt, user_prompt, request_params)
            if not gpt_request:
                return jsonify({"error": f"Error preparing GPT request for prompt {i}"}), 400
            
            # Create child job - ADD PDF_ENABLED FLAG
            processing_jobs[child_job_id] = {
                "status": "queued",
                "submitted_at": time.time(),
                "parent_job_id": job_id,
                "child_index": i,
                "user_prompt": user_prompt,
                "gpt_request": gpt_request,
                "pdf_enabled": generate_pdf  # ADD THIS LINE
            }
            
            # Add to queue for parallel processing
            request_queue.put(child_job_id)
            logger.info(f"Child job {child_job_id} added to processing queue")
        
        # Update parent job with child IDs
        parent_jobs[job_id]["child_job_ids"] = child_job_ids
        
        # Save initial parent job status
        update_parent_job_status(job_id)
        
        processing_time = time.time() - start_time
        logger.info(f"Parent job {job_id} submission completed with {len(child_job_ids)} child jobs in {processing_time:.2f}s")
        
        return jsonify({
            "job_id": job_id,
            "type": "parent",
            "status": "processing",
            "child_job_ids": child_job_ids,
            "total_prompts": len(user_prompts),
            "pdf_generation_enabled": generate_pdf,  # ADD THIS LINE
            "message": f"Parent job submitted successfully with {len(child_job_ids)} child jobs processing in parallel" + 
                      (". PDFs will be generated for each response." if generate_pdf else ". PDF generation disabled.")
        })
        
    except Exception as e:
        logger.error(f"Error submitting parent job: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error submitting parent job: {str(e)}"}), 500
@app.route('/api/pdf/<job_id>', methods=['GET'])
def download_pdf(job_id):
    """Download PDF for a specific job - now checks if PDF generation was enabled"""
    try:
        client_ip = request.remote_addr
        logger.info(f"PDF download request from {client_ip} for job {job_id}")
        
        # First check if the job result exists and if PDF was enabled
        result_path = RESULTS_DIR / f"{job_id}.json"
        if result_path.exists():
            try:
                with open(result_path, 'r') as f:
                    job_result = json.load(f)
                
                # Check if PDF generation was disabled for this job
                if not job_result.get("pdf_enabled", True):
                    logger.warning(f"PDF generation was disabled for job {job_id}")
                    return jsonify({
                        "error": f"PDF generation was disabled for job {job_id}",
                        "message": "This job was submitted with PDF generation disabled (generate_pdf=false)",
                        "pdf_enabled": False
                    }), 404
                    
            except Exception as e:
                logger.warning(f"Could not read job result for {job_id}: {str(e)}")
        
        pdf_path = PDF_DIR / f"{job_id}.pdf"
        
        if not pdf_path.exists():
            logger.warning(f"PDF not found for job {job_id}: {pdf_path}")
            return jsonify({
                "error": f"PDF not found for job {job_id}",
                "message": "The PDF may not have been generated yet, the job may have failed, or PDF generation was disabled"
            }), 404
        
        logger.info(f"Serving PDF for job {job_id}: {pdf_path}")
        
        return send_file(
            str(pdf_path),
            as_attachment=True,
            download_name=f"gpt_response_{job_id}.pdf",
            mimetype='application/pdf'
        )
        
    except Exception as e:
        logger.error(f"Error serving PDF for job {job_id}: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error serving PDF: {str(e)}"}), 500

@app.route('/api/status', methods=['GET'])
def get_job_status():
    """Get the status of a job (parent or child) - now includes PDF status"""
    start_time = time.time()
    job_id = request.args.get('job_id')
    client_ip = request.remote_addr
    
    if not job_id:
        # Return system status
        logger.info(f"System status check from {client_ip}")
        
        try:
            active_child_jobs = len(processing_jobs)
            active_parent_jobs = len(parent_jobs)
            
            response = {
                "active_child_jobs_count": active_child_jobs,
                "active_parent_jobs_count": active_parent_jobs,
                "queue_size": request_queue.qsize(),
                "default_model": DEFAULT_MODEL,
                "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
                "pdf_generation_default": True,  # UPDATED TO SHOW DEFAULT
                "timestamp": time.time()
            }
            
            return jsonify(response)
        
        except Exception as e:
            logger.error(f"Error generating system status: {str(e)}", exc_info=True)
            return jsonify({"error": f"Error generating system status: {str(e)}"}), 500
    
    logger.info(f"Status check from {client_ip} for job {job_id}")
    
    # Check if it's a parent job
    parent_path = RESULTS_DIR / f"{job_id}_parent.json"
    if job_id in parent_jobs or parent_path.exists():
        try:
            if job_id in parent_jobs:
                parent_job = parent_jobs[job_id]
                response = {
                    "job_id": job_id,
                    "type": "parent",
                    "status": parent_job["status"],
                    "submitted_at": parent_job["submitted_at"],
                    "child_job_ids": parent_job["child_job_ids"],
                    "total_prompts": parent_job["total_prompts"],
                    "pdf_generation_enabled": parent_job.get("pdf_enabled", True)  # ADD THIS LINE
                }
            else:
                with open(parent_path, 'r') as f:
                    response = json.load(f)
            
            # Add PDF availability info for completed children
            if "child_job_ids" in response:
                pdf_status = {}
                for child_id in response["child_job_ids"]:
                    pdf_path = PDF_DIR / f"{child_id}.pdf"
                    pdf_status[child_id] = pdf_path.exists()
                response["pdf_availability"] = pdf_status
            
            return jsonify(response)
        except Exception as e:
            logger.error(f"Error reading parent job status for {job_id}: {str(e)}", exc_info=True)
            return jsonify({"error": f"Error reading parent job status: {str(e)}"}), 500
    
    # Check if it's a child job in processing
    if job_id in processing_jobs:
        job = processing_jobs[job_id]
        response = {
            "job_id": job_id,
            "type": "child",
            "status": job["status"],
            "submitted_at": job["submitted_at"],
            "elapsed_time": time.time() - job["submitted_at"],
            "parent_job_id": job.get("parent_job_id"),
            "child_index": job.get("child_index"),
            "pdf_enabled": job.get("pdf_enabled", True),  # ADD THIS LINE
            "pdf_available": False
        }
        
        return jsonify(response)
    
    # Check if child job result file exists
    result_path = RESULTS_DIR / f"{job_id}.json"
    if result_path.exists():
        try:
            with open(result_path, 'r') as f:
                result = json.load(f)
            
            result["type"] = "child"
            result["file_size_kb"] = round(result_path.stat().st_size / 1024, 2)
            
            # Check PDF availability (and if it was enabled)
            pdf_path = PDF_DIR / f"{job_id}.pdf"
            result["pdf_available"] = pdf_path.exists()
            if result["pdf_available"]:
                result["pdf_size_kb"] = round(pdf_path.stat().st_size / 1024, 2)
                result["download_url"] = f"/api/pdf/{job_id}"
            
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error reading result file for job {job_id}: {str(e)}", exc_info=True)
            return jsonify({"error": f"Error reading result file: {str(e)}"}), 500
    
    logger.warning(f"Job {job_id} not found")
    return jsonify({"error": f"Job {job_id} not found"}), 404

@app.route('/api/retrieve', methods=['GET'])
def retrieve_job():
    """Retrieve the result of a job (parent returns all children, child returns individual) - now includes PDF info"""
    start_time = time.time()
    job_id = request.args.get('job_id')
    client_ip = request.remote_addr
    
    logger.info(f"Result retrieval request from {client_ip} for job {job_id}")
    
    if not job_id:
        return jsonify({"error": "No job_id provided"}), 400
    
    # Check if it's a parent job
    parent_path = RESULTS_DIR / f"{job_id}_parent.json"
    if parent_path.exists() or job_id in parent_jobs:
        try:
            # Get parent job info
            if job_id in parent_jobs:
                parent_job = parent_jobs[job_id]
                child_job_ids = parent_job["child_job_ids"]
            else:
                with open(parent_path, 'r') as f:
                    parent_data = json.load(f)
                child_job_ids = parent_data["child_job_ids"]
            
            # Collect all child results
            child_results = []
            completed_count = 0
            error_count = 0
            pdf_count = 0
            
            for child_id in child_job_ids:
                child_path = RESULTS_DIR / f"{child_id}.json"
                if child_path.exists():
                    try:
                        with open(child_path, 'r') as f:
                            child_result = json.load(f)
                        
                        # Add PDF info
                        pdf_path = PDF_DIR / f"{child_id}.pdf"
                        child_result["pdf_available"] = pdf_path.exists()
                        if child_result["pdf_available"]:
                            child_result["pdf_size_kb"] = round(pdf_path.stat().st_size / 1024, 2)
                            child_result["download_url"] = f"/api/pdf/{child_id}"
                            pdf_count += 1
                        
                        child_results.append(child_result)
                        if child_result.get("status") == "completed":
                            completed_count += 1
                        elif child_result.get("status") == "error":
                            error_count += 1
                    except Exception as e:
                        logger.error(f"Error reading child result {child_id}: {str(e)}")
                        child_results.append({
                            "job_id": child_id,
                            "status": "error",
                            "error": f"Could not read result file: {str(e)}",
                            "pdf_available": False
                        })
                        error_count += 1
                else:
                    # Child still processing
                    child_results.append({
                        "job_id": child_id,
                        "status": "processing",
                        "message": "Child job is still processing",
                        "pdf_available": False
                    })
            
            # Sort child results by child_index
            child_results.sort(key=lambda x: x.get("child_index", 0))
            
            # Create parent response with all child results
            parent_response = {
                "job_id": job_id,
                "type": "parent",
                "status": "completed" if completed_count == len(child_job_ids) else "processing",
                "total_children": len(child_job_ids),
                "completed_children": completed_count,
                "error_children": error_count,
                "pdfs_generated": pdf_count,
                "child_results": child_results,
                "retrieved_at": time.time()
            }
            
            logger.info(f"Parent job {job_id} retrieval completed: {completed_count}/{len(child_job_ids)} children completed, {pdf_count} PDFs generated")
            return jsonify(parent_response)
            
        except Exception as e:
            logger.error(f"Error retrieving parent job {job_id}: {str(e)}", exc_info=True)
            return jsonify({"error": f"Error retrieving parent job: {str(e)}"}), 500
    
    # Check if it's a child job
    result_path = RESULTS_DIR / f"{job_id}.json"
    if result_path.exists():
        try:
            file_size = result_path.stat().st_size
            file_size_kb = file_size / 1024
            
            with open(result_path, 'r') as f:
                result = json.load(f)
            
            if file_size_kb < MIN_COMPLETE_SIZE_KB and result.get("status") != "completed":
                return jsonify({
                    "job_id": job_id,
                    "type": "child",
                    "status": "processing",
                    "message": f"Child job is still processing (result size: {file_size_kb:.2f}KB)",
                    "pdf_available": False
                })
            
            result["type"] = "child"
            result["file_size_kb"] = file_size_kb
            result["retrieved_at"] = time.time()
            
            # Add PDF info
            pdf_path = PDF_DIR / f"{job_id}.pdf"
            result["pdf_available"] = pdf_path.exists()
            if result["pdf_available"]:
                result["pdf_size_kb"] = round(pdf_path.stat().st_size / 1024, 2)
                result["download_url"] = f"/api/pdf/{job_id}"
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error reading result file for child job {job_id}: {str(e)}", exc_info=True)
            return jsonify({"error": f"Error reading result file: {str(e)}"}), 500
    
    # Check if child job is still processing
    if job_id in processing_jobs:
        job = processing_jobs[job_id]
        return jsonify({
            "job_id": job_id,
            "type": "child",
            "status": job["status"],
            "submitted_at": job.get("submitted_at", time.time()),
            "parent_job_id": job.get("parent_job_id"),
            "child_index": job.get("child_index"),
            "message": "Child job is still processing",
            "pdf_available": False
        })
    
    logger.warning(f"Job {job_id} not found for retrieval")
    return jsonify({"error": f"Job {job_id} not found"}), 404

@app.route('/api/delete', methods=['GET'])
def delete_job():
    """Delete a job result (parent deletes all children, child deletes individual) - now includes PDF cleanup"""
    start_time = time.time()
    job_id = request.args.get('job_id')
    client_ip = request.remote_addr
    
    logger.info(f"Delete request from {client_ip} for job {job_id}")
    
    if not job_id:
        return jsonify({"error": "No job_id provided"}), 400
    
    # Check if it's a parent job
    parent_path = RESULTS_DIR / f"{job_id}_parent.json"
    if parent_path.exists() or job_id in parent_jobs:
        try:
            # Get child job IDs
            child_job_ids = []
            if job_id in parent_jobs:
                child_job_ids = parent_jobs[job_id]["child_job_ids"]
                del parent_jobs[job_id]
            else:
                with open(parent_path, 'r') as f:
                    parent_data = json.load(f)
                child_job_ids = parent_data["child_job_ids"]
            
            # Delete all child jobs
            deleted_children = []
            deleted_pdfs = []
            failed_deletions = []
            
            for child_id in child_job_ids:
                try:
                    # Remove from processing queue if still there
                    if child_id in processing_jobs:
                        del processing_jobs[child_id]
                    
                    # Delete result file
                    child_path = RESULTS_DIR / f"{child_id}.json"
                    if child_path.exists():
                        child_path.unlink()
                    
                    # Delete PDF file
                    pdf_path = PDF_DIR / f"{child_id}.pdf"
                    if pdf_path.exists():
                        pdf_path.unlink()
                        deleted_pdfs.append(child_id)
                    
                    deleted_children.append(child_id)
                        
                except Exception as e:
                    logger.error(f"Error deleting child job {child_id}: {str(e)}")
                    failed_deletions.append({"job_id": child_id, "error": str(e)})
            
            # Delete parent job file
            if parent_path.exists():
                parent_path.unlink()
            
            logger.info(f"Parent job {job_id} and {len(deleted_children)} child jobs deleted successfully, {len(deleted_pdfs)} PDFs deleted")
            
            response = {
                "job_id": job_id,
                "type": "parent",
                "status": "deleted",
                "deleted_children": deleted_children,
                "deleted_pdfs": deleted_pdfs,
                "failed_deletions": failed_deletions,
                "message": f"Parent job and {len(deleted_children)} child jobs deleted successfully, {len(deleted_pdfs)} PDFs cleaned up"
            }
            
            if failed_deletions:
                response["message"] += f" ({len(failed_deletions)} child deletions failed)"
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error deleting parent job {job_id}: {str(e)}", exc_info=True)
            return jsonify({"error": f"Error deleting parent job: {str(e)}"}), 500
    
    # Check if it's a child job
    result_path = RESULTS_DIR / f"{job_id}.json"
    if result_path.exists():
        try:
            # Get parent job ID before deletion
            parent_job_id = None
            try:
                with open(result_path, 'r') as f:
                    child_data = json.load(f)
                parent_job_id = child_data.get("parent_job_id")
            except:
                pass
            
            # Delete child job file
            result_path.unlink()
            
            # Delete PDF file
            pdf_deleted = False
            pdf_path = PDF_DIR / f"{job_id}.pdf"
            if pdf_path.exists():
                pdf_path.unlink()
                pdf_deleted = True
            
            logger.info(f"Child job {job_id} result file deleted successfully, PDF deleted: {pdf_deleted}")
            
            # Update parent job status if exists
            if parent_job_id:
                update_parent_job_status(parent_job_id)
            
            return jsonify({
                "job_id": job_id,
                "type": "child",
                "status": "deleted",
                "parent_job_id": parent_job_id,
                "pdf_deleted": pdf_deleted,
                "message": f"Child job deleted successfully{'with PDF' if pdf_deleted else ''}"
            })
            
        except Exception as e:
            logger.error(f"Error deleting child job result file for {job_id}: {str(e)}", exc_info=True)
            return jsonify({"error": f"Error deleting child job: {str(e)}"}), 500
    
    # Check if child job is still processing
    if job_id in processing_jobs:
        parent_job_id = processing_jobs[job_id].get("parent_job_id")
        del processing_jobs[job_id]
        
        # Update parent job status if exists
        if parent_job_id:
            update_parent_job_status(parent_job_id)
        
        logger.info(f"Child job {job_id} cancelled while in processing queue")
        return jsonify({
            "job_id": job_id,
            "type": "child",
            "status": "cancelled",
            "parent_job_id": parent_job_id,
            "message": "Child job cancelled successfully"
        })
    
    logger.warning(f"Job {job_id} not found for deletion")
    return jsonify({"error": f"Job {job_id} not found"}), 404

# Helper function to find a job's position in the queue
def get_queue_position(job_id):
    """Determine the approximate position of a job in the queue"""
    try:
        queue_items = list(request_queue.queue)
        
        if job_id in queue_items:
            return queue_items.index(job_id) + 1
        
        if job_id in processing_jobs and processing_jobs[job_id]["status"] == "processing":
            return 0
            
        if job_id in processing_jobs:
            return -1
            
    except Exception as e:
        logger.error(f"Error determining queue position for job {job_id}: {str(e)}")
        return -1
    
    return -1

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    logger.info("Health check request received")
    
    queue_size = request_queue.qsize()
    active_child_jobs = len(processing_jobs)
    active_parent_jobs = len(parent_jobs)
    
    # Count PDF files
    pdf_count = len(list(PDF_DIR.glob("*.pdf")))
    
    response = {
        "status": "healthy",
        "queue_size": queue_size,
        "active_child_jobs": active_child_jobs,
        "active_parent_jobs": active_parent_jobs,
        "worker_running": worker_running,
        "model": DEFAULT_MODEL,
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "max_retries": MAX_RETRIES,
        "min_complete_size_kb": MIN_COMPLETE_SIZE_KB,
        "pdf_generation_enabled": True,
        "total_pdfs_generated": pdf_count
    }
    
    logger.info(f"Health check response: Queue size: {queue_size}, Active child jobs: {active_child_jobs}, Active parent jobs: {active_parent_jobs}, PDFs: {pdf_count}")
    return jsonify(response)

if __name__ == '__main__':
    logger.info("Starting GPT Prompting Service with Parent-Child Job Support and PDF Generation")
    logger.info(f"Default model: {DEFAULT_MODEL}")
    logger.info(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    logger.info(f"Max retries: {MAX_RETRIES}")
    logger.info(f"Results directory: {RESULTS_DIR}")
    logger.info(f"PDF directory: {PDF_DIR}")
    
    RESULTS_DIR.mkdir(exist_ok=True)
    PDF_DIR.mkdir(exist_ok=True)
    
    logger.info("Starting background worker thread")
    worker_thread = threading.Thread(target=worker, daemon=True, name="WorkerThread")
    worker_thread.start()
    
    port = int(os.environ.get('PORT', 8081))
    logger.info(f"Starting Flask web server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
else:
    RESULTS_DIR.mkdir(exist_ok=True)
    PDF_DIR.mkdir(exist_ok=True)
    logger.info("Starting background worker thread (WSGI mode)")
    worker_thread = threading.Thread(target=worker, daemon=True, name="WorkerThread")
    worker_thread.start()
