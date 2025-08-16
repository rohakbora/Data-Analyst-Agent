import os
import sys
import io
import json
import re
import base64
import tempfile
import requests
import tiktoken
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import duckdb
import PyPDF2
from PIL import Image
import traceback
import logging
import seaborn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import signal

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model = "openai/gpt-4o-mini"
aipipe_token = os.getenv("AIPIPE_TOKEN")

if not aipipe_token:
    raise ValueError("AIPIPE_TOKEN environment variable not set")

# Timeout handler
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Request timed out after 5 minutes")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Data Analysis API",
        "endpoints": {
            "/api/": "POST - Submit analysis tasks",
            "/health": "GET - Health check"
        },
        "usage": "Send POST request to /api/ with questions.txt and data files"
    }), 200


# 3. Add a health check endpoint before your existing routes:
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Data Analysis API is running"}), 200


# Scraping config
MAX_TOKENS_PER_CHUNK = 1000
MAX_CONTEXT_TOKENS = 5500

def get_scraped_context(url, question):
    html = get_html(url)
    if not html:
        return None
    
    sections = extract_structured_content(html)
    chunks = chunk_sections(sections)
    relevant_chunks = find_relevant_chunks(chunks, question, MAX_CONTEXT_TOKENS)
    return "\n".join(relevant_chunks)

def extract_structured_content(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "sup"]):
        tag.extract()
    
    sections = []
    current_title = "Introduction"
    current_text = []

    for el in soup.find_all(["h1", "h2", "h3", "p", "ul", "ol", "table"]):
        if el.name in ["h1", "h2", "h3"]:
            if current_text:
                sections.append((current_title, "\n".join(current_text)))
                current_text = []
            current_title = clean_text(el.get_text())
        elif el.name == "table":
            table_text = extract_table(el)
            if table_text:
                current_text.append(table_text)
        else:
            text = clean_text(el.get_text())
            if text:
                current_text.append(text)

    if current_text:
        sections.append((current_title, "\n".join(current_text)))
    
    return sections

def extract_table(table):
    rows = []
    for tr in table.find_all("tr"):
        cells = [clean_text(td.get_text()) for td in tr.find_all(["td", "th"])]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)

def chunk_sections(sections):
    enc = tiktoken.encoding_for_model("gpt-4o")
    chunks = []
    for title, text in sections:
        tokens = enc.encode(text)
        if len(tokens) <= MAX_TOKENS_PER_CHUNK:
            chunks.append(f"{title}\n{text}")
        else:
            words = text.split()
            chunk = []
            token_count = 0
            for word in words:
                token_count += len(enc.encode(word + " "))
                chunk.append(word)
                if token_count > MAX_TOKENS_PER_CHUNK:
                    chunks.append(f"{title}\n{' '.join(chunk)}")
                    chunk = []
                    token_count = 0
            if chunk:
                chunks.append(f"{title}\n{' '.join(chunk)}")
    return chunks

def get_html(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return None

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def extract_visible_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text_blocks = []
    for el in soup.find_all(string=True):
        text = clean_text(el)
        if text:
            text_blocks.append(text)
    return text_blocks

def chunk_text_blocks(text_blocks):
    enc = tiktoken.encoding_for_model("gpt-4o")
    chunks = []
    current_text = ""
    current_tokens = 0
    for block in text_blocks:
        block_tokens = len(enc.encode(block))
        if current_tokens + block_tokens > MAX_TOKENS_PER_CHUNK:
            chunks.append(current_text.strip())
            current_text = ""
            current_tokens = 0
        current_text += block + "\n"
        current_tokens += block_tokens
    if current_text.strip():
        chunks.append(current_text.strip())
    return chunks

def find_relevant_chunks(chunks, question, max_tokens_for_context):
    enc = tiktoken.encoding_for_model("gpt-4o")
    
    # Embed question & chunks locally
    q_embedding = get_embedding(question)
    c_embeddings = get_embedding(chunks)
    
    # Rank by cosine similarity
    scores = cosine_similarity([q_embedding], c_embeddings)[0]
    ranked_chunks = [c for _, c in sorted(zip(scores, chunks), reverse=True)]
    
    selected_chunks = []
    total_tokens = 0
    for chunk in ranked_chunks:
        chunk_tokens = len(enc.encode(chunk))
        if total_tokens + chunk_tokens > max_tokens_for_context:
            break
        selected_chunks.append(chunk)
        total_tokens += chunk_tokens
    return selected_chunks

def get_embedding(text):
    if isinstance(text, str):
        return embedder.encode(text)
    elif isinstance(text, list):
        return embedder.encode(text)
    else:
        raise TypeError("Input must be a string or list of strings")


def get_scraped_context(url, question):
    html = get_html(url)
    if not html:
        return None
    text_blocks = extract_visible_text(html)
    chunks = chunk_text_blocks(text_blocks)
    relevant_chunks = find_relevant_chunks(chunks, question, MAX_CONTEXT_TOKENS)
    return "\n".join(relevant_chunks)

def extract_code(text):
    match = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def call_llm(messages):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {aipipe_token}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.2
    }
    try:
        resp = requests.post(url, headers=headers, json=data)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"LLM call failed: {e}")
        raise

SIZE_LIMIT = 100_000  # 100KB

def replace_image_paths_with_base64(obj, temp_dir):
    if isinstance(obj, dict):
        return {k: replace_image_paths_with_base64(v, temp_dir) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_image_paths_with_base64(v, temp_dir) for v in obj]
    elif isinstance(obj, str):
        normalized_obj = os.path.normpath(obj)
        normalized_temp_dir = os.path.normpath(temp_dir)

        if (normalized_obj.startswith(normalized_temp_dir) and
            os.path.isfile(obj) and
            obj.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))):

            try:
                with open(obj, "rb") as f:
                    image_data = f.read()

                if len(image_data) <= SIZE_LIMIT:
                    encoded = base64.b64encode(image_data).decode("ascii")
                    return encoded   # ✅ return only base64 string

                # Otherwise compress/resize
                with Image.open(obj) as img:
                    width, height = img.size
                    scale_factor = (SIZE_LIMIT / len(image_data)) ** 0.5
                    new_width = max(1, int(width * scale_factor))
                    new_height = max(1, int(height * scale_factor))

                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                    # Save to buffer with optimization
                    img_buffer = io.BytesIO()
                    if obj.lower().endswith('.png'):
                        img.save(img_buffer, format='PNG', optimize=True)
                    else:
                        img.save(img_buffer, format='JPEG', quality=85, optimize=True)

                    image_data = img_buffer.getvalue()

                encoded = base64.b64encode(image_data).decode("ascii")
                return encoded   # ✅ only base64 string

            except Exception as e:
                logging.error(f"Failed to convert {obj} to base64: {e}")
                return obj
        return obj
    else:
        return obj


@app.route('/api/', methods=['POST'])
def analyze():
    # Set up 5-minute timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)  # 300 seconds = 5 minutes
    
    try:
        files = request.files
        questions_file = files.get('questions.txt')
        if not questions_file:
            logging.error("Missing questions.txt")
            return jsonify({"error": "Missing questions.txt"}), 400
        
        try:
            task = questions_file.read().decode('utf-8').strip()
        except Exception as e:
            logging.error(f"Failed to read questions.txt: {e}")
            return jsonify({"error": "Failed to read questions.txt"}), 400
        
        # Extract URLs from task and scrape
        urls = re.findall(r'https?://[^\s"\'<>()]+', task)
        scraped_contexts = {}
        for url in set(urls):
            ctx = get_scraped_context(url, task)
            if ctx:
                scraped_contexts[url] = ctx
        
        scraped_text = ""
        if scraped_contexts:
            scraped_text = "\nScraped contexts from URLs:\n" + "\n\n".join(f"URL: {url}\n{ctx}" for url, ctx in scraped_contexts.items())
        
        # Save attachments
        file_paths = {}
        with tempfile.TemporaryDirectory() as temp_dir:
            for name, file in files.items():
                if name != 'questions.txt':
                    path = os.path.join(temp_dir, name)
                    try:
                        file.save(path)
                        file_paths[name] = path
                    except Exception as e:
                        logging.error(f"Failed to save file {name}: {e}")
            
            files_text = ""
            file_previews = []
            if file_paths:
                files_text = "\nAvailable files (use full paths in code):\n" + "\n".join(f"- {name}: '{path}'" for name, path in file_paths.items())
                for name, path in file_paths.items():
                    preview = None
                    try:
                        if name.lower().endswith('.csv'):
                            df = pd.read_csv(path)
                            preview = f"CSV preview for {name}:\n" + df.head().to_string()
                        elif name.lower().endswith(('.xls', '.xlsx')):
                            df = pd.read_excel(path)
                            preview = f"Excel preview for {name}:\n" + df.head().to_string()
                        elif name.lower().endswith('.json'):
                            with open(path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            preview = f"JSON preview for {name}:\n" + json.dumps(data)[:500]
                        elif name.lower().endswith('.pdf'):
                            reader = PyPDF2.PdfReader(path)
                            first_page = reader.pages[0].extract_text() if reader.pages else ''
                            preview = f"PDF preview for {name}:\n" + (first_page[:500] if first_page else '[No text found]')
                        elif name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                            img = Image.open(path)
                            preview = f"Image info for {name}: format={img.format}, size={img.size}"
                        elif name.lower().endswith('.txt'):
                            with open(path, 'r', encoding='utf-8') as f:
                                lines = ''.join([next(f) for _ in range(10)])
                            preview = f"Text preview for {name}:\n" + lines
                    except Exception as e:
                        preview = f"Could not preview {name}: {e}"
                    if preview:
                        file_previews.append(preview)
                if file_previews:
                    files_text += "\n\nFile previews:\n" + "\n\n".join(file_previews)
            
            # System prompt (unchanged)
            system = f"""
You are a data analyst agent. Solve the data analysis task by answering the questions. 
If there is a link in the question then there is absolutely no need to scrape it, the sacraping has already been done, the data has been provided to you.

CRITICAL RESPONSE RULES:
1. Follow the user's explicit output format exactly:
   - If they ask for "JSON array" or "JSON array of strings", return exactly that: ["answer1", "answer2", ...]
   - If they specify object keys like {{"Ans_of _question_1": ...}}, return exactly that object with the answer as the value.
   - If no specific format is mentioned, return answers as a simple array in question order.
   - If you don't know an answer, return "To Be Determined" (for both arrays and object values).

2. Output ONLY the JSON in the requested format—no explanations, no extra text.

3. If the JSON output structure is not provided, return a list of answers in correct order.

WHEN TO USE CODE:
- If you can answer directly from provided information, output the JSON immediately.
- If you need to compute statistics, query data (e.g., DuckDB on Parquet/S3), analyze files, or create plots:
  • Provide a COMPLETE Python code block in ```python ... ``` that performs the action and prints the result.
  • The code's printed output will be provided back to you; then incorporate it into your final JSON.
- Available imports: pd (pandas), np (numpy), plt (matplotlib.pyplot), duckdb, PdfReader (pypdf2), Image (PIL), io, requests, BeautifulSoup (bs4), re.
- Do not install new packages.
- Handle errors in code.

PLOTS:
- Use matplotlib and save plots as PNG files to '{temp_dir}', e.g., '{temp_dir}/plot1.png', '{temp_dir}/chart2.png', etc. Use unique filenames.
- Always call plt.tight_layout() before saving to ensure proper layout.
- Save with high DPI but keep file size under 100KB: plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
- Always call plt.close() after saving to free memory.
- In JSON output, include the PNG file path as a string.
- Do NOT encode images as base64 or embed them in your code - the system will handle conversion.

DUCKDB QUERIES:
- Example:
  SELECT court, COUNT(*) AS case_count
  FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1')
  WHERE year BETWEEN 2019 AND 2022
  GROUP BY court ORDER BY case_count DESC LIMIT 1;
- When querying S3 paths, use wildcards like year=* to match all years, and filter specific years with a WHERE clause.
- Filter date columns to only include years between 1900 and 2100.
- Use try_cast(column AS DATE) or CAST(SUBSTR(column,1,4) AS INTEGER) for safe filtering.
- In pandas, use errors='coerce' when converting to datetime.
- Column decision_date has datatype DATE in the format YYYY-MM-DD.
- Column date_of_registration has datatype varchar in the format DD-MM-YYYY.

FINAL STEP:
When ready, output ONLY the JSON in the exact format requested by the user.
""".strip()
            
            messages = [
                {"role": "system", "content": system + scraped_text + files_text},
                {"role": "user", "content": task}
            ]
            
            code_globals = {
                'pd': pd,
                'np': np,
                'plt': plt,
                'duckdb': duckdb,
                'PdfReader': PyPDF2.PdfReader,
                'Image': Image,
                'io': io,
                'requests': requests,
                'BeautifulSoup': BeautifulSoup,
                're': re,
                'temp_dir': temp_dir,  # Use temp_dir for saving files
                'seaborn': seaborn
            }
            
            max_iterations = 5  # Increased for better chance of success
            if "DuckDB" in task or "duckdb" in task:
                max_iterations = 7  # Allow more iterations for complex queries
            last_response = None
            last_error = None
            final_json_output = None  # Store the final JSON before temp_dir cleanup
            
            for iteration in range(1, max_iterations + 1):
                logging.info(f"Starting iteration {iteration}")
                try:
                    response = call_llm(messages)
                    last_response = response
                    logging.info(f"LLM response: {response[:500]}...")  # Log truncated response

                    code = extract_code(response)
                    if code:
                        logging.info(f"Extracted code: {code}...")
                        old_stdout = sys.stdout
                        sys.stdout = output_buf = io.StringIO()
                        try:
                            exec(code, code_globals)
                            output = output_buf.getvalue().strip()
                            logging.info(f"Code output: {output}...")
                        except Exception as e:
                            output = f"Execution error: {str(e)}\n{traceback.format_exc()}"
                            logging.error(f"Code execution failed: {output}")
                            last_error = output
                        finally:
                            sys.stdout = old_stdout
                        
                        messages = [
                            {"role": "system", "content": system + scraped_text + files_text + f"Don't stop if the answer is still 'To be determined' unless you are on iteration 5. Current iteration: {iteration}"},
                            {"role": "assistant", "content": response},
                            {"role": "user", "content": f"The code output: {output[:2000]}...\nIf error, fix and provide new code. If successful, provide the FINAL JSON only using file paths for images (not base64), no code or extra text."}
                        ]
                    else:
                        try:
                            json_match = re.search(r'(\[.*?\]|\{.*?\})', response, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(1)
                                json_output = json.loads(json_str)
                                
                                # Convert image paths to base64 BEFORE leaving temp_dir context
                                final_json_output = replace_image_paths_with_base64(json_output, temp_dir)
                                logging.info("Successfully parsed JSON output")
                                break  # Exit the loop, we have our result
                            else:
                                json_output = json.loads(response)
                                final_json_output = replace_image_paths_with_base64(json_output, temp_dir)
                                logging.info("Successfully parsed JSON output")
                                break
                        except json.JSONDecodeError as e:
                            logging.error(f"JSON parse error: {e}")
                            last_error = str(e)
                            messages = [
                                {"role": "system", "content": system + scraped_text + files_text},
                                {"role": "assistant", "content": response},
                                {"role": "user", "content": "Response not valid JSON. Provide ONLY the JSON as specified in the task."}
                            ]
                
                except Exception as e:
                    logging.error(f"Iteration {iteration} failed: {e}")
                    last_error = str(e)
                    if iteration < max_iterations:
                        messages.append({"role": "user", "content": f"Previous attempt failed with error: {str(e)}. Retry with corrected approach."})
                    else:
                        break
        
        # Return the result AFTER temp_dir cleanup
        if final_json_output is not None:
            print("Final result:", final_json_output)
            return jsonify(final_json_output)
        else:
            logging.error("Max iterations reached without success")
            error_msg = {"error": "Failed after max iterations", "last_response": last_response[:1000] if last_response else None, "last_error": last_error}
            return jsonify(error_msg), 500
    
    except TimeoutException:
        logging.error("Request timed out after 5 minutes")
        return jsonify({"error": "Request timed out after 5 minutes"}), 408
    
    finally:
        # Cancel the alarm
        signal.alarm(0)

# 4. Modify the final run statement to:
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host="0.0.0.0", port=port)

# 5. Add error handling for missing token:
if not aipipe_token:
    logging.warning("AIPIPE_TOKEN not set - API calls may fail")
    # Don't raise error, let it fail gracefully during actual API calls