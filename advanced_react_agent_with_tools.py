import io
import uuid
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tracers.stdout import ConsoleCallbackHandler
import assemblyai as aai
import os
from pydantic import BaseModel, Field
import requests
import base64
import re
import json
import pandas as pd
import pdfplumber
import tempfile
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BROWSERLESS_TOKEN = os.getenv("BROWSERLESS_TOKEN")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

aai.settings.api_key = ASSEMBLYAI_API_KEY


@tool
def fetch_html(url: str) -> str:
    """Fetch raw HTML and auto-extract/decode base64 instructions from <script> tags."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        full_text = resp.text[:3000]

        # Auto-extract base64 from atob(`...`)
        decoded_blocks = []
        for script in soup.find_all("script"):
            if script.string:
                matches = re.findall(r'atob\(`(.*?)`\)', script.string, re.DOTALL)
                for b64 in matches:
                    clean_b64 = re.sub(r'\s+', '', b64)
                    try:
                        decoded = base64.b64decode(clean_b64).decode("utf-8", errors="replace")
                        decoded_blocks.append(f"DECODED INSTRUCTION:\n{decoded}")
                    except Exception:
                        continue

        if decoded_blocks:
            full_text += "\n\n" + "\n\n".join(decoded_blocks)
        return full_text
    except Exception as e:
        return f"HTML fetch failed: {e}"

@tool
def download_and_extract_file(url: str, page: int = None) -> str:
    """Download any file and extract structured data (PDF tables, CSV, JSON, etc.)."""
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        ext = os.path.splitext(url)[1].lower() or ".bin"
        suffix = ext if ext != ".bin" else (".pdf" if "pdf" in resp.headers.get("content-type", "") else ".csv")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(resp.content)
            path = tmp.name

        # Extract content
        if path.endswith(".pdf") and page is not None:
            with pdfplumber.open(path) as pdf:
                if page > len(pdf.pages):
                    return f"PDF has {len(pdf.pages)} pages; requested page {page} invalid."
                p = pdf.pages[page - 1]
                tables = p.extract_tables()
                text = p.extract_text()[:800]
                out = f"TEXT (page {page}):\n{text}\n"
                if tables:
                    out += f"\nTABLE (first 5 rows):\n{json.dumps(tables[0][:5] if tables[0] else [], indent=2)}"
                return out
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
            return f"CSV shape: {df.shape}\nSample:\n{df.head(3).to_json(indent=2)}"
        elif path.endswith(".json"):
            with open(path) as f:
                data = json.load(f)
            return f"JSON preview:\n{json.dumps(data, indent=2)[:600]}"
        else:
            return f"File downloaded: {path} (size: {len(resp.content)} bytes)"
    except Exception as e:
        return f"File processing failed: {e}"

# @tool
# def analyze_csv(url: str, operation: str) -> str:
    """
    Perform dynamic, safe analysis on a CSV file.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; QuizAgent/1.0)"}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        import io
        df = pd.read_csv(io.BytesIO(resp.content))

        if operation in ("total_rows", "count"):
            return str(len(df))

        elif operation.startswith("sum:"):
            col = operation.split(":", 1)[1]
            if col not in df.columns:
                return f"Column '{col}' not found. Available: {list(df.columns)}"
            return str(float(df[col].sum()))

        elif operation.startswith("mean:"):
            col = operation.split(":", 1)[1]
            if col not in df.columns:
                return f"Column '{col}' not found. Available: {list(df.columns)}"
            return str(float(df[col].mean()))

        elif operation.startswith("sum_where:") or operation.startswith("count_where:"):
            is_sum = operation.startswith("sum_where:")
            prefix = "sum_where:" if is_sum else "count_where:"
            clause = operation[len(prefix):].strip()
            if not clause:
                return "Empty condition in operation"

            # Supported operators (ordered by length to avoid > vs >= conflict)
            for op in [">=", "<=", "==", "!=", ">", "<"]:
                if op in clause:
                    parts = clause.split(op, 1)
                    if len(parts) != 2:
                        return f"Invalid clause format: {clause}"
                    col, val_str = parts[0].strip(), parts[1].strip()
                    if not col:
                        return "Missing column name"
                    if not val_str:
                        return "Missing value in condition"

                    if col not in df.columns:
                        return f"Column '{col}' not found. Available: {list(df.columns)}"

                    # Try to auto-convert value
                    try:
                        val = float(val_str)
                        if val.is_integer():
                            val = int(val)
                    except ValueError:
                        val = val_str  # keep as string

                    # Build mask
                    try:
                        if op == ">":
                            mask = df[col] > val
                        elif op == ">=":
                            mask = df[col] >= val
                        elif op == "<":
                            mask = df[col] < val
                        elif op == "<=":
                            mask = df[col] <= val
                        elif op == "==":
                            mask = df[col] == val
                        elif op == "!=":
                            mask = df[col] != val
                        else:
                            return "Unsupported operator"
                    except Exception as e:
                        return f"Filter error: {e}"

                    filtered = df[mask]
                    if is_sum:
                        if filtered.empty:
                            result = 0.0
                        else:
                            result = float(filtered[col].sum())
                    else:
                        result = len(filtered)
                    return str(result)

            return f"No valid operator found in: {clause}"

        else:
            return f"Unsupported operation: {operation}"

    except Exception as e:
        return f"CSV analysis failed: {str(e)}"


class CSVEvalInput(BaseModel):
    """Input schema for CSV evaluation tool."""
    url: str = Field(description="URL of the CSV file to analyze")
    expression: str = Field(
        description=(
            "Python/Pandas expression to evaluate. The DataFrame is available as 'df'. "
            "Examples: "
            "'df[df['values'] >= 25160]['values'].sum()', "
            "'df['price'].mean()', "
            "'df[(df['age'] > 18) & (df['salary'] < 50000)]', "
            "'df.groupby('category')['sales'].sum()', "
            "'df['column'].value_counts()', "
            "'len(df[df['status'] == \"active\"])', "
            "'df.describe()'"
        )
    )

@tool(args_schema=CSVEvalInput)
def eval_csv(url: str, expression: str) -> str:
    """
    Evaluate arbitrary Python/Pandas expressions on CSV data.
    
    The CSV is loaded into a DataFrame called 'df'. You can use any pandas
    operations directly. Common libraries (pandas, numpy) are available.
    
    Args:
        url: URL of the CSV file
        expression: Python expression to evaluate (df is the DataFrame)
    
    Returns:
        String representation of the evaluation result
    """
    try:
        # Fetch CSV
        headers = {"User-Agent": "Mozilla/5.0 (compatible; DataAnalyzer/1.0)"}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        
        # Read CSV
        df = pd.read_csv(
            io.BytesIO(resp.content),
            skipinitialspace=True,
            encoding='utf-8-sig',
            header=None
        )
        
        # Strip whitespace from column names
        # df.columns = df.columns.str.strip()
        
        # Make numpy available in eval context
        eval_globals = {
            'df': df,
            'pd': pd,
            'len': len,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
        }
        
        # Evaluate the expression
        result = eval(expression, eval_globals)
        
        # Format output based on result type
        if isinstance(result, pd.DataFrame):
            if len(result) > 100:
                return f"Result: DataFrame with {len(result)} rows\n\nFirst 50 rows:\n{result.head(50).to_string(index=False)}\n\n... ({len(result)-50} more rows)"
            return result.to_string(index=False)
        
        elif isinstance(result, pd.Series):
            if len(result) > 100:
                return f"Result: Series with {len(result)} values\n\nFirst 50 values:\n{result.head(50).to_string()}\n\n... ({len(result)-50} more values)"
            return result.to_string()
        
        elif isinstance(result, (list, tuple)):
            return str(result)
        
        else:
            return str(result)
    
    except requests.RequestException as e:
        return f"Error fetching CSV: {str(e)}"
    except pd.errors.ParserError as e:
        return f"Error parsing CSV: {str(e)}"
    except SyntaxError as e:
        return f"Syntax error in expression: {str(e)}"
    except NameError as e:
        return f"Name error (probably undefined variable): {str(e)}"
    except KeyError as e:
        return f"Column not found: {str(e)}\nAvailable columns: {list(df.columns) if 'df' in locals() else 'N/A'}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


@tool
def scrape_dynamic_page(url: str) -> str:
    """
    Scrape a dynamic page using Browserless Chrome REST API's /content endpoint.
    
    This version uses the JSON-based /content endpoint with explicit headers,
    suitable for pages that rely on JavaScript to render content.
    """
    if not BROWSERLESS_TOKEN:
        return "Error: BROWSERLESS_TOKEN not set."

    api_url = f"https://production-sfo.browserless.io/content?token={BROWSERLESS_TOKEN}"
    
    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "application/json"
    }

    data = {
        "url": url,
    }

    try:
        resp = requests.post(api_url, headers=headers, json=data)
        if resp.status_code == 200:
            return resp.text
        else:
            return f"Browserless API error: {resp.status_code} - {resp.text[:200]}"
    except Exception as e:
        return f"Request failed: {e}"

@tool
def transcribe_audio(audio_url: str) -> str:
    """
    Transcribe audio from a URL using AssemblyAI.
    
    Args:
        audio_url (str): Publicly accessible URL to an audio file (e.g., MP3, WAV).
        
    Returns:
        str: The transcribed text, or an error message if transcription failed.
    """
    try:
        config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.universal)
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(audio_url)
        
        if transcript.status == aai.TranscriptStatus.error:
            return f"Transcription failed: {transcript.error}"
        return transcript.text
        
    except Exception as e:
        return f"Transcription error: {str(e)}"

@tool
def submit_answer(payload: dict, submit_url: str) -> str:
    """Submit answer using dynamic payload structure to the given URL."""
    try:
        # Resolve relative URLs (e.g., "/submit" → full URL)
        if submit_url.startswith("/"):
            submit_url = f"https://tds-llm-analysis.s-anand.net{submit_url}"
        resp = requests.post(submit_url, json=payload, timeout=10)
        return f"Status: {resp.status_code}\nResponse: {resp.text[:400]}"
    except Exception as e:
        return f"Submission failed: {e}"

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0.0,
    convert_system_message_to_human=True
)

tools = [scrape_dynamic_page, download_and_extract_file, eval_csv, transcribe_audio, submit_answer]

# Create modern reactive agent using create_agent (LangGraph under the hood) [[8]]
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are an autonomous quiz-solving agent for a dynamic, data-driven challenge system.

### Core Workflow:
1. Use the `scrape_dynamic_page` to get contents rendered using javascript.
2. **Decode** any base64 instructions (look for 'DECODED INSTRUCTION' in the response).
3. **Follow instructions exactly** — they specify:
   - What data to fetch (URLs, files, APIs)
   - What page (for PDFs) or fields to analyze
   - What computation to perform (sum, extract, visualize, etc.)
   - The **exact submission URL and payload structure**
4. Use `download_and_extract_file` for PDFs/CSVs/JSONs. Specify `page` if needed.
5. If the task involves a **CSV**, use:
   - `eval_csv` for sums, counts, filters (e.g., "sum_where:amount>100")
   - Do NOT use `download_and_extract_file` for numeric operations
6. If the task involves **PDF**, use `download_and_extract_file(page=...)`
7. **Construct the payload dynamically** — NEVER hardcode fields.
   - The payload structure is defined in the decoded instructions.
   - Always include "email", "secret", "url", and "answer" if mentioned.
   - The "answer" may be a number, string, boolean, or base64 image.
8. Submit using `submit_answer` with the **full payload dict** and **exact submit URL**.
9. If the response includes a new "url", solve that next quiz immediately.

### Critical Rules:
- The quiz page at https://tds-llm-analysis.s-anand.net/demo expects:
  { "email", "secret", "url": "/demo", "answer": "anything" }
- **Always resolve relative URLs** (e.g., "/submit" → full domain).
- You have **full autonomy** — do not ask for confirmation.
- Handle errors by re-analyzing instructions or retrying tools.
"""
)


def run_agent_verbose(user_input: str) -> str:
    print("\n🔍 Starting agent execution with input:")
    print(f"   \"{user_input[:100]}...\"\n")
    
    try:
        # Add a unique session ID for tracing
        session_id = str(uuid.uuid4())
        
        # Use ConsoleCallbackHandler to print all steps
        result = agent.invoke(
            {
                "messages": [HumanMessage(content=user_input)]
            },
            config={
                "callbacks": [ConsoleCallbackHandler()],
                "run_name": "QuizAgentRun",
                "configurable": {"session_id": session_id}
            }
        )
        
        final_message = result["messages"][-1] if isinstance(result, dict) else result
        output = final_message.content if hasattr(final_message, "content") else str(final_message)
        
        print("\n✅ Final Agent Output:")
        return output
        
    except Exception as e:
        error_msg = f"\n💥 Agent execution failed: {e}"
        print(error_msg)
        return error_msg


def run_quiz_agent(email, secret, quiz_url):
    print("🚀 Modern Quiz Agent (LangChain + create_agent) Ready!")
    print("=" * 60)

    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_VERBOSE"] = "true"

    if not GOOGLE_API_KEY:
        print("❌ ERROR: GOOGLE_API_KEY not found in environment.")
        exit(1)

    user_task = f"""
Solve the quiz at {quiz_url}.
My email: {email}
My secret: {secret}
Begin by fetching the HTML and decoding any base64 instructions.
"""

    response = run_agent_verbose(user_task)
    print(response)
    print("\n" + "="*60)
