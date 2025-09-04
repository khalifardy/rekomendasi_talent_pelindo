# main_smart.py - Fixed Version with OCR Support
import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
from anthropic import Anthropic
from datetime import datetime
import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile

# === LIBRARY CHECKS ===
# PDF Libraries
try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# Document Libraries
try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# OCR Libraries
HAS_TESSERACT = False
HAS_EASYOCR = False

try:
    import pytesseract
    from PIL import Image
    import pdf2image
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

MODEL_SEKUENSIAL = {
	'AILO-3.2': "claude-opus-4-1-20250805",
	'AILO-3.1' : "claude-opus-4-20250514",
	'AILO-3' : "claude-3-opus-20240229",
	'AILO-2.1': "claude-3-5-sonnet-20241022",
	'AILO-2': "claude-3-sonnet-20240229",
	'AILO-1' : "claude-3-haiku-20240307"

}

# Load environment
load_dotenv()

# Initialize client
api_key = os.getenv("ANTHROPIC_API_KEY")
if api_key:
    client = Anthropic(api_key=api_key)
else:
    client = None
    st.error("Please set ANTHROPIC_API_KEY in .env file")

# Initialize OCR reader if available
@st.cache_resource
def init_ocr_reader():
    """Initialize OCR reader (cached for performance)"""
    if HAS_EASYOCR:
        try:
            reader = easyocr.Reader(['en', 'id'])  # English + Indonesian
            return reader
        except:
            pass
    return None

ocr_reader = init_ocr_reader() if HAS_EASYOCR else None

# Constants
MAX_CONTEXT_TOKENS = 45000
CHUNK_SIZE_TOKENS = 5000
TOKENS_PER_CHAR = 0.25
CHUNK_OVERLAP = 500

# Roles
ROLES = {
    "General Assistant": {
        "system_prompt": "You are a helpful AI assistant. Be friendly, informative, and professional.",
        "icon": "🤖",
    },
    "Human Resources": {
        "system_prompt": """You are an HR specialist focused on data analysis and promotion evaluations. You should:
        - Analyze employee performance data objectively and thoroughly
        - Consider all metrics: experience, skills, achievements, performance scores
        - Provide specific, data-driven recommendations with citations
        - Compare employees fairly based on quantitative metrics
        - Identify top performers and areas for improvement
        - Use exact numbers and percentages from the data
        - Make promotion recommendations based on clear criteria""",
        "icon": "📊",
    },
    "Data Analyst": {
        "system_prompt": """You are a data analyst expert. You should:
        - Perform thorough statistical analysis on all data
        - Identify trends, patterns, and outliers
        - Provide specific numbers and percentages
        - Create clear summaries of complex datasets
        - Make data-driven recommendations
        - Always cite exact figures from the source""",
        "icon": "📈",
    },
}

# ========== SMART DOCUMENT MANAGER ==========
class SmartDocumentManager:
    """Enhanced document manager with chunking and semantic search"""
    
    def __init__(self):
        self.documents = {}
        self.chunks = []
        self.chunk_embeddings = None
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.summaries = {}
        self.total_tokens = 0
        self.active_context = ""
    
    def split_into_chunks(self, text, chunk_size=None):
        """Split text into overlapping chunks"""
        if chunk_size is None:
            chunk_size = int(CHUNK_SIZE_TOKENS / TOKENS_PER_CHAR)
        
        chunks = []
        text_length = len(text)
        
        if text_length <= chunk_size:
            return [text]
        
        start = 0
        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            if end < text_length:
                last_para = text[start:end].rfind('\n\n')
                if last_para > chunk_size * 0.5:
                    end = start + last_para
                else:
                    last_period = text[start:end].rfind('. ')
                    if last_period > chunk_size * 0.5:
                        end = start + last_period + 1
            
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - CHUNK_OVERLAP if end < text_length else end
        
        return chunks
    
    def create_summary(self, text, max_length=500):
        """Create summary of text"""
        words = text.split()[:100]
        return " ".join(words) + "..."
    
    def add_document(self, name, content, file_type):
        """Add document with smart chunking"""
        if not content:
            return None, 0, 0
        
        doc_id = hashlib.md5(f"{name}{len(content)}".encode()).hexdigest()[:8]
        estimated_tokens = int(len(content) * TOKENS_PER_CHAR)
        
        self.documents[doc_id] = {
            'name': name,
            'type': file_type,
            'content': content,
            'tokens': estimated_tokens,
            'added': datetime.now()
        }
        
        text_chunks = self.split_into_chunks(content)
        
        for i, chunk_text in enumerate(text_chunks):
            chunk_info = {
                'doc_id': doc_id,
                'doc_name': name,
                'chunk_index': i,
                'total_chunks': len(text_chunks),
                'content': chunk_text,
                'tokens': int(len(chunk_text) * TOKENS_PER_CHAR)
            }
            self.chunks.append(chunk_info)
        
        if estimated_tokens > MAX_CONTEXT_TOKENS:
            self.summaries[doc_id] = {
                'name': name,
                'summary': self.create_summary(content),
                'total_tokens': estimated_tokens,
                'num_chunks': len(text_chunks)
            }
        
        self.index_chunks()
        
        return doc_id, estimated_tokens, len(text_chunks)
    
    def index_chunks(self):
        """Create TF-IDF embeddings for all chunks"""
        if not self.chunks:
            return
        
        try:
            texts = [chunk['content'] for chunk in self.chunks]
            self.chunk_embeddings = self.vectorizer.fit_transform(texts)
        except Exception as e:
            st.error(f"Error indexing chunks: {e}")
    
    def search_relevant_chunks(self, query, max_tokens=40000, top_k=10):
        """Find most relevant chunks for the query"""
        if not self.chunks or self.chunk_embeddings is None:
            return []
        
        try:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.chunk_embeddings).flatten()
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            selected_chunks = []
            selected_docs = set()
            total_tokens = 0
            
            for idx in top_indices:
                if similarities[idx] < 0.1:
                    continue
                
                chunk = self.chunks[idx]
                chunk_tokens = chunk['tokens']
                
                if total_tokens + chunk_tokens <= max_tokens:
                    selected_chunks.append({
                        **chunk,
                        'relevance_score': similarities[idx]
                    })
                    selected_docs.add(chunk['doc_name'])
                    total_tokens += chunk_tokens
                
                if total_tokens >= max_tokens * 0.9:
                    break
            
            if self.summaries and total_tokens < max_tokens * 0.8:
                summary_text = "\n=== DOCUMENT SUMMARIES ===\n"
                for doc_id, summary_info in self.summaries.items():
                    if summary_info['name'] not in selected_docs:
                        summary_text += f"\n{summary_info['name']} ({summary_info['num_chunks']} parts):\n"
                        summary_text += f"{summary_info['summary']}\n"
                
                if len(summary_text) * TOKENS_PER_CHAR < (max_tokens - total_tokens):
                    selected_chunks.insert(0, {
                        'doc_name': 'SUMMARIES',
                        'content': summary_text,
                        'tokens': int(len(summary_text) * TOKENS_PER_CHAR)
                    })
            
            return selected_chunks
            
        except Exception as e:
            st.error(f"Error searching chunks: {e}")
            return []
    
    def get_context_for_query(self, query):
        """Get optimized context for a specific query"""
        relevant_chunks = self.search_relevant_chunks(query)
        
        if not relevant_chunks:
            return self.get_basic_context()
        
        context_parts = []
        
        for chunk in relevant_chunks:
            if chunk['doc_name'] == 'SUMMARIES':
                context_parts.append(chunk['content'])
            else:
                header = f"\n{'='*60}\n"
                header += f"DOCUMENT: {chunk['doc_name']}"
                if 'chunk_index' in chunk:
                    header += f" (Part {chunk['chunk_index'] + 1}/{chunk['total_chunks']})"
                if 'relevance_score' in chunk:
                    header += f" [Relevance: {chunk['relevance_score']:.2f}]"
                header += f"\n{'='*60}\n"
                
                context_parts.append(header + chunk['content'])
        
        return "\n".join(context_parts)
    
    def get_basic_context(self):
        """Get basic context when no query is provided"""
        parts = []
        tokens = 0
        
        if self.summaries:
            summary_text = "\n=== DOCUMENT SUMMARIES ===\n"
            for doc_id, info in self.summaries.items():
                summary_text += f"\n{info['name']}: {info['summary']}\n"
            parts.append(summary_text)
            tokens += len(summary_text) * TOKENS_PER_CHAR
        
        for doc_id, doc in self.documents.items():
            if doc['tokens'] < 10000 and tokens + doc['tokens'] < MAX_CONTEXT_TOKENS:
                parts.append(f"\n{'='*60}\nDOCUMENT: {doc['name']}\n{'='*60}\n{doc['content']}")
                tokens += doc['tokens']
        
        return "\n".join(parts)
    
    def get_stats(self):
        """Get statistics about stored documents"""
        return {
            'num_documents': len(self.documents),
            'num_chunks': len(self.chunks),
            'total_tokens': sum(doc['tokens'] for doc in self.documents.values()),
            'indexed': self.chunk_embeddings is not None
        }
    
    def remove_document(self, doc_id):
        """Remove document and its chunks"""
        if doc_id in self.documents:
            self.chunks = [c for c in self.chunks if c['doc_id'] != doc_id]
            del self.documents[doc_id]
            if doc_id in self.summaries:
                del self.summaries[doc_id]
            self.index_chunks()
    
    def clear_all(self):
        """Clear all documents"""
        self.documents = {}
        self.chunks = []
        self.chunk_embeddings = None
        self.summaries = {}
        self.total_tokens = 0
        self.active_context = ""

# ========== OCR FUNCTIONS ==========
def is_scanned_pdf(pdf_file):
    """Check if PDF is scanned (image-based)"""
    has_text = False
    
    if HAS_PYMUPDF:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.getvalue())
                tmp_path = tmp.name
            
            doc = fitz.open(tmp_path)
            for page_num in range(min(3, len(doc))):
                page = doc.load_page(page_num)
                text = page.get_text().strip()
                if len(text) > 50:
                    has_text = True
                    break
            doc.close()
            os.unlink(tmp_path)
        except:
            pass
    
    elif HAS_PYPDF2:
        try:
            pdf_file.seek(0)
            reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(min(3, len(reader.pages))):
                text = reader.pages[page_num].extract_text().strip()
                if len(text) > 50:
                    has_text = True
                    break
        except:
            pass
    
    return not has_text

def extract_text_with_ocr(pdf_file):
    """Extract text from scanned PDF using OCR"""
    if HAS_TESSERACT:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.getvalue())
                tmp_path = tmp.name
            
            images = pdf2image.convert_from_path(tmp_path, dpi=200)
            all_text = ""
            progress = st.progress(0)
            
            for i, image in enumerate(images):
                progress.progress((i + 1) / len(images), f"OCR Page {i + 1}/{len(images)}")
                try:
                    text = pytesseract.image_to_string(image, lang='ind+eng')
                except:
                    text = pytesseract.image_to_string(image)
                all_text += f"\n--- Page {i + 1} ---\n{text}\n"
            
            progress.empty()
            os.unlink(tmp_path)
            return all_text
        except Exception as e:
            st.error(f"Tesseract OCR failed: {e}")
    
    if HAS_EASYOCR and ocr_reader:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.getvalue())
                tmp_path = tmp.name
            
            images = pdf2image.convert_from_path(tmp_path, dpi=200)
            all_text = ""
            progress = st.progress(0)
            
            for i, image in enumerate(images):
                progress.progress((i + 1) / len(images), f"OCR Page {i + 1}/{len(images)}")
                img_array = np.array(image)
                results = ocr_reader.readtext(img_array)
                page_text = " ".join([text for _, text, prob in results if prob > 0.5])
                all_text += f"\n--- Page {i + 1} ---\n{page_text}\n"
            
            progress.empty()
            os.unlink(tmp_path)
            return all_text
        except Exception as e:
            st.error(f"EasyOCR failed: {e}")
    
    return None

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF (with OCR support for scanned PDFs)"""
    # Check if it's scanned
    if is_scanned_pdf(pdf_file):
        st.info("📸 Detected scanned PDF. Using OCR...")
        text = extract_text_with_ocr(pdf_file)
        if text:
            st.success("✅ OCR completed")
            return text
        else:
            st.error("❌ OCR failed. Please install OCR libraries.")
            return None
    
    # Regular text extraction
    text = ""
    
    if HAS_PYMUPDF:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.getvalue())
                tmp_path = tmp.name
            
            doc = fitz.open(tmp_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += f"\n--- Page {page_num + 1} ---\n{page.get_text()}"
            doc.close()
            os.unlink(tmp_path)
            
            if text.strip():
                return text
        except Exception as e:
            st.warning(f"PyMuPDF failed: {e}")
    
    if HAS_PYPDF2:
        try:
            pdf_file.seek(0)
            reader = PyPDF2.PdfReader(pdf_file)
            for page_num, page in enumerate(reader.pages):
                text += f"\n--- Page {page_num + 1} ---\n{page.extract_text()}"
            
            if text.strip():
                return text
        except Exception as e:
            st.error(f"PDF extraction failed: {e}")
    
    return None

# ========== OTHER FILE EXTRACTION FUNCTIONS ==========
def extract_text_from_excel(excel_file):
    """Extract text from Excel"""
    try:
        excel_data = pd.ExcelFile(excel_file)
        all_text = f"# EXCEL: {excel_file.name}\n{'='*60}\n\n"
        
        for sheet_name in excel_data.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            all_text += f"\n## SHEET: {sheet_name}\n"
            all_text += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n"
            
            if len(df) <= 100:
                all_text += "DATA:\n" + df.to_string(max_rows=None, max_cols=None) + "\n\n"
            else:
                all_text += "DATA SAMPLE (first 50 rows):\n"
                all_text += df.head(50).to_string(max_cols=None)
                all_text += f"\n... {len(df) - 50} more rows ...\n\n"
            
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                all_text += "STATISTICS:\n"
                all_text += df[numeric_cols].describe().to_string() + "\n\n"
        
        return all_text
    except Exception as e:
        st.error(f"Excel extraction error: {e}")
        return None

def extract_text_from_csv(csv_file):
    """Extract text from CSV"""
    try:
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
            try:
                csv_file.seek(0)
                df = pd.read_csv(csv_file, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            return None
        
        all_text = f"# CSV: {csv_file.name}\n{'='*60}\n\n"
        all_text += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n"
        
        if len(df) <= 100:
            all_text += df.to_string(max_rows=None, max_cols=None)
        else:
            all_text += "DATA SAMPLE:\n" + df.head(50).to_string(max_cols=None)
            all_text += f"\n... {len(df) - 50} more rows ...\n"
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            all_text += "\n\nSTATISTICS:\n"
            all_text += df[numeric_cols].describe().to_string()
        
        return all_text
    except Exception as e:
        st.error(f"CSV extraction error: {e}")
        return None

def extract_text_from_word(word_file):
    """Extract text from Word"""
    if not HAS_DOCX:
        st.error("python-docx not installed")
        return None
    
    try:
        doc = docx.Document(word_file)
        all_text = f"# WORD: {word_file.name}\n{'='*60}\n\n"
        
        for para in doc.paragraphs:
            if para.text.strip():
                all_text += para.text + "\n\n"
        
        return all_text
    except Exception as e:
        st.error(f"Word extraction error: {e}")
        return None

def extract_text_from_txt(txt_file):
    """Extract text from TXT file"""
    try:
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
            try:
                txt_file.seek(0)
                return txt_file.read().decode(encoding)
            except UnicodeDecodeError:
                continue
        return None
    except:
        return None

# ========== STREAMLIT APP ==========
st.title("🚀 Nexus Talent: Recommendation based on AI")
st.caption("Smart Rekomendasi Talent")

# Initialize document manager
if 'smart_doc_manager' not in st.session_state:
    st.session_state.smart_doc_manager = SmartDocumentManager()

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Status
    st.subheader("📊 System Status")
    col1, = st.columns(1)  # Note the comma
    with col1:
        if HAS_TESSERACT or HAS_EASYOCR:
            st.success("✅ OCR Ready")
        else:
            st.error("❌ No OCR")
    
    # OCR Status Detail
    if not HAS_TESSERACT and not HAS_EASYOCR:
        st.warning("⚠️ Cannot read scanned PDFs without OCR!")
        with st.expander("🔧 Install OCR"):
            st.code("""
# Option 1: EasyOCR (Easiest)
pip install easyocr pdf2image

# Option 2: Tesseract (Best for Indonesian)
# Install Tesseract first, then:
pip install pytesseract pdf2image
            """)
    
    # Stats
    stats = st.session_state.smart_doc_manager.get_stats()
    st.metric("Documents", stats['num_documents'])
    st.metric("Total Chunks", stats['num_chunks'])
    if stats['total_tokens'] > 0:
        st.metric("Total Tokens", f"{stats['total_tokens']:,}")
    
    # Model selection
    st.subheader("🧠 Model")
    model = st.selectbox(
        "Choose model:",
        ['AILO-3.2', 
         'AILO-3.1', 
         'AILO-3', 
         'AILO-2.1', 
         'AILO-2', 
         'AILO-1']
    )
    
    # Role
    role = "Human Resources"
    st.subheader("🎭 Role")
    st.info(f"{ROLES[role]['icon']} {role}")
    # Temperature
    temperature = st.slider("Temperature:", 0.0, 1.0, 0.3, 0.1)
    
    # Search testing
    if stats['num_chunks'] > 0:
        st.subheader("🔍 Test Search")
        test_query = st.text_input("Test query:")
        if test_query:
            with st.spinner("Searching..."):
                results = st.session_state.smart_doc_manager.search_relevant_chunks(
                    test_query, max_tokens=10000, top_k=3
                )
                if results:
                    st.write(f"Found {len(results)} relevant chunks:")
                    for r in results[:3]:
                        if 'relevance_score' in r:
                            st.caption(f"• {r['doc_name']} (Score: {r['relevance_score']:.2f})")
    
    # File upload
    st.subheader("📚 Document Upload")
    
    uploaded = st.file_uploader(
        "Upload files:",
        type=["pdf", "txt", "xlsx", "xls", "csv", "docx", "doc"],
        accept_multiple_files=True,
        help="Large files will be automatically chunked and indexed"
    )
    
    if uploaded and client:
        for file in uploaded:
            existing = any(
                doc['name'] == file.name 
                for doc in st.session_state.smart_doc_manager.documents.values()
            )
            
            if not existing:
                with st.spinner(f"Processing {file.name}..."):
                    text = None
                    ext = file.name.split('.')[-1].lower()
                    
                    if ext == 'pdf':
                        text = extract_text_from_pdf(file)
                    elif ext in ['xlsx', 'xls']:
                        text = extract_text_from_excel(file)
                    elif ext == 'csv':
                        text = extract_text_from_csv(file)
                    elif ext in ['docx', 'doc']:
                        text = extract_text_from_word(file)
                    elif ext == 'txt':
                        text = extract_text_from_txt(file)
                    
                    if text:
                        doc_id, tokens, num_chunks = st.session_state.smart_doc_manager.add_document(
                            file.name, text, ext
                        )
                        
                        if doc_id:
                            icon = {'pdf': '📕', 'xlsx': '📊', 'csv': '📈',
                                   'docx': '📄', 'txt': '📝'}.get(ext, '📎')
                            
                            if tokens > MAX_CONTEXT_TOKENS:
                                st.warning(f"{icon} {file.name}")
                                st.info(f"Large: {tokens:,} tokens → {num_chunks} chunks")
                            else:
                                st.success(f"{icon} {file.name}: {tokens:,} tokens")
                    else:
                        st.error(f"❌ Failed: {file.name}")
    
    # Document list
    if st.session_state.smart_doc_manager.documents:
        st.divider()
        st.write("**Loaded Documents:**")
        
        for doc_id, doc in st.session_state.smart_doc_manager.documents.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                icon = {'pdf': '📕', 'xlsx': '📊', 'csv': '📈',
                       'docx': '📄', 'txt': '📝'}.get(doc['type'], '📎')
                chunks = [c for c in st.session_state.smart_doc_manager.chunks 
                         if c['doc_id'] == doc_id]
                if len(chunks) > 1:
                    st.caption(f"{icon} {doc['name']} ({len(chunks)} chunks)")
                else:
                    st.caption(f"{icon} {doc['name']}")
            
            with col2:
                if st.button("🗑️", key=f"del_{doc_id}"):
                    st.session_state.smart_doc_manager.remove_document(doc_id)
                    st.rerun()
        
        if st.button("🗑️ Clear All"):
            st.session_state.smart_doc_manager.clear_all()
            st.session_state.messages = []
            st.rerun()

# Main interface
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"**Role:** {ROLES[role]['icon']} {role}")
with col2:
    st.markdown(f"**Model:** {model.split('-')[1].title()}")
with col3:
    stats = st.session_state.smart_doc_manager.get_stats()
    if stats['num_documents'] > 0:
        st.markdown(f"**Docs:** {stats['num_documents']} ({stats['num_chunks']} chunks)")

# Info bar
if stats['num_documents'] > 0:
    if stats['total_tokens'] > MAX_CONTEXT_TOKENS:
        st.info(f"💡 Smart mode: Using semantic search from {stats['total_tokens']:,} total tokens")
    else:
        st.success(f"✅ All documents loaded: {stats['total_tokens']:,} tokens")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about your documents..."):
    if not client:
        st.error("Please configure API key")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching relevant content..."):
                    context = st.session_state.smart_doc_manager.get_context_for_query(prompt)
                
                system = ROLES[role]["system_prompt"]
                
                if context:
                    system += f"""

You have access to relevant document sections:

{context}

IMPORTANT: Use specific data and numbers from these documents."""
                
                messages = [
                    {"role": "user" if m["role"] == "user" else "assistant", 
                     "content": m["content"]}
                    for m in st.session_state.messages
                ]
                
                placeholder = st.empty()
                full_response = ""
                
                with client.messages.stream(
                    model=MODEL_SEKUENSIAL.get(model),
                    max_tokens=4096,
                    temperature=temperature,
                    system=system,
                    messages=messages
                ) as stream:
                    for text in stream.text_stream:
                        full_response += text
                        placeholder.markdown(full_response + "▌")
                
                placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Clear chat
if st.button("🔄 Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Help
with st.expander("ℹ️ How It Works"):
    st.markdown("""
    ### Features:
    - **Smart Chunking**: Large documents split intelligently
    - **Semantic Search**: TF-IDF finds relevant sections
    - **OCR Support**: Reads scanned PDFs (if OCR installed)
    - **Token Management**: Handles unlimited document size
    
    ### OCR Setup:
    ```bash
    # Easy option:
    pip install easyocr pdf2image
    
    # Or Tesseract:
    pip install pytesseract pdf2image
    ```
    """)