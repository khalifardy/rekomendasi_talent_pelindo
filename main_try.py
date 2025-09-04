import streamlit as st
import os
from dotenv import load_dotenv
import PyPDF2
import tempfile
import pandas as pd
from anthropic import Anthropic
import json
from file_handling import pdf_read
from utils import sanitize_filename

# Import library tambahan untuk OCR
try:
    import fitz  # PyMuPDF - lebih robust untuk PDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    st.warning("PyMuPDF not installed. Install with: pip install PyMuPDF")

try:
    import pytesseract
    from PIL import Image
    import pdf2image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    st.warning("OCR libraries not installed. Install with: pip install pytesseract pillow pdf2image")

# Muat variabel lingkungan dari file .env
load_dotenv()

st.title("🤖 AI Assistant with Role-Play & Knowledge Base")

# Inisialisasi Claude client
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Daftar peran (role) yang telah ditentukan sebelumnya
ROLES = {
    "General Assistant": {
        "system_prompt": "You are a helpful AI assistant. Be friendly, informative, and professional.",
        "icon": "🤖",
    },
    "Customer Service": {
        "system_prompt": """You are a professional customer service representative. You should:
        - Be polite, empathetic, and patient
        - Focus on solving customer problems
        - Ask clarifying questions when needed
        - Offer alternatives and solutions
        - Maintain a helpful and positive tone
        - If you can't solve something, explain how to escalate""",
        "icon": "📞",
    },
    "Technical Support": {
        "system_prompt": """You are a technical support specialist. You should:
        - Provide clear, step-by-step technical solutions
        - Ask about system specifications and error messages
        - Suggest troubleshooting steps in logical order
        - Explain technical concepts in simple terms
        - Be patient with non-technical users""",
        "icon": "⚙️",
    },
    "Teacher/Tutor": {
        "system_prompt": """You are an educational tutor. You should:
        - Explain concepts clearly and simply
        - Use examples and analogies to aid understanding
        - Encourage learning and curiosity
        - Break down complex topics into manageable parts
        - Provide practice questions or exercises when appropriate""",
        "icon": "📚",
    },
    "Human Resources": {
        "system_prompt": """You are an HR specialist focused on promotion evaluations. You should:
        - Analyze employee performance data objectively
        - Consider experience, skills, and achievements
        - Provide fair and balanced recommendations
        - Explain your reasoning clearly
        - Follow company promotion criteria strictly
        - Suggest areas for improvement when needed""",
        "icon": "📊",
    },
}

# Fungsi yang telah diperbaiki untuk mengekstrak teks dari PDF
def extract_text_from_pdf(pdf_file,client):
    """Mengekstrak teks dari file PDF dengan dukungan OCR untuk scan"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name
        
        #text = ""
        # Pass original filename, not temp path!
        clean_name = sanitize_filename(pdf_file.name)
        
        # Call with clean filename
        result = pdf_read(client, tmp_file_path, clean_name)

        # Method 1: Coba PyMuPDF dulu (lebih baik untuk PDF kompleks)
        #if HAS_PYMUPDF:
            #try:
                #doc = fitz.open(tmp_file_path)
                #for page_num in range(len(doc)):
                    #page = doc.load_page(page_num)
                    #page_text = page.get_text()
                    
                    #if page_text.strip():  # Jika ada teks
                        #text += f"\n--- Page {page_num + 1} ---\n"
                        #text += page_text + "\n"
                    #else:  # Jika tidak ada teks (kemungkinan scan), gunakan OCR
                        #if HAS_OCR:
                            #st.info(f"Using OCR for page {page_num + 1} (scanned image detected)")
                            # Convert page ke image
                            #pix = page.get_pixmap()
                            #img_data = pix.tobytes("png")
                            
                            # Save temporary image
                            #with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as img_tmp:
                                #img_tmp.write(img_data)
                                #img_tmp_path = img_tmp.name
                            
                            # OCR pada image
                            #ocr_text = pytesseract.image_to_string(Image.open(img_tmp_path), lang='ind+eng')
                            #os.unlink(img_tmp_path)
                            
                            #if ocr_text.strip():
                                #text += f"\n--- Page {page_num + 1} (OCR) ---\n"
                                #text += ocr_text + "\n"
                        #else:
                            #text += f"\n--- Page {page_num + 1} (Image - OCR not available) ---\n"
                            #text += "[This page contains images/scanned content but OCR is not installed]\n"
                
                #doc.close()
                
                #if text.strip():
        os.unlink(tmp_file_path)
        return client
                    
            #except Exception as e:
                #st.warning(f"PyMuPDF failed: {e}, trying PyPDF2...")
        
        # Method 2: Fallback ke PyPDF2
        #try:
            #with open(tmp_file_path, "rb") as file:
                #pdf_reader = PyPDF2.PdfReader(file)
                
                #for page_num, page in enumerate(pdf_reader.pages):
                    #page_text = page.extract_text()
                    
                    #if page_text.strip():
                        #text += f"\n--- Page {page_num + 1} ---\n"
                        #text += page_text + "\n"
                    #else:
                        # Jika PyPDF2 tidak bisa ekstrak teks, coba OCR
                        #if HAS_OCR:
                            #st.info(f"Using OCR for page {page_num + 1}")
                            
                            # Convert PDF page ke image menggunakan pdf2image
                            #images = pdf2image.convert_from_path(
                                #tmp_file_path, 
                                #first_page=page_num + 1, 
                                #last_page=page_num + 1,
                                #dpi=200
                            #)
                            
                            #if images:
                                # OCR pada image
                                #ocr_text = pytesseract.image_to_string(images[0], lang='ind+eng')
                                #if ocr_text.strip():
                                    #text += f"\n--- Page {page_num + 1} (OCR) ---\n"
                                    #text += ocr_text + "\n"
                        #else:
                            #text += f"\n--- Page {page_num + 1} (Unable to extract) ---\n"
                            #text += "[This page may contain scanned content - OCR not available]\n"
                
        #except Exception as e:
            #st.error(f"PyPDF2 also failed: {e}")
            
            # Method 3: Last resort - Full OCR
            #if HAS_OCR:
                #st.info("Attempting full OCR extraction...")
                #try:
                    #images = pdf2image.convert_from_path(tmp_file_path, dpi=200)
                    
                    #for i, image in enumerate(images):
                        #ocr_text = pytesseract.image_to_string(image, lang='ind+eng')
                        #if ocr_text.strip():
                            #text += f"\n--- Page {i + 1} (Full OCR) ---\n"
                            #text += ocr_text + "\n"
                            
                #except Exception as ocr_error:
                    #st.error(f"OCR extraction failed: {ocr_error}")
                    #return None
            #else:
                #st.error("Cannot extract text from this PDF. OCR libraries are required for scanned documents.")
                #return None
        
        #os.unlink(tmp_file_path)
        #return text if text.strip() else None
        
    except Exception as e:
        st.error(f"Error extracting PDF text: {str(e)}")
        return None

# Fungsi untuk mengekstrak teks dari file Excel (tetap sama)
def extract_text_from_excel(excel_file):
    """Mengekstrak teks dari file Excel dengan format yang lebih baik"""
    try:
        excel_data = pd.ExcelFile(excel_file)
        all_text = ""
        
        for sheet_name in excel_data.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name, engine="openpyxl")
            
            all_text += f"\n\n=== SHEET: {sheet_name} ===\n"
            all_text += f"Columns: {', '.join(str(df.columns))}\n"
            all_text += f"Total rows: {len(df)}\n\n"
            
            all_text += "DATA:\n"
            
            for index, row in df.iterrows():
                row_text = f"Row {index + 1}: "
                row_items = []
                for col in df.columns:
                    value = row[col]
                    if pd.isna(value):
                        value = "N/A"
                    row_items.append(f"{col}={value}")
                row_text += " | ".join(row_items)
                all_text += row_text + "\n"
            
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                all_text += "\n--- NUMERIC SUMMARY ---\n"
                summary = df[numeric_cols].describe().to_string()
                all_text += summary + "\n"
        
        return all_text
    except Exception as e:
        st.error(f"Error extracting Excel text: {str(e)}")
        return None

# Format pesan untuk Claude API (tetap sama)
def format_messages_for_claude(messages, system_prompt, knowledge_base=""):
    """Format messages untuk Claude API dengan knowledge base yang diperbaiki"""
    full_system_prompt = system_prompt
    
    if knowledge_base:
        full_system_prompt += f"""

IMPORTANT: You have access to the following knowledge base from uploaded documents. Use this information to answer questions when relevant:

===== START OF KNOWLEDGE BASE =====
{knowledge_base}
===== END OF KNOWLEDGE BASE =====

Instructions for using the knowledge base:
1. When answering questions, prioritize information from the knowledge base when applicable.
2. Always mention which document/sheet the information came from.
3. If asked about data, provide specific values and details from the documents.
4. If the requested information is not in the knowledge base, clearly state that.
"""
    
    claude_messages = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "assistant"
        claude_messages.append({
            "role": role,
            "content": msg["content"]
        })
    
    return full_system_prompt, claude_messages

# --- Sidebar untuk Konfigurasi ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Library status
    st.subheader("📚 Library Status")
    if HAS_PYMUPDF:
        st.success("✅ PyMuPDF (Better PDF handling)")
    else:
        st.error("❌ PyMuPDF not installed")
        
    if HAS_OCR:
        st.success("✅ OCR Libraries (Scan support)")
    else:
        st.error("❌ OCR Libraries not installed")

    # Model selection
    st.subheader("🧠 Model Selection")
    claude_model = st.selectbox(
        "Choose Claude model:",
        options=[
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ],
        index=0
    )

    # Role selection
    st.subheader("🎭 Select Role")
    selected_role = st.selectbox(
        "Choose assistant role:", 
        options=list(ROLES.keys()), 
        index=0
    )

    # Temperature setting
    temperature = st.slider(
        "Temperature (creativity):",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )

    # Knowledge Base section
    st.subheader("📚 Knowledge Base")
    
    if st.button("🗑️ Clear Knowledge Base"):
        st.session_state.knowledge_base = ""
        st.session_state.uploaded_files_tracker = []
        st.success("Knowledge base cleared!")
    
    uploaded_files = st.file_uploader(
        "Upload PDF or Excel documents:",
        type=["pdf", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    # Track file yang sudah diupload
    if "uploaded_files_tracker" not in st.session_state:
        st.session_state.uploaded_files_tracker = []
    
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = ""

    # Proses file yang diupload
    if uploaded_files:
        files_to_process = []
        
        for file in uploaded_files:
            file_id = f"{file.name}_{file.size}"
            if file_id not in st.session_state.uploaded_files_tracker:
                files_to_process.append((file, file_id))
        
        if files_to_process:
            excel_mime_types = [
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/vnd.ms-excel",
                "application/wps-office.xlsx",
                "application/wps-office.xls",
                "application/octet-stream",
                "application/zip",
                "application/x-zip-compressed",
                "application/vnd.oasis.opendocument.spreadsheet",
            ]
            
            with st.spinner(f"Processing {len(files_to_process)} new file(s)..."):
                for uploaded_file, file_id in files_to_process:
                    st.write(f"📄 Processing: {uploaded_file.name}")
                    
                    if uploaded_file.type == "application/pdf":
                        pdf_text = extract_text_from_pdf(uploaded_file,client)
                        if pdf_text:
                            #st.session_state.knowledge_base += f"\n\n{'='*50}\nDOCUMENT: {uploaded_file.name} (PDF)\n{'='*50}\n{pdf_text}"
                            #st.session_state.uploaded_files_tracker.append(file_id)
                            st.success(f"✅ Successfully processed: {uploaded_file.name}")
                            client = pdf_text
                            print(client.beta.files.list())
                        else:
                            st.error(f"❌ Failed to process: {uploaded_file.name}")
                    
                    elif uploaded_file.type in excel_mime_types:
                        excel_text = extract_text_from_excel(uploaded_file)
                        if excel_text:
                            st.session_state.knowledge_base += f"\n\n{'='*50}\nDOCUMENT: {uploaded_file.name} (Excel)\n{'='*50}\n{excel_text}"
                            st.session_state.uploaded_files_tracker.append(file_id)
                            st.success(f"✅ Successfully processed: {uploaded_file.name}")
                        else:
                            st.error(f"❌ Failed to process: {uploaded_file.name}")

    # Tampilkan status knowledge base
    if st.session_state.knowledge_base:
        st.divider()
        word_count = len(st.session_state.knowledge_base.split())
        char_count = len(st.session_state.knowledge_base)
        st.metric("Knowledge Base Size", f"{word_count:,} words")
        st.metric("Characters", f"{char_count:,}")
        
        estimated_tokens = word_count * 1.3
        st.caption(f"~{int(estimated_tokens):,} tokens")
        
        if st.session_state.uploaded_files_tracker:
            st.caption("Loaded documents:")
            for file_id in st.session_state.uploaded_files_tracker:
                file_name = file_id.split('_')[0]
                st.caption(f"• {file_name}")
        
        with st.expander("📋 Preview Knowledge Base (first 500 chars)"):
            st.text(st.session_state.knowledge_base[:500] + "...")

# --- Inisialisasi Session State untuk Chat ---
if "claude_model" not in st.session_state:
    st.session_state["claude_model"] = claude_model

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_role" not in st.session_state:
    st.session_state.current_role = selected_role

if st.session_state.claude_model != claude_model:
    st.session_state.claude_model = claude_model

if st.session_state.current_role != selected_role:
    st.session_state.messages = []
    st.session_state.current_role = selected_role
    st.rerun()

# --- Main Chat Interface ---
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**Current Role:** {ROLES[selected_role]['icon']} {selected_role}")
with col2:
    st.markdown(f"**Model:** {claude_model.split('-')[1].title()}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What can I help you with?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        try:
            knowledge_base = st.session_state.get("knowledge_base", "")
            system_prompt, claude_messages = format_messages_for_claude(
                st.session_state.messages,
                ROLES[selected_role]["system_prompt"],
                knowledge_base
            )
            
            message_placeholder = st.empty()
            full_response = ""
            
            with client.messages.stream(
                model=st.session_state.claude_model,
                max_tokens=4096,
                temperature=temperature,
                system=system_prompt,
                messages=claude_messages
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            error_message = f"Error: {str(e)}"
            st.error(error_message)
            if "api_key" in str(e).lower():
                st.error("Please check your ANTHROPIC_API_KEY in the .env file")

# Clear chat button
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if st.button("🔄 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Instructions
with st.expander("ℹ️ How to use & Installation Guide"):
    st.markdown("""
    ### For PDF Scan Support (Required for scanned documents):
    
    ```bash
    # Install OCR libraries
    pip install pytesseract pillow pdf2image
    
    # Install PyMuPDF for better PDF handling
    pip install PyMuPDF
    
    # Install Tesseract OCR Engine
    # Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
    # macOS: brew install tesseract tesseract-lang
    # Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-ind
    ```
    
    ### Features:
    - **Text-based PDFs**: Extracted normally with PyPDF2/PyMuPDF
    - **Scanned PDFs**: Automatically detected and processed with OCR
    - **Multi-language OCR**: Supports Indonesian + English text recognition
    - **Fallback system**: Multiple methods tried if one fails
    - **Excel files**: Full support for all sheets and data types
    
    ### Troubleshooting:
    - If OCR doesn't work, check Tesseract installation
    - For Indonesian text, ensure 'ind' language pack is installed
    - Large PDF files may take longer to process with OCR
    """)
    
    if st.session_state.get("knowledge_base"):
        st.info(f"Knowledge base is loaded with {len(st.session_state.knowledge_base)} characters")