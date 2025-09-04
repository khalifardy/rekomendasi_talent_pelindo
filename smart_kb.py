# smart_kb.py - Smart Knowledge Base Module with Token Optimization

import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
from anthropic import Anthropic
import streamlit as st

class DocumentChunker:
    """Memecah dokumen menjadi chunks yang manageable"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize chunker
        
        Args:
            chunk_size: Target size for each chunk in characters
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk_text(self, text: str, doc_name: str) -> List[Dict]:
        """
        Split text into overlapping chunks with metadata
        
        Args:
            text: The document text to chunk
            doc_name: Name of the source document
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_chunk_sentences = []
        chunk_id = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
                current_chunk_sentences.append(sentence)
            else:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append({
                        'id': f"{doc_name}_chunk_{chunk_id}",
                        'text': current_chunk.strip(),
                        'doc_name': doc_name,
                        'chunk_id': chunk_id,
                        'sentences': current_chunk_sentences.copy()
                    })
                    chunk_id += 1
                
                # Start new chunk with overlap
                if self.overlap > 0 and len(current_chunk_sentences) > 2:
                    # Calculate how many sentences to overlap
                    overlap_count = max(1, len(current_chunk_sentences) // 4)
                    overlap_sentences = current_chunk_sentences[-overlap_count:]
                    current_chunk = " ".join(overlap_sentences) + " " + sentence + " "
                    current_chunk_sentences = overlap_sentences + [sentence]
                else:
                    current_chunk = sentence + " "
                    current_chunk_sentences = [sentence]
        
        # Add the last chunk
        if current_chunk:
            chunks.append({
                'id': f"{doc_name}_chunk_{chunk_id}",
                'text': current_chunk.strip(),
                'doc_name': doc_name,
                'chunk_id': chunk_id,
                'sentences': current_chunk_sentences
            })
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Improved sentence splitting regex
        sentence_endings = r'[.!?]+'
        sentences = re.split(f'(?<={sentence_endings})\\s+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for s in sentences:
            s = s.strip()
            if s and len(s) > 10:  # Filter out very short fragments
                cleaned_sentences.append(s)
        
        return cleaned_sentences


class SemanticRetriever:
    """Retrieve chunks most relevant to a query using TF-IDF similarity"""
    
    def __init__(self):
        """Initialize the retriever with TF-IDF vectorizer"""
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        self.chunk_vectors = None
        self.chunks = []
        
    def index_chunks(self, chunks: List[Dict]):
        """
        Create TF-IDF vector index from chunks
        
        Args:
            chunks: List of chunk dictionaries
        """
        self.chunks = chunks
        
        if not chunks:
            return
            
        texts = [chunk['text'] for chunk in chunks]
        
        if texts:
            try:
                self.chunk_vectors = self.vectorizer.fit_transform(texts)
            except Exception as e:
                st.warning(f"Error creating vector index: {str(e)}")
                self.chunk_vectors = None
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-k most relevant chunks for a query
        
        Args:
            query: The search query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunk dictionaries with similarity scores
        """
        if not self.chunks or self.chunk_vectors is None:
            return []
        
        try:
            # Transform query to vector
            query_vector = self.vectorizer.transform([query])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_vector, self.chunk_vectors).flatten()
            
            # Get top-k indices
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            # Build result list with similarity scores
            relevant_chunks = []
            for idx in top_indices:
                if similarities[idx] > 0.05:  # Minimum similarity threshold
                    chunk = self.chunks[idx].copy()
                    chunk['similarity_score'] = float(similarities[idx])
                    relevant_chunks.append(chunk)
            
            return relevant_chunks
            
        except Exception as e:
            st.warning(f"Error retrieving chunks: {str(e)}")
            return []


class TokenManager:
    """Manage token counting and optimization"""
    
    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", max_tokens: int = 50000):
        """
        Initialize token manager
        
        Args:
            model_name: Name of the model (for token counting)
            max_tokens: Maximum allowed tokens
        """
        self.max_tokens = max_tokens
        self.reserved_tokens = 5000  # Reserve for response
        
        # Use tiktoken for token counting
        try:
            # Try to get encoding for the specific model
            self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except:
            # Fallback to cl100k_base encoding
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
            
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            # Fallback to character-based estimation
            return len(text) // 4
    
    def optimize_context(self, chunks: List[Dict], max_context_tokens: int = 20000) -> str:
        """
        Build optimized context from chunks within token limits
        
        Args:
            chunks: List of chunk dictionaries
            max_context_tokens: Maximum tokens for context
            
        Returns:
            Optimized context string
        """
        optimized_text = ""
        current_tokens = 0
        
        for chunk in chunks:
            chunk_text = f"\n[Source: {chunk['doc_name']} - Part {chunk['chunk_id']}]\n{chunk['text']}\n"
            chunk_tokens = self.count_tokens(chunk_text)
            
            if current_tokens + chunk_tokens <= max_context_tokens:
                optimized_text += chunk_text
                current_tokens += chunk_tokens
            else:
                # Stop if we've reached the token limit
                break
        
        return optimized_text


def create_document_summary(text: str, doc_name: str, client: Anthropic) -> str:
    """
    Create a concise summary of a document using Claude
    
    Args:
        text: Document text to summarize
        doc_name: Name of the document
        client: Anthropic client
        
    Returns:
        Summary text
    """
    try:
        # Limit text length for summarization
        max_chunk_size = 10000
        if len(text) > max_chunk_size:
            text = text[:max_chunk_size] + "\n\n[Document truncated for summarization]"
        
        response = client.messages.create(
            model="claude-3-haiku-20240307",  # Use cheaper model
            max_tokens=1000,
            temperature=0,
            messages=[{
                "role": "user",
                "content": f"""Please create a concise summary of this document.
                
Focus on:
1. Main topics and key information
2. Important data, numbers, or facts
3. Key decisions or conclusions

Document: {doc_name}

Content:
{text}

Provide a structured summary in 200-300 words."""
            }]
        )
        
        return response.content[0].text
        
    except Exception as e:
        st.warning(f"Could not summarize {doc_name}: {str(e)}")
        # Return truncated text as fallback
        return text[:2000] + "..." if len(text) > 2000 else text


class SmartKnowledgeBase:
    """Main class for managing the smart knowledge base"""
    
    def __init__(self, client: Anthropic, chunk_size: int = 1500, overlap: int = 200):
        """
        Initialize the smart knowledge base
        
        Args:
            client: Anthropic client for summarization
            chunk_size: Size of document chunks
            overlap: Overlap between chunks
        """
        self.client = client
        self.chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)
        self.retriever = SemanticRetriever()
        self.token_manager = TokenManager()
        
        self.all_chunks = []
        self.summaries = {}
        self.full_documents = {}
        
    def add_document(self, text: str, doc_name: str, doc_type: str = "pdf"):
        """
        Add a document to the knowledge base
        
        Args:
            text: Document text
            doc_name: Name of the document
            doc_type: Type of document (pdf, excel, etc.)
        """
        if not text:
            return
            
        # Store full document
        self.full_documents[doc_name] = {
            'text': text,
            'type': doc_type,
            'token_count': self.token_manager.count_tokens(text)
        }
        
        # Create chunks
        chunks = self.chunker.chunk_text(text, doc_name)
        self.all_chunks.extend(chunks)
        
        # Create summary for large documents
        if len(text) > 5000:
            with st.spinner(f"Creating summary for {doc_name}..."):
                summary = create_document_summary(text, doc_name, self.client)
                self.summaries[doc_name] = summary
        
        # Re-index all chunks
        self.retriever.index_chunks(self.all_chunks)
        
    def get_relevant_context(self, query: str, max_chunks: int = 10) -> Tuple[str, List[str]]:
        """
        Get relevant context for a query
        
        Args:
            query: The user's query
            max_chunks: Maximum number of chunks to retrieve
            
        Returns:
            Tuple of (context string, list of sources)
        """
        # Retrieve relevant chunks
        relevant_chunks = self.retriever.retrieve_relevant_chunks(query, top_k=max_chunks)
        
        context = "=== RELEVANT KNOWLEDGE FROM DOCUMENTS ===\n\n"
        sources = []
        
        # Add summaries if available
        if self.summaries:
            context += "📄 Document Summaries:\n"
            for doc_name, summary in list(self.summaries.items())[:3]:  # Limit to 3 summaries
                context += f"\n[{doc_name} - Summary]\n{summary}\n"
                sources.append(f"{doc_name} (Summary)")
            context += "\n" + "="*50 + "\n"
        
        # Add relevant chunks
        if relevant_chunks:
            context += "\n📌 Relevant Excerpts:\n"
            for chunk in relevant_chunks:
                context += f"\n[{chunk['doc_name']} - Section {chunk['chunk_id']}] "
                context += f"(Relevance: {chunk['similarity_score']:.2%})\n"
                context += chunk['text'] + "\n"
                context += "-"*30 + "\n"
                sources.append(f"{chunk['doc_name']} - Section {chunk['chunk_id']}")
        
        # Optimize context to fit token limits
        optimized_context = self.token_manager.optimize_context(
            relevant_chunks, 
            max_context_tokens=15000
        )
        
        return optimized_context, sources
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the knowledge base
        
        Returns:
            Dictionary with statistics
        """
        total_tokens = sum(
            doc['token_count'] 
            for doc in self.full_documents.values()
        )
        
        return {
            'total_documents': len(self.full_documents),
            'total_chunks': len(self.all_chunks),
            'total_summaries': len(self.summaries),
            'estimated_total_tokens': total_tokens,
            'documents': list(self.full_documents.keys())
        }
    
    def clear(self):
        """Clear all data from the knowledge base"""
        self.all_chunks = []
        self.summaries = {}
        self.full_documents = {}
        self.retriever = SemanticRetriever()


class ConversationManager:
    """Manage conversation history to optimize token usage"""
    
    def __init__(self, max_history_tokens: int = 10000):
        """
        Initialize conversation manager
        
        Args:
            max_history_tokens: Maximum tokens for conversation history
        """
        self.max_history_tokens = max_history_tokens
        self.token_manager = TokenManager()
        
    def optimize_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        Optimize message history to fit within token budget
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Optimized list of messages
        """
        if not messages:
            return []
        
        optimized = []
        total_tokens = 0
        
        # Process messages in reverse order (keep recent ones)
        for msg in reversed(messages):
            msg_tokens = self.token_manager.count_tokens(msg['content'])
            
            if total_tokens + msg_tokens <= self.max_history_tokens:
                optimized.insert(0, msg)
                total_tokens += msg_tokens
            else:
                # Add a summary message for omitted history
                if len(messages) - len(optimized) > 2:
                    summary_msg = {
                        'role': 'system',
                        'content': f"[Note: {len(messages) - len(optimized)} earlier messages omitted to manage context size]"
                    }
                    optimized.insert(0, summary_msg)
                break
        
        return optimized


def format_messages_with_smart_kb(
    messages: List[Dict], 
    system_prompt: str, 
    kb_manager: Optional[SmartKnowledgeBase], 
    query: str
) -> Tuple[str, List[Dict]]:
    """
    Format messages for Claude API with smart knowledge base retrieval
    
    Args:
        messages: Conversation history
        system_prompt: System prompt for the role
        kb_manager: Smart knowledge base manager
        query: Current user query
        
    Returns:
        Tuple of (formatted system prompt, formatted messages)
    """
    full_system_prompt = system_prompt
    
    # Add relevant context from knowledge base
    if kb_manager and kb_manager.all_chunks:
        context, sources = kb_manager.get_relevant_context(query)
        
        if context:
            full_system_prompt += f"""

You have access to relevant information from uploaded documents.
Sources being used: {', '.join(sources[:5])}

{context}

Instructions for using the knowledge base:
1. Use the provided context to answer questions accurately
2. Always cite the specific source section when using information
3. If the requested information is not in the provided context, clearly state that
4. Be specific and use exact values/data from the documents when available
"""
    
    # Optimize conversation history
    conv_manager = ConversationManager(max_history_tokens=10000)
    optimized_messages = conv_manager.optimize_messages(messages)
    
    # Format messages for Claude API
    claude_messages = []
    for msg in optimized_messages:
        role = "user" if msg["role"] == "user" else "assistant"
        claude_messages.append({
            "role": role,
            "content": msg["content"]
        })
    
    return full_system_prompt, claude_messages