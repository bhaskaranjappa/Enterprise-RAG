# Create the LLM generation module
generation_code = '''"""
LLM generation module supporting multiple model types including local and hosted models.
Handles prompt formatting, response generation, and model management.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Generator
from abc import ABC, abstractmethod
import json
import requests
import time

from langchain_core.documents import Document


logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Base class for LLM implementations."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text response from prompt."""
        pass
    
    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """Generate streaming text response."""
        pass


class OllamaLLM(BaseLLM):
    """
    Ollama local LLM implementation.
    Supports models like Mixtral, Llama2, Mistral running locally via Ollama.
    """
    
    def __init__(
        self,
        model_name: str = "llama2",
        base_url: str = "http://localhost:11434",
        timeout: int = 120
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        self.api_url = f"{base_url}/api/generate"
        
        # Test connection
        self._test_connection()
        
        logger.info(f"Initialized Ollama LLM with model: {model_name}")
    
    def _test_connection(self):
        """Test connection to Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                available_models = [model["name"] for model in response.json().get("models", [])]
                logger.info(f"Connected to Ollama. Available models: {available_models}")
                
                if self.model_name not in available_models:
                    logger.warning(f"Model {self.model_name} not found. Available: {available_models}")
            else:
                logger.error(f"Failed to connect to Ollama: {response.status_code}")
        except requests.RequestException as e:
            logger.error(f"Cannot connect to Ollama at {self.base_url}: {e}")
            raise ConnectionError(f"Ollama not available at {self.base_url}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text using Ollama API."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                **kwargs
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.RequestException as e:
            logger.error(f"Ollama generation failed: {e}")
            raise
    
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """Generate streaming response using Ollama."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                **kwargs
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        if "response" in chunk:
                            yield chunk["response"]
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
                        
        except requests.RequestException as e:
            logger.error(f"Ollama streaming failed: {e}")
            raise


class HuggingFaceLLM(BaseLLM):
    """
    Hugging Face transformers LLM implementation.
    Supports local inference with various open-source models.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        device: str = "auto",
        torch_dtype: str = "auto"
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        
        self._load_model()
        logger.info(f"Initialized Hugging Face LLM: {model_name}")
    
    def _load_model(self):
        """Load the Hugging Face model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            
            # Determine device
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            
            # Create generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if device == "cuda" else -1
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except ImportError as e:
            logger.error(f"Transformers library not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load Hugging Face model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text using Hugging Face pipeline."""
        try:
            result = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            
            generated_text = result[0]["generated_text"]
            # Remove the original prompt from the response
            response = generated_text[len(prompt):].strip()
            return response
            
        except Exception as e:
            logger.error(f"Hugging Face generation failed: {e}")
            raise
    
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """Generate streaming response (simulated for Hugging Face)."""
        # Hugging Face doesn't natively support streaming, so we simulate it
        response = self.generate(prompt, max_tokens, temperature, **kwargs)
        
        # Yield response in chunks
        words = response.split()
        for i, word in enumerate(words):
            if i > 0:
                yield " "
            yield word
            time.sleep(0.05)  # Simulate streaming delay


class RAGGenerator:
    """
    Main RAG generation class that combines retrieval with language generation.
    Handles prompt formatting, context building, and response generation.
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        max_context_length: int = 4000,
        prompt_template: Optional[str] = None
    ):
        self.llm = llm
        self.max_context_length = max_context_length
        
        # Default prompt template
        self.prompt_template = prompt_template or """You are a helpful assistant. Use the following context to answer the user's question. If you cannot find the answer in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
        
        logger.info("Initialized RAG generator")
    
    def generate_answer(
        self,
        question: str,
        retrieved_documents: List[Document],
        max_tokens: int = 200,
        temperature: float = 0.7,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Generate answer using retrieved context.
        
        Args:
            question: User's question
            retrieved_documents: Documents retrieved from vector store
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            include_sources: Whether to include source information
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            # Build context from retrieved documents
            context = self._build_context(retrieved_documents)
            
            # Format prompt
            prompt = self.prompt_template.format(
                context=context,
                question=question
            )
            
            # Generate answer
            answer = self.llm.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Prepare response
            response = {
                "answer": answer.strip(),
                "question": question,
                "context_used": len(retrieved_documents) > 0,
                "num_sources": len(retrieved_documents)
            }
            
            if include_sources:
                response["sources"] = self._extract_sources(retrieved_documents)
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return {
                "answer": "I'm sorry, I encountered an error while generating the answer.",
                "question": question,
                "error": str(e),
                "context_used": False,
                "num_sources": 0
            }
    
    def generate_answer_stream(
        self,
        question: str,
        retrieved_documents: List[Document],
        max_tokens: int = 200,
        temperature: float = 0.7
    ) -> Generator[str, None, None]:
        """Generate streaming answer."""
        try:
            # Build context
            context = self._build_context(retrieved_documents)
            
            # Format prompt
            prompt = self.prompt_template.format(
                context=context,
                question=question
            )
            
            # Generate streaming response
            for chunk in self.llm.generate_stream(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            ):
                yield chunk
                
        except Exception as e:
            logger.error(f"Failed to generate streaming answer: {e}")
            yield "I'm sorry, I encountered an error while generating the answer."
    
    def _build_context(self, documents: List[Document]) -> str:
        """
        Build context string from retrieved documents.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            # Create context chunk with source info
            source_info = ""
            if "title" in doc.metadata:
                source_info = f"[Source: {doc.metadata['title']}] "
            
            chunk = f"{source_info}{doc.page_content}"
            
            # Check if adding this chunk would exceed context limit
            if current_length + len(chunk) > self.max_context_length:
                if current_length == 0:  # First chunk is too long, truncate it
                    available_space = self.max_context_length - len(source_info) - 10
                    truncated_content = doc.page_content[:available_space] + "..."
                    chunk = f"{source_info}{truncated_content}"
                    context_parts.append(chunk)
                break
            
            context_parts.append(chunk)
            current_length += len(chunk) + 2  # +2 for separator
        
        return "\\n\\n".join(context_parts)
    
    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract source information from documents."""
        sources = []
        seen_sources = set()
        
        for doc in documents:
            # Create source identifier
            source_id = doc.metadata.get("file_path", doc.metadata.get("title", "unknown"))
            
            if source_id not in seen_sources:
                source_info = {
                    "title": doc.metadata.get("title", "Unknown"),
                    "source_type": doc.metadata.get("source_type", "unknown"),
                    "file_path": doc.metadata.get("file_path", ""),
                }
                
                if "similarity_score" in doc.metadata:
                    source_info["relevance_score"] = round(doc.metadata["similarity_score"], 3)
                
                sources.append(source_info)
                seen_sources.add(source_id)
        
        return sources
    
    def update_prompt_template(self, new_template: str):
        """Update the prompt template."""
        self.prompt_template = new_template
        logger.info("Updated prompt template")


def create_llm(llm_type: str = "ollama", **kwargs) -> BaseLLM:
    """
    Factory function to create LLM instances.
    
    Args:
        llm_type: Type of LLM ("ollama", "huggingface")
        **kwargs: Additional arguments for LLM initialization
        
    Returns:
        LLM instance
    """
    if llm_type.lower() == "ollama":
        return OllamaLLM(**kwargs)
    elif llm_type.lower() == "huggingface":
        return HuggingFaceLLM(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")
'''

# Save the generation module
with open('src/rag_system/generation/llm_manager.py', 'w') as f:
    f.write(generation_code)

print("✅ LLM generation module created")