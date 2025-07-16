# services/llm_client.py
import logging
import time
from typing import Dict, List, Any, Optional
import json
import requests
import os

# LLM client libraries
try:
    from openai import AzureOpenAI, OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from config import Config

logger = logging.getLogger(__name__)

class LLMClient:
    """Client for various LLM providers"""
    
    def __init__(self):
        self.config = Config()
        self.llm_config = self.config.LLM_CONFIG
        self.provider = self.llm_config['provider']
        self.clients = {}
        self.initialized = False
        
        # Initialize clients based on configuration
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize LLM clients based on configuration"""
        try:
            if self.provider == 'azure_openai' and OPENAI_AVAILABLE:
                success = self._initialize_azure_client()
                if not success:
                    self._fallback_to_mock()
            elif self.provider == 'openai' and OPENAI_AVAILABLE:
                success = self._initialize_openai_client()
                if not success:
                    self._fallback_to_mock()
            elif self.provider == 'anthropic' and ANTHROPIC_AVAILABLE:
                success = self._initialize_anthropic_client()
                if not success:
                    self._fallback_to_mock()
            else:
                logger.warning(f"Provider {self.provider} not available, using mock client")
                self._initialize_mock_client()
                
        except Exception as e:
            logger.error(f"Error initializing LLM clients: {str(e)}")
            self._initialize_mock_client()
    
    def _initialize_azure_client(self) -> bool:
        """Initialize Azure OpenAI client"""
        try:
            azure_config = self.llm_config['azure']
            
            # Check for required environment variables or config
            api_key = azure_config.get('api_key') or os.getenv('AZURE_OPENAI_API_KEY')
            endpoint = azure_config.get('endpoint') or os.getenv('AZURE_OPENAI_ENDPOINT')
            
            if not api_key or not endpoint:
                logger.warning("Azure OpenAI API key or endpoint not provided")
                return False
            
            self.clients['azure'] = AzureOpenAI(
                api_key=api_key,
                api_version=azure_config['api_version'],
                azure_endpoint=endpoint
            )
            
            # Test the connection
            test_result = self._test_azure_connection()
            if test_result:
                logger.info("Azure OpenAI client initialized and tested successfully")
                self.initialized = True
                return True
            else:
                logger.warning("Azure OpenAI client failed connection test")
                return False
            
        except Exception as e:
            logger.error(f"Error initializing Azure OpenAI client: {str(e)}")
            return False
    
    def _initialize_openai_client(self) -> bool:
        """Initialize OpenAI client"""
        try:
            openai_config = self.llm_config['openai']
            
            api_key = openai_config.get('api_key') or os.getenv('OPENAI_API_KEY')
            
            if not api_key:
                logger.warning("OpenAI API key not provided")
                return False
            
            self.clients['openai'] = OpenAI(api_key=api_key)
            
            # Test the connection
            test_result = self._test_openai_connection()
            if test_result:
                logger.info("OpenAI client initialized and tested successfully")
                self.initialized = True
                return True
            else:
                logger.warning("OpenAI client failed connection test")
                return False
            
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            return False
    
    def _initialize_anthropic_client(self) -> bool:
        """Initialize Anthropic client"""
        try:
            anthropic_config = self.llm_config['anthropic']
            
            api_key = anthropic_config.get('api_key') or os.getenv('ANTHROPIC_API_KEY')
            
            if not api_key:
                logger.warning("Anthropic API key not provided")
                return False
            
            self.clients['anthropic'] = anthropic.Anthropic(api_key=api_key)
            
            # Test the connection
            test_result = self._test_anthropic_connection()
            if test_result:
                logger.info("Anthropic client initialized and tested successfully")
                self.initialized = True
                return True
            else:
                logger.warning("Anthropic client failed connection test")
                return False
            
        except Exception as e:
            logger.error(f"Error initializing Anthropic client: {str(e)}")
            return False
    
    def _initialize_mock_client(self):
        """Initialize mock client for development/testing"""
        self.clients['mock'] = MockLLMClient()
        self.provider = 'mock'
        self.initialized = True
        logger.info("Mock LLM client initialized")
    
    def _fallback_to_mock(self):
        """Fallback to mock client when real clients fail"""
        logger.warning(f"Falling back to mock client for provider: {self.provider}")
        self._initialize_mock_client()
    
    def _test_azure_connection(self) -> bool:
        """Test Azure OpenAI connection"""
        try:
            client = self.clients['azure']
            azure_config = self.llm_config['azure']
            
            response = client.chat.completions.create(
                model=azure_config['deployment_name'],
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5,
                temperature=0.1
            )
            return True
        except Exception as e:
            logger.error(f"Azure OpenAI connection test failed: {str(e)}")
            return False
    
    def _test_openai_connection(self) -> bool:
        """Test OpenAI connection"""
        try:
            client = self.clients['openai']
            openai_config = self.llm_config['openai']
            
            response = client.chat.completions.create(
                model=openai_config['model'],
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5,
                temperature=0.1
            )
            return True
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {str(e)}")
            return False
    
    def _test_anthropic_connection(self) -> bool:
        """Test Anthropic connection"""
        try:
            client = self.clients['anthropic']
            anthropic_config = self.llm_config['anthropic']
            
            response = client.messages.create(
                model=anthropic_config['model'],
                max_tokens=5,
                messages=[{"role": "user", "content": "Test"}]
            )
            return True
        except Exception as e:
            logger.error(f"Anthropic connection test failed: {str(e)}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7,
                system_prompt: str = None) -> str:
        """Generate text using the configured LLM provider"""
        try:
            if not self.initialized:
                logger.warning("LLM client not properly initialized, using fallback")
                return self._generate_fallback_response(prompt)
            
            start_time = time.time()
            
            if self.provider == 'azure_openai':
                response = self._generate_azure(prompt, max_tokens, temperature, system_prompt)
            elif self.provider == 'openai':
                response = self._generate_openai(prompt, max_tokens, temperature, system_prompt)
            elif self.provider == 'anthropic':
                response = self._generate_anthropic(prompt, max_tokens, temperature, system_prompt)
            else:
                response = self._generate_mock(prompt, max_tokens, temperature, system_prompt)
            
            end_time = time.time()
            
            logger.info(f"LLM generation completed in {end_time - start_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error generating with LLM: {str(e)}")
            return self._generate_fallback_response(prompt)
    
    def _generate_azure(self, prompt: str, max_tokens: int, temperature: float,
                       system_prompt: str = None) -> str:
        """Generate using Azure OpenAI"""
        try:
            client = self.clients['azure']
            azure_config = self.llm_config['azure']
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model=azure_config['deployment_name'],
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Azure OpenAI generation error: {str(e)}")
            raise
    
    def _generate_openai(self, prompt: str, max_tokens: int, temperature: float,
                        system_prompt: str = None) -> str:
        """Generate using OpenAI"""
        try:
            client = self.clients['openai']
            openai_config = self.llm_config['openai']
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model=openai_config['model'],
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI generation error: {str(e)}")
            raise
    
    def _generate_anthropic(self, prompt: str, max_tokens: int, temperature: float,
                           system_prompt: str = None) -> str:
        """Generate using Anthropic Claude"""
        try:
            client = self.clients['anthropic']
            anthropic_config = self.llm_config['anthropic']
            
            # Anthropic uses a different message format
            message_content = prompt
            if system_prompt:
                message_content = f"System: {system_prompt}\n\nUser: {prompt}"
            
            response = client.messages.create(
                model=anthropic_config['model'],
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": message_content}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic generation error: {str(e)}")
            raise
    
    def _generate_mock(self, prompt: str, max_tokens: int, temperature: float,
                      system_prompt: str = None) -> str:
        """Generate using mock client"""
        return self.clients['mock'].generate(prompt, max_tokens, temperature, system_prompt)
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate fallback response when LLM fails"""
        prompt_lower = prompt.lower()
        
        # Entity extraction fallback
        if 'extract entities' in prompt_lower or 'entity' in prompt_lower:
            return json.dumps([
                {"entity": "sales", "type": "table", "confidence": 0.8},
                {"entity": "customer", "type": "table", "confidence": 0.7},
                {"entity": "amount", "type": "column", "confidence": 0.9}
            ])
        
        # SQL generation fallback
        if 'select' in prompt_lower or 'sql' in prompt_lower:
            return json.dumps({
                "sql": "SELECT * FROM customers LIMIT 10;",
                "rationale": "Fallback query - LLM service unavailable",
                "confidence": 0.3,
                "tables_used": ["customers"],
                "assumptions": ["Used fallback response due to LLM unavailability"]
            })
        
        return "I apologize, but I'm unable to process your request at the moment due to technical issues with the LLM service. Please check your API configuration or try again later."
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection to the LLM provider"""
        try:
            if not self.initialized:
                return {
                    'status': 'error',
                    'provider': self.provider,
                    'message': 'LLM client not properly initialized'
                }
            
            test_prompt = "Hello, please respond with 'Connection successful'"
            response = self.generate(test_prompt, max_tokens=50, temperature=0.1)
            
            return {
                'status': 'success',
                'provider': self.provider,
                'response': response,
                'message': 'LLM connection test successful'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'provider': self.provider,
                'error': str(e),
                'message': 'LLM connection test failed'
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if self.provider == 'azure_openai':
            config = self.llm_config['azure']
            return {
                'provider': 'Azure OpenAI',
                'model': config['model_name'],
                'deployment': config['deployment_name'],
                'endpoint': config['endpoint'],
                'initialized': self.initialized
            }
        elif self.provider == 'openai':
            config = self.llm_config['openai']
            return {
                'provider': 'OpenAI',
                'model': config['model'],
                'initialized': self.initialized
            }
        elif self.provider == 'anthropic':
            config = self.llm_config['anthropic']
            return {
                'provider': 'Anthropic',
                'model': config['model'],
                'initialized': self.initialized
            }
        else:
            return {
                'provider': 'Mock',
                'model': 'mock-model',
                'initialized': self.initialized
            }


class MockLLMClient:
    """Mock LLM client for development and testing"""
    
    def __init__(self):
        self.response_templates = {
            'entity_extraction': [
                {"entity": "customers", "type": "table", "confidence": 0.9},
                {"entity": "orders", "type": "table", "confidence": 0.8},
                {"entity": "products", "type": "table", "confidence": 0.7},
                {"entity": "customer_id", "type": "column", "confidence": 0.9},
                {"entity": "order_date", "type": "column", "confidence": 0.8},
                {"entity": "total_amount", "type": "column", "confidence": 0.9}
            ],
            'sql_generation': {
                "sql": "SELECT c.customer_name, COUNT(o.order_id) as order_count, SUM(o.total_amount) as total_spent FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.customer_id, c.customer_name ORDER BY total_spent DESC LIMIT 10;",
                "rationale": "This query finds the top 10 customers by total spending, including their order count and total amount spent. It uses a LEFT JOIN to include customers even if they haven't placed orders.",
                "confidence": 0.85,
                "tables_used": ["customers", "orders"],
                "assumptions": ["Assuming customer_id is the join key", "Ordering by total spending in descending order"]
            }
        }
    
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7,
                system_prompt: str = None) -> str:
        """Generate mock response based on prompt content"""
        prompt_lower = prompt.lower()
        
        # Simulate processing time
        time.sleep(0.5)
        
        # Entity extraction
        if 'extract entities' in prompt_lower or ('entity' in prompt_lower and 'json' in prompt_lower):
            # Customize based on query content
            entities = []
            
            # Analyze the prompt to extract table/column mentions
            if 'customer' in prompt_lower:
                entities.extend([
                    {"entity": "customers", "type": "table", "confidence": 0.9},
                    {"entity": "customer_name", "type": "column", "confidence": 0.8},
                    {"entity": "customer_id", "type": "column", "confidence": 0.9}
                ])
            
            if 'order' in prompt_lower:
                entities.extend([
                    {"entity": "orders", "type": "table", "confidence": 0.8},
                    {"entity": "order_date", "type": "column", "confidence": 0.8},
                    {"entity": "order_id", "type": "column", "confidence": 0.9}
                ])
            
            if any(word in prompt_lower for word in ['sales', 'revenue', 'amount', 'total']):
                entities.extend([
                    {"entity": "sales", "type": "table", "confidence": 0.9},
                    {"entity": "total_amount", "type": "column", "confidence": 0.9},
                    {"entity": "revenue", "type": "term", "confidence": 0.7}
                ])
            
            if not entities:
                entities = self.response_templates['entity_extraction'][:3]
            
            return json.dumps(entities)
        
        # SQL generation
        elif 'sql' in prompt_lower and 'json' in prompt_lower:
            sql_response = self.response_templates['sql_generation'].copy()
            
            # Customize SQL based on query content
            if 'customer' in prompt_lower and 'total' in prompt_lower:
                sql_response['sql'] = "SELECT customer_name, SUM(total_amount) as total_spent FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY customer_name ORDER BY total_spent DESC;"
                sql_response['rationale'] = "Query to find customers and their total spending"
            
            elif 'count' in prompt_lower and 'order' in prompt_lower:
                sql_response['sql'] = "SELECT COUNT(*) as order_count FROM orders WHERE order_date >= '2024-01-01';"
                sql_response['rationale'] = "Count of orders for the specified period"
            
            elif 'product' in prompt_lower:
                sql_response['sql'] = "SELECT product_name, SUM(quantity_sold) as total_sold FROM products p JOIN order_items oi ON p.product_id = oi.product_id GROUP BY product_name ORDER BY total_sold DESC LIMIT 10;"
                sql_response['rationale'] = "Top 10 products by quantity sold"
            
            return json.dumps(sql_response)
        
        # SQL refinement
        elif 'refine' in prompt_lower or 'feedback' in prompt_lower:
            refined_response = {
                "sql": "SELECT c.customer_name, COUNT(o.order_id) as order_count, SUM(o.total_amount) as total_spent FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id WHERE o.order_date >= '2024-01-01' GROUP BY c.customer_id, c.customer_name ORDER BY total_spent DESC LIMIT 10;",
                "rationale": "Refined query based on feedback to include date filter and improved ordering",
                "confidence": 0.9,
                "changes": ["Added date filter for 2024", "Improved ordering logic", "Added explicit column selection"]
            }
            return json.dumps(refined_response)
        
        # General responses
        else:
            return f"Mock response for: {prompt[:100]}... (This is a simulated LLM response for development purposes)"
