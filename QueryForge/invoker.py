# automation_scripts.ipynb
# QueryForge Pro - Automation Scripts
# Use this notebook to automate platform operations

import requests
import pandas as pd
import json
import time
import os
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# ======================== CONFIGURATION ========================

# API Configuration
API_BASE_URL = "http://localhost:5000/api"  # Change this to your server URL
API_HEADERS = {
    "Content-Type": "application/json"
}

# Default settings
DEFAULT_PROJECT_ID = None  # Will be set after creating/selecting a project

# ======================== UTILITY FUNCTIONS ========================

def make_request(method: str, endpoint: str, data: Dict = None, params: Dict = None) -> Dict:
    """Make API request with error handling"""
    url = f"{API_BASE_URL}/{endpoint.lstrip('/')}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=API_HEADERS, params=params)
        elif method.upper() == "POST":
            response = requests.post(url, headers=API_HEADERS, json=data)
        elif method.upper() == "PUT":
            response = requests.put(url, headers=API_HEADERS, json=data)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=API_HEADERS)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"Error details: {error_detail}")
            except:
                print(f"Response text: {e.response.text}")
        raise

def print_json(data: Dict, title: str = "Response"):
    """Pretty print JSON data"""
    print(f"\n{title}:")
    print("=" * 50)
    print(json.dumps(data, indent=2, default=str))
    print("=" * 50)

def wait_for_job(job_id: str, endpoint: str, max_wait: int = 300) -> Dict:
    """Wait for async job to complete"""
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            status = make_request("GET", f"{endpoint}/{job_id}")
            
            if status.get('status') == 'completed':
                print(f"✅ Job {job_id} completed successfully!")
                return status
            elif status.get('status') == 'failed':
                print(f"❌ Job {job_id} failed: {status.get('message', 'Unknown error')}")
                return status
            else:
                progress = status.get('progress', 0) * 100
                message = status.get('message', 'Processing...')
                print(f"⏳ Job {job_id}: {progress:.1f}% - {message}")
                time.sleep(5)
                
        except Exception as e:
            print(f"Error checking job status: {e}")
            time.sleep(5)
    
    print(f"⏰ Job {job_id} timeout after {max_wait} seconds")
    return {"status": "timeout"}

# ======================== PROJECT MANAGEMENT ========================

def create_project(name: str, description: str = "", owner: str = "automation") -> Dict:
    """Create a new project"""
    data = {
        "name": name,
        "description": description,
        "owner": owner
    }
    
    result = make_request("POST", "projects", data)
    print(f"✅ Created project: {result['project']['name']} (ID: {result['project']['id']})")
    return result['project']

def list_projects() -> List[Dict]:
    """List all projects"""
    result = make_request("GET", "projects")
    projects = result.get('projects', [])
    
    print(f"\n📁 Found {len(projects)} projects:")
    for project in projects:
        print(f"  - {project['name']} (ID: {project['id']}) - {project['status']}")
        if project.get('description'):
            print(f"    Description: {project['description']}")
        print(f"    Sources: {project.get('sources_count', 0)}, "
              f"Dictionary: {project.get('dictionary_entries_count', 0)}")
    
    return projects

def get_project(project_id: int) -> Dict:
    """Get project details"""
    result = make_request("GET", f"projects/{project_id}")
    return result['project']

def delete_project(project_id: int) -> None:
    """Delete a project"""
    result = make_request("DELETE", f"projects/{project_id}")
    print(f"✅ Project {project_id} deleted")

def clone_project(project_id: int, new_name: str) -> Dict:
    """Clone a project"""
    data = {"name": new_name}
    result = make_request("POST", f"projects/{project_id}/clone", data)
    print(f"✅ Project cloned as: {result['project']['name']} (ID: {result['project']['id']})")
    return result['project']

# ======================== DATA SOURCE MANAGEMENT ========================

def upload_file(project_id: int, file_path: str) -> Dict:
    """Upload a file data source"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # For file upload, we need to use a different approach
    url = f"{API_BASE_URL}/projects/{project_id}/sources/upload"
    
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
        response.raise_for_status()
        result = response.json()
    
    print(f"✅ Uploaded file: {os.path.basename(file_path)}")
    print(f"   Tables created: {result.get('tables_created', 0)}")
    return result

def add_database_source(project_id: int, name: str, db_config: Dict) -> Dict:
    """Add a database connection as data source"""
    data = {
        "name": name,
        "db_type": db_config["type"],
        "connection_config": db_config
    }
    
    result = make_request("POST", f"projects/{project_id}/sources/database", data)
    print(f"✅ Added database source: {name}")
    print(f"   Tables imported: {result.get('tables_imported', 0)}")
    return result

def list_data_sources(project_id: int) -> List[Dict]:
    """List data sources for a project"""
    result = make_request("GET", f"projects/{project_id}/sources")
    sources = result.get('sources', [])
    
    print(f"\n💾 Found {len(sources)} data sources:")
    for source in sources:
        print(f"  - {source['name']} ({source['type']}/{source.get('subtype', 'N/A')})")
        print(f"    Status: {source['ingest_status']}")
        if source.get('file_size'):
            print(f"    Size: {source['file_size'] / 1024 / 1024:.1f} MB")
        print(f"    Tables: {source.get('tables_count', 0)}")
    
    return sources

def get_source_tables(source_id: int) -> List[Dict]:
    """Get tables for a data source"""
    result = make_request("GET", f"sources/{source_id}/tables")
    tables = result.get('tables', [])
    
    print(f"\n📊 Found {len(tables)} tables:")
    for table in tables:
        print(f"  - {table['name']} ({table['row_count']} rows, {table['column_count']} columns)")
        if table.get('description'):
            print(f"    Description: {table['description']}")
    
    return tables

def get_table_columns(table_id: int) -> List[Dict]:
    """Get columns for a table"""
    result = make_request("GET", f"tables/{table_id}/columns")
    columns = result.get('columns', [])
    
    print(f"\n📋 Found {len(columns)} columns:")
    for col in columns:
        print(f"  - {col['name']} ({col['data_type']})")
        if col.get('description'):
            print(f"    Description: {col['description']}")
        if col.get('business_category'):
            print(f"    Category: {col['business_category']}")
    
    return columns

# ======================== DATA DICTIONARY MANAGEMENT ========================

def create_dictionary_entry(project_id: int, term: str, definition: str, 
                          category: str = "business_term", **kwargs) -> Dict:
    """Create a dictionary entry"""
    data = {
        "term": term,
        "definition": definition,
        "category": category,
        **kwargs
    }
    
    result = make_request("POST", f"projects/{project_id}/dictionary", data)
    print(f"✅ Created dictionary entry: {term}")
    return result['entry']

def list_dictionary_entries(project_id: int) -> List[Dict]:
    """List dictionary entries for a project"""
    result = make_request("GET", f"projects/{project_id}/dictionary")
    entries = result.get('entries', [])
    
    print(f"\n📚 Found {len(entries)} dictionary entries:")
    for entry in entries:
        print(f"  - {entry['term']} ({entry['category']})")
        print(f"    Definition: {entry['definition'][:100]}...")
        if entry.get('synonyms'):
            print(f"    Synonyms: {', '.join(entry['synonyms'])}")
    
    return entries

def generate_dictionary_suggestions(project_id: int) -> Dict:
    """Generate automatic dictionary suggestions"""
    result = make_request("POST", f"projects/{project_id}/dictionary/suggest")
    suggestions = result.get('suggestions', {})
    
    print(f"✅ Generated {result.get('auto_generated_count', 0)} suggestions")
    
    for category, terms in suggestions.items():
        if isinstance(terms, list) and terms:
            print(f"\n{category.replace('_', ' ').title()}: {len(terms)} terms")
            for term in terms[:5]:  # Show first 5
                print(f"  - {term.get('term', 'N/A')}: {term.get('auto_definition', 'N/A')[:80]}...")
    
    return suggestions

def bulk_create_dictionary_entries(project_id: int, entries: List[Dict]) -> List[int]:
    """Create multiple dictionary entries from suggestions"""
    created_ids = []
    
    for entry_data in entries:
        try:
            entry = create_dictionary_entry(project_id, **entry_data)
            created_ids.append(entry['id'])
        except Exception as e:
            print(f"❌ Failed to create entry '{entry_data.get('term', 'Unknown')}': {e}")
    
    print(f"✅ Created {len(created_ids)} dictionary entries")
    return created_ids

# ======================== EMBEDDINGS & INDEXING ========================

def create_embeddings_batch(project_id: int, model_name: str, 
                          object_types: List[str]) -> str:
    """Create embeddings in batch"""
    data = {
        "model_name": model_name,
        "object_types": object_types
    }
    
    result = make_request("POST", f"projects/{project_id}/embeddings/batch", data)
    job_id = result.get('job_id')
    
    print(f"✅ Started embedding creation job: {job_id}")
    print(f"   Model: {model_name}")
    print(f"   Object types: {', '.join(object_types)}")
    
    return job_id

def list_embeddings(project_id: int) -> List[Dict]:
    """List embeddings for a project"""
    result = make_request("GET", f"projects/{project_id}/embeddings")
    embeddings = result.get('embeddings', [])
    
    print(f"\n🔢 Found {len(embeddings)} embeddings:")
    
    # Group by model
    by_model = {}
    for emb in embeddings:
        model = emb['model_name']
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(emb)
    
    for model, embs in by_model.items():
        print(f"  {model}: {len(embs)} embeddings")
        by_type = {}
        for emb in embs:
            obj_type = emb['object_type']
            by_type[obj_type] = by_type.get(obj_type, 0) + 1
        for obj_type, count in by_type.items():
            print(f"    - {obj_type}: {count}")
    
    return embeddings

def create_index(project_id: int, name: str, index_type: str, 
                object_scope: Dict, **kwargs) -> Dict:
    """Create a search index"""
    data = {
        "name": name,
        "index_type": index_type,
        "object_scope": object_scope,
        **kwargs
    }
    
    result = make_request("POST", f"projects/{project_id}/indexes", data)
    print(f"✅ Created index: {name} (ID: {result['index']['id']})")
    print(f"   Type: {index_type}")
    print(f"   Status: {result['index']['status']}")
    
    return result['index']

def list_indexes(project_id: int) -> List[Dict]:
    """List indexes for a project"""
    result = make_request("GET", f"projects/{project_id}/indexes")
    indexes = result.get('indexes', [])
    
    print(f"\n🔍 Found {len(indexes)} indexes:")
    for idx in indexes:
        print(f"  - {idx['name']} ({idx['index_type']})")
        print(f"    Status: {idx['status']}")
        print(f"    Vectors: {idx.get('total_vectors', 0)}")
        if idx.get('index_size_mb'):
            print(f"    Size: {idx['index_size_mb']} MB")
    
    return indexes

# ======================== SEARCH OPERATIONS ========================

def search(query: str, project_id: int = None, index_id: int = None, 
          search_type: str = "hybrid", top_k: int = 10) -> Dict:
    """Perform search"""
    params = {
        "query": query,
        "search_type": search_type,
        "top_k": top_k
    }
    
    if index_id:
        params["index_id"] = index_id
    if project_id:
        params["project_id"] = project_id
    
    result = make_request("POST", "search", params)
    
    print(f"🔍 Search results for: '{query}'")
    print(f"   Found {result['total_results']} results in {result['search_time_ms']}ms")
    
    for i, res in enumerate(result['results'][:5], 1):  # Show top 5
        print(f"\n{i}. {res.get('object_type', 'Unknown')} - Score: {res.get('score', 0):.3f}")
        if res.get('table_name'):
            print(f"   Table: {res['table_name']}")
        if res.get('column_name'):
            print(f"   Column: {res['column_name']}")
        if res.get('term'):
            print(f"   Term: {res['term']}")
        print(f"   Text: {res.get('object_text', 'N/A')[:100]}...")
    
    return result

# ======================== CHAT (NL → SQL) OPERATIONS ========================

def extract_entities(query: str, project_id: int) -> Dict:
    """Extract entities from natural language query"""
    data = {
        "query": query,
        "project_id": project_id
    }
    
    result = make_request("POST", "chat/entities", data)
    
    print(f"🧠 Extracted entities from: '{query}'")
    print(f"   Processing time: {result.get('extraction_time_ms', 0)}ms")
    
    entities = result.get('entities', [])
    for entity in entities:
        print(f"   - {entity['entity']} ({entity['type']}) - Confidence: {entity['confidence']:.3f}")
    
    return result

def map_entities(entities: List[Dict], project_id: int) -> Dict:
    """Map entities to schema objects"""
    data = {
        "entities": entities,
        "project_id": project_id
    }
    
    result = make_request("POST", "chat/mapping", data)
    
    print(f"🗺️  Entity mapping completed in {result.get('mapping_time_ms', 0)}ms")
    print(f"   Mapped {result.get('mapped_entities', 0)}/{result.get('total_entities', 0)} entities")
    
    for mapping in result.get('mappings', []):
        print(f"\n   Entity: {mapping['entity']} ({mapping['entity_type']})")
        best_match = mapping.get('best_match')
        if best_match:
            print(f"   → {best_match['type']}: {best_match['name']} (confidence: {best_match['confidence']:.3f})")
    
    return result

def generate_sql(query: str, entities: List[Dict], mappings: List[Dict], project_id: int) -> Dict:
    """Generate SQL from natural language"""
    data = {
        "query": query,
        "entities": entities,
        "mappings": mappings,
        "project_id": project_id
    }
    
    result = make_request("POST", "chat/sql", data)
    
    print(f"⚡ SQL generation completed in {result.get('generation_time_ms', 0)}ms")
    print(f"   Confidence: {result.get('confidence', 0):.3f}")
    print(f"   Tables used: {', '.join(result.get('tables_used', []))}")
    
    sql = result.get('sql', '')
    print(f"\n📝 Generated SQL:")
    print("-" * 50)
    print(sql)
    print("-" * 50)
    
    if result.get('rationale'):
        print(f"\n💭 Rationale: {result['rationale']}")
    
    return result

def execute_sql(sql_query: str, project_id: int) -> Dict:
    """Execute SQL query"""
    data = {
        "sql_query": sql_query,
        "project_id": project_id
    }
    
    result = make_request("POST", "chat/execute", data)
    
    if result.get('success'):
        print(f"✅ SQL executed successfully in {result.get('execution_time_ms', 0)}ms")
        print(f"   Returned {result.get('row_count', 0)} rows")
        
        # Show sample data
        data_rows = result.get('data', [])
        if data_rows:
            print(f"\n📊 Sample results:")
            df = pd.DataFrame(data_rows[:10])  # Show first 10 rows
            print(df.to_string(index=False, max_cols=5))
            
            if result.get('truncated'):
                print(f"\n   ... (showing first 10 of {result.get('row_count', 0)} rows)")
    else:
        print(f"❌ SQL execution failed: {result.get('error', 'Unknown error')}")
    
    return result

def end_to_end_nlq(query: str, project_id: int, auto_execute: bool = False) -> Dict:
    """Run complete NL → SQL pipeline"""
    print(f"\n🚀 Starting end-to-end NLQ pipeline for: '{query}'")
    print("=" * 80)
    
    # Step 1: Extract entities
    print("\n1️⃣  Extracting entities...")
    entity_result = extract_entities(query, project_id)
    entities = entity_result.get('entities', [])
    
    if not entities:
        print("❌ No entities extracted. Cannot proceed.")
        return {"error": "No entities extracted"}
    
    # Step 2: Map entities
    print("\n2️⃣  Mapping entities to schema...")
    mapping_result = map_entities(entities, project_id)
    mappings = mapping_result.get('mappings', [])
    
    if not mappings:
        print("❌ No entity mappings found. Cannot proceed.")
        return {"error": "No entity mappings found"}
    
    # Step 3: Generate SQL
    print("\n3️⃣  Generating SQL...")
    sql_result = generate_sql(query, entities, mappings, project_id)
    sql_query = sql_result.get('sql', '')
    
    if not sql_query:
        print("❌ No SQL generated. Cannot proceed.")
        return {"error": "No SQL generated"}
    
    # Step 4: Execute SQL (if requested)
    execution_result = None
    if auto_execute:
        print("\n4️⃣  Executing SQL...")
        execution_result = execute_sql(sql_query, project_id)
    else:
        print("\n4️⃣  SQL ready for execution (auto_execute=False)")
        print("   Use execute_sql() to run the query")
    
    return {
        "query": query,
        "entities": entity_result,
        "mappings": mapping_result,
        "sql": sql_result,
        "execution": execution_result
    }

# ======================== SYSTEM ADMINISTRATION ========================

def get_system_health() -> Dict:
    """Get system health status"""
    result = make_request("GET", "admin/health")
    health = result.get('health', {})
    
    print(f"💊 System Health Status: {health.get('status', 'unknown').upper()}")
    print(f"   Timestamp: {health.get('timestamp', 'N/A')}")
    
    # Database health
    db_health = health.get('database', {})
    print(f"\n💾 Database: {db_health.get('status', 'unknown').upper()}")
    print(f"   Engine: {db_health.get('engine', 'N/A')}")
    print(f"   Tables: {db_health.get('tables_count', 0)}")
    
    records = db_health.get('records', {})
    for key, count in records.items():
        print(f"   {key.title()}: {count}")
    
    # System metrics
    system = health.get('system', {})
    if system:
        print(f"\n🖥️  System Metrics:")
        cpu = system.get('cpu', {})
        memory = system.get('memory', {})
        disk = system.get('disk', {})
        
        print(f"   CPU: {cpu.get('usage_percent', 0):.1f}% ({cpu.get('count', 0)} cores)")
        print(f"   Memory: {memory.get('usage_percent', 0):.1f}% ({memory.get('available_gb', 0):.1f}GB available)")
        print(f"   Disk: {disk.get('usage_percent', 0):.1f}% ({disk.get('free_gb', 0):.1f}GB free)")
    
    return health

def browse_tables(page: int = 1, per_page: int = 20) -> Dict:
    """Browse all tables in the system"""
    params = {"page": page, "per_page": per_page}
    result = make_request("GET", "admin/tables", params=params)
    
    tables = result.get('tables', [])
    pagination = result.get('pagination', {})
    
    print(f"\n📊 Tables (Page {page} of {pagination.get('pages', 1)}):")
    print(f"   Total: {pagination.get('total', 0)} tables")
    
    for table in tables:
        print(f"\n   - {table['name']} (ID: {table['id']})")
        print(f"     Project: {table.get('project_name', 'N/A')}")
        print(f"     Source: {table.get('source_name', 'N/A')} ({table.get('source_type', 'N/A')})")
        print(f"     Rows: {table.get('row_count', 0)}, Columns: {table.get('column_count', 0)}")
    
    return result

def execute_admin_sql(sql: str) -> Dict:
    """Execute SQL with admin privileges"""
    data = {"sql": sql}
    result = make_request("POST", "admin/execute", data)
    
    if result.get('success'):
        print(f"✅ Admin SQL executed successfully")
        print(f"   Execution time: {result.get('execution_time_seconds', 0):.3f}s")
        print(f"   Rows returned: {result.get('row_count', 0)}")
        
        if result.get('data'):
            df = pd.DataFrame(result['data'])
            print(f"\n📊 Results:")
            print(df.to_string(index=False))
    else:
        print(f"❌ Admin SQL failed: {result.get('error', 'Unknown error')}")
    
    return result

# ======================== WORKFLOW AUTOMATION ========================

class QueryForgeAutomation:
    """Main automation class for complex workflows"""
    
    def __init__(self, api_url: str = API_BASE_URL):
        self.api_url = api_url
        self.current_project_id = None
        self.current_project = None
    
    def setup_project(self, project_name: str, description: str = "") -> Dict:
        """Create or select a project for automation"""
        # Check if project exists
        projects = list_projects()
        existing = next((p for p in projects if p['name'] == project_name), None)
        
        if existing:
            print(f"📁 Using existing project: {project_name}")
            self.current_project = existing
            self.current_project_id = existing['id']
        else:
            print(f"📁 Creating new project: {project_name}")
            self.current_project = create_project(project_name, description)
            self.current_project_id = self.current_project['id']
        
        return self.current_project
    
    def ingest_data(self, file_paths: List[str] = None, db_configs: List[Dict] = None) -> Dict:
        """Ingest multiple data sources"""
        if not self.current_project_id:
            raise ValueError("No project selected. Call setup_project() first.")
        
        results = {"files": [], "databases": []}
        
        # Upload files
        if file_paths:
            for file_path in file_paths:
                try:
                    result = upload_file(self.current_project_id, file_path)
                    results["files"].append(result)
                except Exception as e:
                    print(f"❌ Failed to upload {file_path}: {e}")
        
        # Add databases
        if db_configs:
            for config in db_configs:
                try:
                    name = config.pop('name', f"DB_{len(results['databases'])}")
                    result = add_database_source(self.current_project_id, name, config)
                    results["databases"].append(result)
                except Exception as e:
                    print(f"❌ Failed to add database {name}: {e}")
        
        return results
    
    def build_knowledge_base(self, generate_suggestions: bool = True, 
                          embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> Dict:
        """Build complete knowledge base with dictionary and embeddings"""
        if not self.current_project_id:
            raise ValueError("No project selected. Call setup_project() first.")
        
        results = {}
        
        # Generate dictionary suggestions
        if generate_suggestions:
            print("\n📚 Generating dictionary suggestions...")
            suggestions = generate_dictionary_suggestions(self.current_project_id)
            results['dictionary_suggestions'] = suggestions
            
            # Auto-create high-confidence terms
            auto_create = []
            for category, terms in suggestions.items():
                if isinstance(terms, list):
                    for term in terms:
                        if term.get('confidence', 0) > 0.8:  # High confidence only
                            auto_create.append({
                                'term': term['term'],
                                'definition': term.get('enhanced_definition') or term.get('auto_definition'),
                                'category': term.get('category', 'business_term'),
                                'domain': term.get('suggested_domain')
                            })
            
            if auto_create:
                print(f"\n📝 Auto-creating {len(auto_create)} high-confidence dictionary entries...")
                created_ids = bulk_create_dictionary_entries(self.current_project_id, auto_create)
                results['auto_created_entries'] = created_ids
        
        # Create embeddings
        print(f"\n🔢 Creating embeddings with {embedding_model}...")
        job_id = create_embeddings_batch(
            self.current_project_id, 
            embedding_model, 
            ['tables', 'columns', 'dictionary']
        )
        
        # Wait for embedding job to complete
        print("⏳ Waiting for embedding creation to complete...")
        embedding_result = wait_for_job(job_id, "embeddings/job")
        results['embeddings'] = embedding_result
        
        if embedding_result.get('status') == 'completed':
            # Create indexes
            print("\n🔍 Creating search indexes...")
            
            # FAISS index for semantic search
            faiss_index = create_index(
                self.current_project_id,
                "Main FAISS Index",
                "faiss",
                {"object_types": ["tables", "columns", "dictionary"]},
                metric="cosine",
                embedding_model=embedding_model
            )
            results['faiss_index'] = faiss_index
            
            # TF-IDF index for lexical search
            tfidf_index = create_index(
                self.current_project_id,
                "TF-IDF Index",
                "tfidf",
                {"object_types": ["tables", "columns", "dictionary"]},
                build_params={"max_features": 10000, "ngram_range": [1, 2]}
            )
            results['tfidf_index'] = tfidf_index
        
        return results
    
    def test_nlq_pipeline(self, test_queries: List[str], auto_execute: bool = False) -> List[Dict]:
        """Test the NLQ pipeline with multiple queries"""
        if not self.current_project_id:
            raise ValueError("No project selected. Call setup_project() first.")
        
        results = []
        
        for query in test_queries:
            print(f"\n{'='*80}")
            print(f"Testing query: {query}")
            print(f"{'='*80}")
            
            try:
                result = end_to_end_nlq(query, self.current_project_id, auto_execute)
                results.append(result)
                
                # Brief summary
                if result.get('sql'):
                    confidence = result['sql'].get('confidence', 0)
                    print(f"\n✅ Pipeline completed - SQL confidence: {confidence:.3f}")
                else:
                    print(f"\n❌ Pipeline failed")
                    
            except Exception as e:
                print(f"\n❌ Pipeline error: {e}")
                results.append({"query": query, "error": str(e)})
        
        return results

# ======================== EXAMPLE USAGE ========================

def demo_workflow():
    """Demonstrate a complete automation workflow"""
    print("🚀 QueryForge Pro - Automation Demo")
    print("=" * 80)
    
    # Initialize automation
    automation = QueryForgeAutomation()
    
    # Step 1: Setup project
    project = automation.setup_project(
        "Demo Project", 
        "Automated demo project created via Jupyter notebook"
    )
    
    # Step 2: Example data ingestion (you would replace with real file paths)
    # automation.ingest_data(
    #     file_paths=["data/sales.csv", "data/customers.xlsx"],
    #     db_configs=[{
    #         "name": "Production DB",
    #         "type": "postgresql",
    #         "host": "localhost",
    #         "port": 5432,
    #         "database": "mydb",
    #         "username": "user",
    #         "password": "password"
    #     }]
    # )
    
    # Step 3: Build knowledge base
    # kb_result = automation.build_knowledge_base()
    
    # Step 4: Test NLQ pipeline
    test_queries = [
        "Show me all customers",
        "What are the top selling products this month?",
        "List revenue by region",
        "Find customers with high order values"
    ]
    
    # nlq_results = automation.test_nlq_pipeline(test_queries, auto_execute=False)
    
    print("\n✅ Demo workflow completed!")
    print("   Uncomment the actual workflow steps to run with real data")

# Run demo
if __name__ == "__main__":
    # Example: Get system health
    print("Checking system health...")
    get_system_health()
    
    # Example: List existing projects
    list_projects()
    
    # Uncomment to run full demo
    # demo_workflow()

print("\n📝 Notebook loaded successfully!")
print("Available functions:")
print("  Projects: create_project, list_projects, get_project, delete_project, clone_project")
print("  Data Sources: upload_file, add_database_source, list_data_sources")
print("  Dictionary: create_dictionary_entry, generate_dictionary_suggestions")
print("  Embeddings: create_embeddings_batch, list_embeddings, create_index")
print("  Search: search")
print("  Chat/NLQ: extract_entities, map_entities, generate_sql, execute_sql, end_to_end_nlq")
print("  Admin: get_system_health, browse_tables, execute_admin_sql")
print("  Workflow: QueryForgeAutomation class")
print("\nTo get started:")
print("  1. Update API_BASE_URL if your server is not on localhost:5000")
print("  2. Run demo_workflow() or use individual functions")
print("  3. Use QueryForgeAutomation class for complex workflows")