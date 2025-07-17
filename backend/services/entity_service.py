# backend/services/entity_service.py
"""
Entity mapping and resolution service
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from difflib import SequenceMatcher
import re

from backend.models import (
    Table, Column, DictionaryEntry, Source,
    EntityType, MappingConfidence
)
from backend.services.embedding_service import embedding_service
from config import Config

logger = logging.getLogger(__name__)

class EntityService:
    """Service for entity mapping and resolution"""
    
    def __init__(self):
        self.similarity_threshold = Config.ENTITY_CONFIG['similarity_threshold']
        self.max_entities = Config.ENTITY_CONFIG['max_entities']
        self.fuzzy_threshold = Config.ENTITY_CONFIG['fuzzy_threshold']
        self.ranking_weights = Config.ENTITY_CONFIG['ranking_weights']
    
    async def map_entities_to_schema(
        self,
        entities: List[Dict[str, Any]],
        project_id: int,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Map extracted entities to database schema objects"""
        
        try:
            # Get all available objects for mapping
            schema_objects = await self._get_schema_objects(project_id, db)
            
            mappings = []
            unmapped_entities = []
            confidence_scores = {}
            
            for entity in entities:
                entity_text = entity.get('entity', '').strip()
                entity_type = entity.get('type', 'unknown')
                entity_confidence = entity.get('confidence', 0.0)
                
                # Find potential mappings
                candidates = await self._find_mapping_candidates(
                    entity_text, entity_type, schema_objects, db
                )
                
                if candidates:
                    # Rank candidates
                    ranked_candidates = self._rank_candidates(
                        entity_text, candidates, entity_confidence
                    )
                    
                    # Take best match if above threshold
                    best_match = ranked_candidates[0]
                    if best_match['final_score'] >= self.similarity_threshold:
                        mapping = {
                            'entity': entity_text,
                            'entity_type': entity_type,
                            'target_type': best_match['object_type'],
                            'target_id': best_match['object_id'],
                            'target_name': best_match['object_name'],
                            'similarity_score': best_match['final_score'],
                            'confidence_level': self._get_confidence_level(best_match['final_score']),
                            'mapping_method': best_match['method'],
                            'alternatives': ranked_candidates[1:3]  # Top 2 alternatives
                        }
                        mappings.append(mapping)
                        confidence_scores[entity_text] = best_match['final_score']
                    else:
                        unmapped_entities.append(entity)
                else:
                    unmapped_entities.append(entity)
            
            return {
                'mappings': mappings,
                'unmapped_entities': unmapped_entities,
                'confidence_scores': confidence_scores,
                'total_entities': len(entities),
                'mapped_entities': len(mappings),
                'processing_time': 0.0  # TODO: Add timing
            }
            
        except Exception as e:
            logger.error(f"Entity mapping failed: {e}")
            return {
                'mappings': [],
                'unmapped_entities': entities,
                'error': str(e)
            }
    
    async def _get_schema_objects(
        self,
        project_id: int,
        db: AsyncSession
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get all schema objects for mapping"""
        
        schema_objects = {
            'tables': [],
            'columns': [],
            'dictionary_entries': []
        }
        
        # Get tables
        tables_query = select(Table).join(Source).where(Source.project_id == project_id)
        tables_result = await db.execute(tables_query)
        tables = tables_result.scalars().all()
        
        for table in tables:
            schema_objects['tables'].append({
                'object_id': table.id,
                'object_type': 'table',
                'object_name': table.name,
                'display_name': table.display_name,
                'description': table.description,
                'schema_name': table.schema_name,
                'search_terms': self._generate_search_terms(
                    table.name, table.display_name, table.description
                )
            })
        
        # Get columns
        columns_query = select(Column).join(Table).join(Source).where(
            Source.project_id == project_id
        )
        columns_result = await db.execute(columns_query)
        columns = columns_result.scalars().all()
        
        for column in columns:
            schema_objects['columns'].append({
                'object_id': column.id,
                'object_type': 'column',
                'object_name': column.name,
                'display_name': column.display_name,
                'description': column.description,
                'data_type': column.data_type,
                'table_id': column.table_id,
                'search_terms': self._generate_search_terms(
                    column.name, column.display_name, column.description
                )
            })
        
        # Get dictionary entries
        dict_query = select(DictionaryEntry).where(
            DictionaryEntry.project_id == project_id,
            DictionaryEntry.is_deleted == False
        )
        dict_result = await db.execute(dict_query)
        dict_entries = dict_result.scalars().all()
        
        for entry in dict_entries:
            search_terms = self._generate_search_terms(
                entry.term, None, entry.definition
            )
            # Add synonyms and abbreviations
            if entry.synonyms:
                search_terms.extend(entry.synonyms)
            if entry.abbreviations:
                search_terms.extend(entry.abbreviations)
            
            schema_objects['dictionary_entries'].append({
                'object_id': entry.id,
                'object_type': 'dictionary_entry',
                'object_name': entry.term,
                'definition': entry.definition,
                'category': entry.category.value,
                'synonyms': entry.synonyms or [],
                'abbreviations': entry.abbreviations or [],
                'search_terms': search_terms
            })
        
        return schema_objects
    
    def _generate_search_terms(
        self,
        name: str,
        display_name: Optional[str] = None,
        description: Optional[str] = None
    ) -> List[str]:
        """Generate search terms for an object"""
        
        terms = []
        
        if name:
            terms.append(name.lower())
            # Add variations (with underscores, spaces, etc.)
            terms.extend(self._generate_name_variations(name))
        
        if display_name and display_name != name:
            terms.append(display_name.lower())
            terms.extend(self._generate_name_variations(display_name))
        
        if description:
            # Extract key terms from description
            desc_terms = self._extract_key_terms(description)
            terms.extend(desc_terms)
        
        # Remove duplicates and empty terms
        return list(set([term for term in terms if term.strip()]))
    
    def _generate_name_variations(self, name: str) -> List[str]:
        """Generate variations of a name"""
        
        variations = []
        name_lower = name.lower()
        
        # Replace underscores with spaces
        if '_' in name_lower:
            variations.append(name_lower.replace('_', ' '))
        
        # Replace spaces with underscores
        if ' ' in name_lower:
            variations.append(name_lower.replace(' ', '_'))
        
        # Remove underscores and spaces
        clean_name = re.sub(r'[_\s]+', '', name_lower)
        if clean_name != name_lower:
            variations.append(clean_name)
        
        # Split camelCase
        camel_split = re.sub(r'([a-z])([A-Z])', r'\1 \2', name).lower()
        if camel_split != name_lower:
            variations.append(camel_split)
        
        return variations
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from description text"""
        
        if not text:
            return []
        
        # Simple keyword extraction
        text_lower = text.lower()
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
            'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text_lower)
        key_terms = [word for word in words if word not in stop_words]
        
        return key_terms[:10]  # Limit to top 10 terms
    
    async def _find_mapping_candidates(
        self,
        entity_text: str,
        entity_type: str,
        schema_objects: Dict[str, List[Dict[str, Any]]],
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Find potential mapping candidates for an entity"""
        
        candidates = []
        entity_lower = entity_text.lower()
        
        # Determine which object types to search based on entity type
        search_targets = self._get_search_targets(entity_type)
        
        for target_type in search_targets:
            if target_type not in schema_objects:
                continue
            
            for obj in schema_objects[target_type]:
                # Calculate similarity scores
                scores = self._calculate_similarity_scores(entity_lower, obj)
                
                if scores['max_score'] > 0.3:  # Minimum threshold
                    candidate = {
                        'object_id': obj['object_id'],
                        'object_type': obj['object_type'],
                        'object_name': obj['object_name'],
                        'display_name': obj.get('display_name'),
                        'description': obj.get('description'),
                        'similarity_scores': scores,
                        'max_score': scores['max_score'],
                        'method': scores['best_method']
                    }
                    candidates.append(candidate)
        
        # Sort by similarity score
        candidates.sort(key=lambda x: x['max_score'], reverse=True)
        
        return candidates[:20]  # Limit candidates
    
    def _get_search_targets(self, entity_type: str) -> List[str]:
        """Get target object types based on entity type"""
        
        entity_type_lower = entity_type.lower()
        
        if entity_type_lower in ['table', 'entity']:
            return ['tables', 'dictionary_entries']
        elif entity_type_lower in ['column', 'field', 'attribute']:
            return ['columns', 'dictionary_entries']
        elif entity_type_lower in ['value', 'data']:
            return ['columns', 'dictionary_entries']
        elif entity_type_lower in ['business_term', 'keyword']:
            return ['dictionary_entries', 'tables', 'columns']
        else:
            # Search all types for unknown entity types
            return ['tables', 'columns', 'dictionary_entries']
    
    def _calculate_similarity_scores(
        self,
        entity_text: str,
        obj: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate various similarity scores"""
        
        scores = {
            'exact_match': 0.0,
            'fuzzy_match': 0.0,
            'partial_match': 0.0,
            'description_match': 0.0,
            'max_score': 0.0,
            'best_method': 'none'
        }
        
        search_terms = obj.get('search_terms', [])
        object_name = obj.get('object_name', '').lower()
        description = obj.get('description', '').lower() if obj.get('description') else ''
        
        # Exact match
        if entity_text == object_name:
            scores['exact_match'] = 1.0
        elif entity_text in search_terms:
            scores['exact_match'] = 1.0
        
        # Fuzzy match with object name
        if object_name:
            fuzzy_score = SequenceMatcher(None, entity_text, object_name).ratio()
            scores['fuzzy_match'] = max(scores['fuzzy_match'], fuzzy_score)
        
        # Fuzzy match with search terms
        for term in search_terms:
            fuzzy_score = SequenceMatcher(None, entity_text, term).ratio()
            scores['fuzzy_match'] = max(scores['fuzzy_match'], fuzzy_score)
        
        # Partial match (substring)
        if entity_text in object_name or object_name in entity_text:
            scores['partial_match'] = 0.8
        
        for term in search_terms:
            if entity_text in term or term in entity_text:
                scores['partial_match'] = max(scores['partial_match'], 0.7)
        
        # Description match
        if description and entity_text in description:
            scores['description_match'] = 0.6
        
        # Determine best score and method
        if scores['exact_match'] > 0:
            scores['max_score'] = scores['exact_match']
            scores['best_method'] = 'exact'
        elif scores['fuzzy_match'] > scores['partial_match']:
            scores['max_score'] = scores['fuzzy_match']
            scores['best_method'] = 'fuzzy'
        elif scores['partial_match'] > scores['description_match']:
            scores['max_score'] = scores['partial_match']
            scores['best_method'] = 'partial'
        elif scores['description_match'] > 0:
            scores['max_score'] = scores['description_match']
            scores['best_method'] = 'description'
        
        return scores
    
    def _rank_candidates(
        self,
        entity_text: str,
        candidates: List[Dict[str, Any]],
        entity_confidence: float
    ) -> List[Dict[str, Any]]:
        """Rank mapping candidates using weighted scoring"""
        
        for candidate in candidates:
            scores = candidate['similarity_scores']
            
            # Calculate weighted final score
            final_score = (
                scores['exact_match'] * self.ranking_weights['exact_match'] +
                scores['fuzzy_match'] * self.ranking_weights['fuzzy_match'] +
                scores['partial_match'] * 0.5 +  # Lower weight for partial matches
                scores['description_match'] * 0.3  # Lower weight for description matches
            )
            
            # Boost score based on entity confidence
            final_score *= entity_confidence
            
            # Object type importance boost
            if candidate['object_type'] == 'table':
                final_score *= self.ranking_weights['table_importance']
            
            candidate['final_score'] = min(final_score, 1.0)  # Cap at 1.0
        
        # Sort by final score
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        return candidates
    
    def _get_confidence_level(self, score: float) -> MappingConfidence:
        """Get confidence level based on score"""
        
        if score >= 0.9:
            return MappingConfidence.HIGH
        elif score >= 0.7:
            return MappingConfidence.MEDIUM
        elif score >= 0.5:
            return MappingConfidence.LOW
        else:
            return MappingConfidence.UNCERTAIN
    
    async def validate_mapping(
        self,
        entity_text: str,
        target_type: str,
        target_id: int,
        db: AsyncSession
    ) -> bool:
        """Validate if a mapping makes sense"""
        
        try:
            # Basic validation - check if target object exists
            if target_type == 'table':
                query = select(Table).where(Table.id == target_id)
                result = await db.execute(query)
                return result.scalar_one_or_none() is not None
            
            elif target_type == 'column':
                query = select(Column).where(Column.id == target_id)
                result = await db.execute(query)
                return result.scalar_one_or_none() is not None
            
            elif target_type == 'dictionary_entry':
                query = select(DictionaryEntry).where(DictionaryEntry.id == target_id)
                result = await db.execute(query)
                return result.scalar_one_or_none() is not None
            
            return False
            
        except Exception as e:
            logger.error(f"Mapping validation failed: {e}")
            return False
    
    async def improve_mapping_with_feedback(
        self,
        entity_text: str,
        correct_mapping: Dict[str, Any],
        project_id: int,
        db: AsyncSession
    ):
        """Improve future mappings based on user feedback"""
        
        # TODO: Implement machine learning feedback incorporation
        # For now, just log the feedback for future analysis
        
        logger.info(f"Mapping feedback received: {entity_text} -> {correct_mapping}")
        
        # Could store in a feedback table for training improvements
        pass

# Global service instance
entity_service = EntityService()