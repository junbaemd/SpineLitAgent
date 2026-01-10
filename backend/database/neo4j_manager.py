"""
Neo4j Knowledge Graph Manager

Manages the knowledge graph for literature relationships in Neo4j.
Stores papers, authors, concepts, and their relationships.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
from datetime import datetime

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class Neo4jManager:
    """
    Manages Neo4j database operations for the literature knowledge graph.
    
    Schema:
        Nodes:
            - Paper: {id, title, abstract, pub_date, citations, importance_score}
            - Author: {name, affiliation}
            - Concept: {name, type}
        
        Relationships:
            - (Paper)-[:CITES {weight}]->(Paper)
            - (Paper)-[:HAS_AUTHOR]->(Author)
            - (Paper)-[:SUPPORTS {strength}]->(Concept)
            - (Paper)-[:CONTRADICTS]->(Concept)
            - (Concept)-[:SYNTHESIZES]->(Concept)
            - (Paper)-[:BELONGS_TO_CLUSTER {cluster_id}]->(Concept)
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "spinelitpassword")
        self._driver: Optional[Driver] = None
    
    @property
    def driver(self) -> Driver:
        """Lazy initialization of the Neo4j driver."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
        return self._driver
    
    def close(self):
        """Close the database connection."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
    
    @contextmanager
    def session(self):
        """Context manager for database sessions."""
        session = self.driver.session()
        try:
            yield session
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """Test if the database connection is working."""
        try:
            with self.session() as session:
                result = session.run("RETURN 1 as test")
                return result.single()["test"] == 1
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def initialize_schema(self):
        """Create indexes and constraints for the schema."""
        with self.session() as session:
            # Indexes for Paper
            session.run("""
                CREATE INDEX paper_id_idx IF NOT EXISTS 
                FOR (p:Paper) ON (p.id)
            """)
            session.run("""
                CREATE INDEX paper_title_idx IF NOT EXISTS 
                FOR (p:Paper) ON (p.title)
            """)
            session.run("""
                CREATE INDEX paper_score_idx IF NOT EXISTS 
                FOR (p:Paper) ON (p.importance_score)
            """)
            
            # Indexes for Author
            session.run("""
                CREATE INDEX author_name_idx IF NOT EXISTS 
                FOR (a:Author) ON (a.name)
            """)
            
            # Indexes for Concept
            session.run("""
                CREATE INDEX concept_name_idx IF NOT EXISTS 
                FOR (c:Concept) ON (c.name)
            """)
            
            # Uniqueness constraint for Paper ID
            session.run("""
                CREATE CONSTRAINT paper_unique_id IF NOT EXISTS 
                FOR (p:Paper) REQUIRE p.id IS UNIQUE
            """)
            
            logger.info("Schema initialized successfully")
    
    # ============ Paper Operations ============
    
    def insert_paper(self, paper: Dict[str, Any]) -> bool:
        """
        Insert or update a paper in the database.
        
        Args:
            paper: Paper dictionary with id, title, abstract, etc.
        
        Returns:
            True if successful, False otherwise
        """
        query = """
        MERGE (p:Paper {id: $id})
        SET p.title = $title,
            p.abstract = $abstract,
            p.journal = $journal,
            p.pub_date = $pub_date,
            p.citations = $citations,
            p.doi = $doi,
            p.pmid = $pmid,
            p.arxiv_id = $arxiv_id,
            p.source = $source,
            p.importance_score = $importance_score,
            p.evidence_weight = $evidence_weight,
            p.updated_at = datetime()
        RETURN p.id as id
        """
        
        try:
            with self.session() as session:
                pub_date = paper.get("pub_date")
                if isinstance(pub_date, datetime):
                    pub_date = pub_date.isoformat()
                
                result = session.run(
                    query,
                    id=paper.get("id", ""),
                    title=paper.get("title", ""),
                    abstract=paper.get("abstract", ""),
                    journal=paper.get("journal", ""),
                    pub_date=pub_date,
                    citations=paper.get("citations", 0),
                    doi=paper.get("doi"),
                    pmid=paper.get("pmid"),
                    arxiv_id=paper.get("arxiv_id"),
                    source=paper.get("source", "unknown"),
                    importance_score=paper.get("importance_score", 0.0),
                    evidence_weight=paper.get("evidence_weight", 0.25)
                )
                result.consume()
                return True
        except Exception as e:
            logger.error(f"Error inserting paper: {e}")
            return False
    
    def insert_papers_batch(self, papers: List[Dict[str, Any]]) -> int:
        """Insert multiple papers in a batch operation."""
        success_count = 0
        for paper in papers:
            if self.insert_paper(paper):
                success_count += 1
        return success_count
    
    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Get a paper by ID."""
        query = """
        MATCH (p:Paper {id: $id})
        RETURN p
        """
        try:
            with self.session() as session:
                result = session.run(query, id=paper_id)
                record = result.single()
                if record:
                    return dict(record["p"])
                return None
        except Exception as e:
            logger.error(f"Error getting paper {paper_id}: {e}")
            return None
    
    def search_papers(
        self,
        query_text: Optional[str] = None,
        min_score: Optional[float] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search papers by title/abstract or filter by score."""
        cypher = "MATCH (p:Paper)"
        conditions = []
        params = {"limit": limit}
        
        if query_text:
            conditions.append("(p.title CONTAINS $query OR p.abstract CONTAINS $query)")
            params["query"] = query_text
        
        if min_score is not None:
            conditions.append("p.importance_score >= $min_score")
            params["min_score"] = min_score
        
        if conditions:
            cypher += " WHERE " + " AND ".join(conditions)
        
        cypher += " RETURN p ORDER BY p.importance_score DESC LIMIT $limit"
        
        try:
            with self.session() as session:
                result = session.run(cypher, **params)
                return [dict(record["p"]) for record in result]
        except Exception as e:
            logger.error(f"Error searching papers: {e}")
            return []
    
    # ============ Author Operations ============
    
    def add_author_to_paper(
        self,
        paper_id: str,
        author_name: str,
        affiliation: Optional[str] = None
    ) -> bool:
        """Add an author relationship to a paper."""
        query = """
        MATCH (p:Paper {id: $paper_id})
        MERGE (a:Author {name: $author_name})
        SET a.affiliation = COALESCE($affiliation, a.affiliation)
        MERGE (p)-[:HAS_AUTHOR]->(a)
        RETURN a.name as name
        """
        try:
            with self.session() as session:
                session.run(
                    query,
                    paper_id=paper_id,
                    author_name=author_name,
                    affiliation=affiliation
                )
                return True
        except Exception as e:
            logger.error(f"Error adding author: {e}")
            return False
    
    # ============ Citation Operations ============
    
    def add_citation(
        self,
        citing_paper_id: str,
        cited_paper_id: str,
        weight: float = 1.0
    ) -> bool:
        """Add a citation relationship between papers."""
        query = """
        MATCH (citing:Paper {id: $citing_id})
        MATCH (cited:Paper {id: $cited_id})
        MERGE (citing)-[r:CITES]->(cited)
        SET r.weight = $weight
        RETURN citing.id as citing, cited.id as cited
        """
        try:
            with self.session() as session:
                session.run(
                    query,
                    citing_id=citing_paper_id,
                    cited_id=cited_paper_id,
                    weight=weight
                )
                return True
        except Exception as e:
            logger.error(f"Error adding citation: {e}")
            return False
    
    def get_citing_papers(self, paper_id: str) -> List[Dict[str, Any]]:
        """Get papers that cite the given paper."""
        query = """
        MATCH (citing:Paper)-[:CITES]->(p:Paper {id: $id})
        RETURN citing
        ORDER BY citing.importance_score DESC
        """
        try:
            with self.session() as session:
                result = session.run(query, id=paper_id)
                return [dict(record["citing"]) for record in result]
        except Exception as e:
            logger.error(f"Error getting citing papers: {e}")
            return []
    
    def get_cited_papers(self, paper_id: str) -> List[Dict[str, Any]]:
        """Get papers cited by the given paper."""
        query = """
        MATCH (p:Paper {id: $id})-[:CITES]->(cited:Paper)
        RETURN cited
        ORDER BY cited.importance_score DESC
        """
        try:
            with self.session() as session:
                result = session.run(query, id=paper_id)
                return [dict(record["cited"]) for record in result]
        except Exception as e:
            logger.error(f"Error getting cited papers: {e}")
            return []
    
    # ============ Concept Operations ============
    
    def add_concept(
        self,
        name: str,
        concept_type: str = "topic"
    ) -> bool:
        """Add a concept node."""
        query = """
        MERGE (c:Concept {name: $name})
        SET c.type = $type
        RETURN c.name as name
        """
        try:
            with self.session() as session:
                session.run(query, name=name, type=concept_type)
                return True
        except Exception as e:
            logger.error(f"Error adding concept: {e}")
            return False
    
    def link_paper_to_concept(
        self,
        paper_id: str,
        concept_name: str,
        relationship: str = "SUPPORTS",
        strength: float = 1.0
    ) -> bool:
        """Link a paper to a concept with a relationship."""
        valid_relationships = ["SUPPORTS", "CONTRADICTS", "MENTIONS"]
        if relationship not in valid_relationships:
            relationship = "MENTIONS"
        
        query = f"""
        MATCH (p:Paper {{id: $paper_id}})
        MERGE (c:Concept {{name: $concept_name}})
        MERGE (p)-[r:{relationship}]->(c)
        SET r.strength = $strength
        RETURN p.id as paper, c.name as concept
        """
        try:
            with self.session() as session:
                session.run(
                    query,
                    paper_id=paper_id,
                    concept_name=concept_name,
                    strength=strength
                )
                return True
        except Exception as e:
            logger.error(f"Error linking paper to concept: {e}")
            return False
    
    def get_papers_for_concept(
        self,
        concept_name: str,
        relationship: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all papers related to a concept."""
        if relationship:
            query = f"""
            MATCH (p:Paper)-[:{relationship}]->(c:Concept {{name: $name}})
            RETURN p
            ORDER BY p.importance_score DESC
            """
        else:
            query = """
            MATCH (p:Paper)-[]->(c:Concept {name: $name})
            RETURN DISTINCT p
            ORDER BY p.importance_score DESC
            """
        
        try:
            with self.session() as session:
                result = session.run(query, name=concept_name)
                return [dict(record["p"]) for record in result]
        except Exception as e:
            logger.error(f"Error getting papers for concept: {e}")
            return []
    
    # ============ Dialectical Operations ============
    
    def store_dialectical_analysis(
        self,
        topic: str,
        thesis_paper_ids: List[str],
        antithesis_paper_ids: List[str],
        synthesis_text: str
    ) -> bool:
        """Store a dialectical analysis result."""
        query = """
        MERGE (thesis:Concept {name: $topic + '_thesis', type: 'thesis'})
        MERGE (antithesis:Concept {name: $topic + '_antithesis', type: 'antithesis'})
        MERGE (synthesis:Concept {name: $topic + '_synthesis', type: 'synthesis'})
        SET synthesis.text = $synthesis_text,
            synthesis.created_at = datetime()
        
        WITH thesis, antithesis, synthesis
        
        UNWIND $thesis_ids as thesis_id
        MATCH (p:Paper {id: thesis_id})
        MERGE (p)-[:SUPPORTS]->(thesis)
        
        WITH thesis, antithesis, synthesis, $antithesis_ids as anti_ids
        
        UNWIND anti_ids as antithesis_id
        MATCH (p:Paper {id: antithesis_id})
        MERGE (p)-[:SUPPORTS]->(antithesis)
        
        WITH thesis, antithesis, synthesis
        MERGE (thesis)-[:SYNTHESIZES]->(synthesis)
        MERGE (antithesis)-[:SYNTHESIZES]->(synthesis)
        
        RETURN synthesis.name as synthesis_name
        """
        try:
            with self.session() as session:
                session.run(
                    query,
                    topic=topic,
                    thesis_ids=thesis_paper_ids,
                    antithesis_ids=antithesis_paper_ids,
                    synthesis_text=synthesis_text
                )
                return True
        except Exception as e:
            logger.error(f"Error storing dialectical analysis: {e}")
            return False
    
    def get_dialectical_analysis(self, topic: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored dialectical analysis."""
        query = """
        MATCH (thesis:Concept {name: $topic + '_thesis'})
        MATCH (antithesis:Concept {name: $topic + '_antithesis'})
        MATCH (synthesis:Concept {name: $topic + '_synthesis'})
        
        OPTIONAL MATCH (tp:Paper)-[:SUPPORTS]->(thesis)
        OPTIONAL MATCH (ap:Paper)-[:SUPPORTS]->(antithesis)
        
        RETURN synthesis.text as synthesis,
               collect(DISTINCT tp.id) as thesis_papers,
               collect(DISTINCT ap.id) as antithesis_papers,
               synthesis.created_at as created_at
        """
        try:
            with self.session() as session:
                result = session.run(query, topic=topic)
                record = result.single()
                if record:
                    return {
                        "topic": topic,
                        "synthesis": record["synthesis"],
                        "thesis_papers": record["thesis_papers"],
                        "antithesis_papers": record["antithesis_papers"],
                        "created_at": record["created_at"]
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting dialectical analysis: {e}")
            return None
    
    # ============ Custom Query ============
    
    def execute_cypher(self, query: str, **params) -> List[Dict[str, Any]]:
        """Execute a custom Cypher query."""
        try:
            with self.session() as session:
                result = session.run(query, **params)
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Error executing Cypher: {e}")
            return []
    
    # ============ Statistics ============
    
    def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        query = """
        MATCH (p:Paper) WITH count(p) as papers
        MATCH (a:Author) WITH papers, count(a) as authors
        MATCH (c:Concept) WITH papers, authors, count(c) as concepts
        MATCH ()-[r:CITES]->() WITH papers, authors, concepts, count(r) as citations
        RETURN papers, authors, concepts, citations
        """
        try:
            with self.session() as session:
                result = session.run(query)
                record = result.single()
                if record:
                    return dict(record)
                return {"papers": 0, "authors": 0, "concepts": 0, "citations": 0}
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"papers": 0, "authors": 0, "concepts": 0, "citations": 0}
    
    def clear_database(self) -> bool:
        """Clear all data from the database. USE WITH CAUTION."""
        try:
            with self.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                return True
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False
