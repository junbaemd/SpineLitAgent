"""
SpineLitAgent FastAPI Main Application

API endpoints for literature discovery, citation analysis,
dialectical synthesis, and knowledge graph queries.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============ Pydantic Models ============

class PaperSearchRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    max_results: int = Field(default=50, ge=1, le=200)
    sources: Optional[List[str]] = Field(
        default=None,
        description="Sources to search: pubmed, semantic_scholar, arxiv"
    )


class PaperResponse(BaseModel):
    id: str
    source: str
    title: str
    abstract: str
    authors: List[str]
    journal: Optional[str]
    pub_date: Optional[str]
    doi: Optional[str]
    citations: int
    importance_score: Optional[float] = None
    evidence_weight: Optional[float] = None


class DialecticRequest(BaseModel):
    topic: str = Field(..., description="Topic for dialectical analysis")
    paper_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of paper IDs to analyze"
    )
    min_papers: int = Field(default=3, ge=2)


class DialecticResponse(BaseModel):
    topic: str
    thesis: Dict[str, Any]
    antithesis: Dict[str, Any]
    synthesis: Dict[str, Any]
    paper_clusters: Dict[str, List[str]]


class CypherRequest(BaseModel):
    query: str = Field(..., description="Cypher query to execute")
    params: Optional[Dict[str, Any]] = Field(default=None)


class HealthResponse(BaseModel):
    status: str
    database: str
    scheduler: str
    timestamp: str


class SchedulerJobResponse(BaseModel):
    id: str
    name: Optional[str]
    next_run: Optional[str]
    trigger: str


# ============ Lifespan Management ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    from backend.scheduler import get_scheduler
    from backend.database.neo4j_manager import Neo4jManager
    
    logger.info("Starting SpineLitAgent...")
    
    # Initialize database schema
    try:
        db = Neo4jManager()
        if db.test_connection():
            db.initialize_schema()
            logger.info("Database connected and schema initialized")
        else:
            logger.warning("Database connection failed - running without persistence")
        db.close()
    except Exception as e:
        logger.warning(f"Database initialization error: {e}")
    
    # Start scheduler
    scheduler = get_scheduler()
    scheduler.start(run_immediately=False)
    logger.info("Scheduler started")
    
    yield
    
    # Cleanup
    scheduler.stop()
    logger.info("SpineLitAgent shutdown complete")


# ============ FastAPI App ============

app = FastAPI(
    title="SpineLitAgent API",
    description="Literature review AI agent for spine surgery research",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Health Check ============

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    from datetime import datetime
    from backend.database.neo4j_manager import Neo4jManager
    from backend.scheduler import get_scheduler
    
    db_status = "unknown"
    try:
        db = Neo4jManager()
        db_status = "connected" if db.test_connection() else "disconnected"
        db.close()
    except Exception:
        db_status = "error"
    
    scheduler = get_scheduler()
    scheduler_status = "running" if scheduler.scheduler.running else "stopped"
    
    return HealthResponse(
        status="healthy",
        database=db_status,
        scheduler=scheduler_status,
        timestamp=datetime.now().isoformat()
    )


# ============ Paper Search ============

@app.post("/api/papers/search", response_model=List[PaperResponse])
async def search_papers(request: PaperSearchRequest):
    """
    Search for papers across multiple sources.
    
    - **query**: Search query (supports boolean operators)
    - **max_results**: Maximum total results (default: 50)
    - **sources**: List of sources to search (default: all)
    """
    from backend.api.literature_fetcher import LiteratureAggregator
    from backend.api.citation_analyzer import CitationAnalyzer
    
    try:
        aggregator = LiteratureAggregator()
        analyzer = CitationAnalyzer()
        
        papers = await aggregator.search_all(
            request.query,
            max_results_per_source=request.max_results // 3,
            sources=request.sources
        )
        
        # Analyze and format results
        results = []
        for paper in papers:
            paper_dict = paper.to_dict()
            paper_dict = analyzer.analyze_paper(paper_dict)
            
            results.append(PaperResponse(
                id=paper_dict["id"],
                source=paper_dict["source"],
                title=paper_dict["title"],
                abstract=paper_dict["abstract"],
                authors=paper_dict["authors"],
                journal=paper_dict.get("journal"),
                pub_date=paper_dict.get("pub_date"),
                doi=paper_dict.get("doi"),
                citations=paper_dict.get("citations", 0),
                importance_score=paper_dict.get("importance_score"),
                evidence_weight=paper_dict.get("evidence_weight")
            ))
        
        await aggregator.close()
        
        # Sort by importance score
        results.sort(key=lambda x: x.importance_score or 0, reverse=True)
        
        return results[:request.max_results]
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/papers/{paper_id}")
async def get_paper(paper_id: str):
    """Get details for a specific paper."""
    from backend.database.neo4j_manager import Neo4jManager
    
    try:
        db = Neo4jManager()
        paper = db.get_paper(paper_id)
        db.close()
        
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        
        return paper
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get paper error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/papers/{paper_id}/citations")
async def get_paper_citations(
    paper_id: str,
    direction: str = Query("both", regex="^(citing|cited|both)$")
):
    """
    Get citation information for a paper.
    
    - **direction**: 'citing' (papers that cite this), 'cited' (papers this cites), 'both'
    """
    from backend.database.neo4j_manager import Neo4jManager
    
    try:
        db = Neo4jManager()
        
        result = {"paper_id": paper_id}
        
        if direction in ["citing", "both"]:
            result["citing_papers"] = db.get_citing_papers(paper_id)
        
        if direction in ["cited", "both"]:
            result["cited_papers"] = db.get_cited_papers(paper_id)
        
        db.close()
        return result
        
    except Exception as e:
        logger.error(f"Citations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Dialectical Analysis ============

@app.post("/api/analysis/dialectic", response_model=DialecticResponse)
async def analyze_dialectic(request: DialecticRequest, background_tasks: BackgroundTasks):
    """
    Perform dialectical analysis on papers for a topic.
    
    Clusters papers into thesis and antithesis groups,
    extracts positions, and generates a synthesis.
    """
    from backend.analysis.dialectic_engine import DialecticEngine
    from backend.database.neo4j_manager import Neo4jManager
    
    try:
        engine = DialecticEngine()
        db = Neo4jManager()
        
        # Get papers
        if request.paper_ids:
            papers = [db.get_paper(pid) for pid in request.paper_ids]
            papers = [p for p in papers if p is not None]
        else:
            papers = db.search_papers(query_text=request.topic, limit=50)
        
        if len(papers) < request.min_papers:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough papers found. Need at least {request.min_papers}, found {len(papers)}"
            )
        
        # Run analysis
        analysis = engine.analyze(request.topic, papers)
        
        # Store results in background
        background_tasks.add_task(
            db.store_dialectical_analysis,
            topic=request.topic,
            thesis_paper_ids=analysis.thesis.supporting_papers,
            antithesis_paper_ids=analysis.antithesis.supporting_papers,
            synthesis_text=analysis.synthesis.text
        )
        
        return DialecticResponse(
            topic=analysis.topic,
            thesis={"text": analysis.thesis.text, "supporting_papers": analysis.thesis.supporting_papers, "evidence_strength": analysis.thesis.evidence_strength},
            antithesis={"text": analysis.antithesis.text, "supporting_papers": analysis.antithesis.supporting_papers, "evidence_strength": analysis.antithesis.evidence_strength},
            synthesis={"text": analysis.synthesis.text, "confidence": analysis.synthesis.confidence},
            paper_clusters=analysis.paper_clusters
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dialectic analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analysis/dialectic/{topic}")
async def get_stored_dialectic(topic: str):
    """Get a previously stored dialectical analysis."""
    from backend.database.neo4j_manager import Neo4jManager
    
    try:
        db = Neo4jManager()
        result = db.get_dialectical_analysis(topic)
        db.close()
        
        if not result:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get dialectic error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Graph Queries ============

@app.post("/api/graph/query")
async def execute_graph_query(request: CypherRequest):
    """
    Execute a custom Cypher query on the knowledge graph.
    
    **CAUTION**: This endpoint executes arbitrary Cypher. Use with care.
    """
    from backend.database.neo4j_manager import Neo4jManager
    
    # Basic safety check
    dangerous_keywords = ["DELETE", "REMOVE", "DROP", "CREATE INDEX", "CREATE CONSTRAINT"]
    query_upper = request.query.upper()
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            raise HTTPException(
                status_code=400,
                detail=f"Query contains prohibited keyword: {keyword}"
            )
    
    try:
        db = Neo4jManager()
        results = db.execute_cypher(request.query, **(request.params or {}))
        db.close()
        
        return {"results": results, "count": len(results)}
        
    except Exception as e:
        logger.error(f"Graph query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/stats")
async def get_graph_stats():
    """Get knowledge graph statistics."""
    from backend.database.neo4j_manager import Neo4jManager
    
    try:
        db = Neo4jManager()
        stats = db.get_stats()
        db.close()
        
        return stats
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Scheduler Endpoints ============

@app.get("/api/scheduler/jobs", response_model=List[SchedulerJobResponse])
async def get_scheduler_jobs():
    """Get status of all scheduled jobs."""
    from backend.scheduler import get_scheduler
    
    scheduler = get_scheduler()
    jobs = scheduler.get_job_status()
    
    return [
        SchedulerJobResponse(
            id=job["id"],
            name=job.get("name"),
            next_run=job.get("next_run"),
            trigger=job["trigger"]
        )
        for job in jobs
    ]


@app.post("/api/scheduler/trigger/fetch")
async def trigger_manual_fetch(
    query: str = Query(..., description="Search query to fetch"),
    background_tasks: BackgroundTasks = None
):
    """Trigger a manual literature fetch."""
    from backend.scheduler import get_scheduler
    
    scheduler = get_scheduler()
    result = await scheduler.trigger_manual_fetch(query)
    
    return result


@app.post("/api/scheduler/trigger/dialectic")
async def trigger_manual_dialectic(request: DialecticRequest):
    """Trigger a manual dialectical analysis."""
    from backend.scheduler import get_scheduler
    
    scheduler = get_scheduler()
    result = await scheduler.trigger_manual_dialectic(
        request.topic,
        request.paper_ids
    )
    
    return result


# ============ Concept Endpoints ============

@app.get("/api/concepts/{concept_name}/papers")
async def get_papers_for_concept(
    concept_name: str,
    relationship: Optional[str] = Query(None, regex="^(SUPPORTS|CONTRADICTS|MENTIONS)$")
):
    """Get all papers related to a concept."""
    from backend.database.neo4j_manager import Neo4jManager
    
    try:
        db = Neo4jManager()
        papers = db.get_papers_for_concept(concept_name, relationship)
        db.close()
        
        return {"concept": concept_name, "papers": papers, "count": len(papers)}
        
    except Exception as e:
        logger.error(f"Concept papers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Run with Uvicorn ============

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
