"""
Scheduler Module

Implements automated scheduling for literature updates using APScheduler.
Daily, weekly, and monthly tasks for keeping the knowledge base current.
"""

import os
import logging
from datetime import datetime
from typing import Optional, Callable, List, Dict, Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class LiteratureScheduler:
    """
    Scheduler for automated literature processing tasks.
    
    Tasks:
        - Daily: Fetch new papers for configured queries
        - Weekly: Recalculate citation scores
        - Monthly: Regenerate dialectical syntheses
    """
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.queries: List[str] = [
            "PELD spine endoscopic",
            "lumbar disc herniation minimally invasive",
            "spinal stenosis endoscopic decompression",
            "percutaneous endoscopic discectomy"
        ]
        self.high_impact_threshold = 10.0
        self._alert_callback: Optional[Callable] = None
    
    def set_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback for high-impact paper alerts."""
        self._alert_callback = callback
    
    def add_query(self, query: str):
        """Add a query to the daily fetch list."""
        if query not in self.queries:
            self.queries.append(query)
            logger.info(f"Added query: {query}")
    
    def remove_query(self, query: str):
        """Remove a query from the daily fetch list."""
        if query in self.queries:
            self.queries.remove(query)
            logger.info(f"Removed query: {query}")
    
    async def daily_fetch_task(self):
        """
        Daily task: Fetch new papers for all configured queries.
        """
        from backend.api.literature_fetcher import LiteratureAggregator
        from backend.api.citation_analyzer import CitationAnalyzer
        from backend.database.neo4j_manager import Neo4jManager
        
        logger.info("Starting daily fetch task...")
        
        try:
            aggregator = LiteratureAggregator()
            analyzer = CitationAnalyzer()
            db = Neo4jManager()
            
            total_papers = 0
            high_impact_papers = []
            
            for query in self.queries:
                logger.info(f"Fetching papers for: {query}")
                papers = await aggregator.search_all(
                    query,
                    max_results_per_source=20,
                    sources=["pubmed", "semantic_scholar"]
                )
                
                for paper in papers:
                    # Analyze and score
                    paper_dict = paper.to_dict()
                    paper_dict = analyzer.analyze_paper(paper_dict)
                    
                    # Store in database
                    db.insert_paper(paper_dict)
                    
                    # Check for high impact
                    if paper_dict.get("importance_score", 0) > self.high_impact_threshold:
                        high_impact_papers.append(paper_dict)
                    
                    total_papers += 1
            
            # Alert for high impact papers
            if high_impact_papers and self._alert_callback:
                for paper in high_impact_papers:
                    self._alert_callback({
                        "type": "high_impact_paper",
                        "paper_id": paper.get("id"),
                        "title": paper.get("title"),
                        "score": paper.get("importance_score"),
                        "timestamp": datetime.now().isoformat()
                    })
            
            await aggregator.close()
            db.close()
            
            logger.info(f"Daily fetch complete. Processed {total_papers} papers, {len(high_impact_papers)} high-impact.")
            
        except Exception as e:
            logger.error(f"Daily fetch task error: {e}")
    
    async def weekly_citation_update(self):
        """
        Weekly task: Update citation counts and recalculate scores.
        """
        from backend.api.literature_fetcher import SemanticScholarFetcher
        from backend.api.citation_analyzer import CitationAnalyzer
        from backend.database.neo4j_manager import Neo4jManager
        
        logger.info("Starting weekly citation update...")
        
        try:
            fetcher = SemanticScholarFetcher()
            analyzer = CitationAnalyzer()
            db = Neo4jManager()
            
            # Get all papers from database
            papers = db.search_papers(limit=1000)
            updated_count = 0
            
            for paper in papers:
                paper_id = paper.get("id", "")
                
                # Skip if not from semantic scholar or no DOI
                doi = paper.get("doi")
                if not doi:
                    continue
                
                # Try to get updated citation count
                try:
                    updated = await fetcher.get_paper_details(f"DOI:{doi}")
                    if updated:
                        paper["citations"] = updated.citations
                        paper = analyzer.analyze_paper(paper)
                        db.insert_paper(paper)
                        updated_count += 1
                except Exception:
                    continue
            
            await fetcher.close()
            db.close()
            
            logger.info(f"Weekly citation update complete. Updated {updated_count} papers.")
            
        except Exception as e:
            logger.error(f"Weekly citation update error: {e}")
    
    async def monthly_synthesis_regeneration(self):
        """
        Monthly task: Regenerate dialectical syntheses for key topics.
        """
        from backend.analysis.dialectic_engine import DialecticEngine
        from backend.database.neo4j_manager import Neo4jManager
        
        logger.info("Starting monthly synthesis regeneration...")
        
        topics = [
            "PELD recurrence",
            "endoscopic spine surgery complications",
            "annular closure devices efficacy"
        ]
        
        try:
            engine = DialecticEngine()
            db = Neo4jManager()
            
            for topic in topics:
                logger.info(f"Regenerating synthesis for: {topic}")
                
                # Get papers for topic
                papers = db.get_papers_for_concept(topic)
                
                if len(papers) < 3:
                    logger.warning(f"Not enough papers for {topic}, skipping")
                    continue
                
                # Run dialectical analysis
                analysis = engine.analyze(topic, papers)
                
                # Store results
                db.store_dialectical_analysis(
                    topic=topic,
                    thesis_paper_ids=analysis.thesis.supporting_papers,
                    antithesis_paper_ids=analysis.antithesis.supporting_papers,
                    synthesis_text=analysis.synthesis.text
                )
                
                logger.info(f"Synthesis regenerated for: {topic}")
            
            db.close()
            logger.info("Monthly synthesis regeneration complete.")
            
        except Exception as e:
            logger.error(f"Monthly synthesis regeneration error: {e}")
    
    def start(self, run_immediately: bool = False):
        """
        Start the scheduler with all configured tasks.
        
        Args:
            run_immediately: If True, run all tasks once immediately
        """
        # Daily fetch at 6 AM
        self.scheduler.add_job(
            self.daily_fetch_task,
            CronTrigger(hour=6, minute=0),
            id="daily_fetch",
            replace_existing=True
        )
        
        # Weekly citation update on Sundays at 3 AM
        self.scheduler.add_job(
            self.weekly_citation_update,
            CronTrigger(day_of_week="sun", hour=3, minute=0),
            id="weekly_citation",
            replace_existing=True
        )
        
        # Monthly synthesis on 1st of each month at 2 AM
        self.scheduler.add_job(
            self.monthly_synthesis_regeneration,
            CronTrigger(day=1, hour=2, minute=0),
            id="monthly_synthesis",
            replace_existing=True
        )
        
        self.scheduler.start()
        logger.info("Scheduler started")
        
        if run_immediately:
            logger.info("Running immediate task execution...")
            # Add immediate jobs
            self.scheduler.add_job(
                self.daily_fetch_task,
                id="immediate_fetch",
                replace_existing=True
            )
    
    def stop(self):
        """Stop the scheduler."""
        self.scheduler.shutdown()
        logger.info("Scheduler stopped")
    
    def get_job_status(self) -> List[Dict[str, Any]]:
        """Get status of all scheduled jobs."""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger)
            })
        return jobs
    
    async def trigger_manual_fetch(self, query: str) -> Dict[str, Any]:
        """
        Trigger a manual fetch for a specific query.
        
        Returns:
            Summary of the fetch operation
        """
        from backend.api.literature_fetcher import LiteratureAggregator
        from backend.api.citation_analyzer import CitationAnalyzer
        from backend.database.neo4j_manager import Neo4jManager
        
        logger.info(f"Manual fetch triggered for: {query}")
        
        try:
            aggregator = LiteratureAggregator()
            analyzer = CitationAnalyzer()
            db = Neo4jManager()
            
            papers = await aggregator.search_all(
                query,
                max_results_per_source=30
            )
            
            results = []
            for paper in papers:
                paper_dict = paper.to_dict()
                paper_dict = analyzer.analyze_paper(paper_dict)
                db.insert_paper(paper_dict)
                results.append({
                    "id": paper_dict.get("id"),
                    "title": paper_dict.get("title"),
                    "score": paper_dict.get("importance_score")
                })
            
            await aggregator.close()
            db.close()
            
            return {
                "query": query,
                "papers_fetched": len(results),
                "top_papers": sorted(
                    results,
                    key=lambda x: x.get("score", 0),
                    reverse=True
                )[:5]
            }
            
        except Exception as e:
            logger.error(f"Manual fetch error: {e}")
            return {"query": query, "error": str(e)}
    
    async def trigger_manual_dialectic(
        self,
        topic: str,
        paper_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Trigger a manual dialectical analysis.
        
        Args:
            topic: Topic to analyze
            paper_ids: Optional list of paper IDs to include
        
        Returns:
            Dialectical analysis results
        """
        from backend.analysis.dialectic_engine import DialecticEngine
        from backend.database.neo4j_manager import Neo4jManager
        
        logger.info(f"Manual dialectic analysis triggered for: {topic}")
        
        try:
            engine = DialecticEngine()
            db = Neo4jManager()
            
            if paper_ids:
                papers = [db.get_paper(pid) for pid in paper_ids]
                papers = [p for p in papers if p is not None]
            else:
                papers = db.search_papers(query_text=topic, limit=50)
            
            if not papers:
                return {"topic": topic, "error": "No papers found"}
            
            analysis = engine.analyze(topic, papers)
            
            # Store results
            db.store_dialectical_analysis(
                topic=topic,
                thesis_paper_ids=analysis.thesis.supporting_papers,
                antithesis_paper_ids=analysis.antithesis.supporting_papers,
                synthesis_text=analysis.synthesis.text
            )
            
            db.close()
            
            return analysis.to_dict()
            
        except Exception as e:
            logger.error(f"Manual dialectic error: {e}")
            return {"topic": topic, "error": str(e)}


# Singleton instance
_scheduler_instance: Optional[LiteratureScheduler] = None


def get_scheduler() -> LiteratureScheduler:
    """Get or create the scheduler singleton."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = LiteratureScheduler()
    return _scheduler_instance
