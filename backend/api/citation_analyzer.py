"""
Citation Analyzer Module

Calculates importance scores for papers based on citations,
journal quality, and recency factors.
"""

import os
import math
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class JournalMetrics:
    """Journal quality metrics."""
    name: str
    quartile: str  # Q1, Q2, Q3, Q4
    impact_factor: Optional[float] = None
    h_index: Optional[int] = None


# Common spine and neurosurgery journal weights (hardcoded fallback)
JOURNAL_WEIGHTS = {
    # Q1 Journals (weight: 1.0)
    "spine": 1.0,
    "the spine journal": 1.0,
    "spine journal": 1.0,
    "european spine journal": 1.0,
    "neurosurgery": 1.0,
    "journal of neurosurgery: spine": 1.0,
    "journal of neurosurgery spine": 1.0,
    "jns spine": 1.0,
    "the lancet": 1.0,
    "new england journal of medicine": 1.0,
    "nejm": 1.0,
    "jama": 1.0,
    "jama surgery": 1.0,
    "nature medicine": 1.0,
    
    # Q2 Journals (weight: 0.75)
    "clinical spine surgery": 0.75,
    "world neurosurgery": 0.75,
    "global spine journal": 0.75,
    "asian spine journal": 0.75,
    "journal of spinal disorders and techniques": 0.75,
    "operative neurosurgery": 0.75,
    
    # Q3 Journals (weight: 0.5)
    "journal of korean neurosurgical society": 0.5,
    "neurospine": 0.5,
    "surgical neurology international": 0.5,
    
    # Q4 and others (weight: 0.25)
    "default": 0.25
}


class CitationAnalyzer:
    """
    Analyzes paper importance based on citations, journal quality, and recency.
    
    Importance Score Formula:
        score = (citations / years_since_pub) * journal_weight * e^(-0.1 * years_since)
    
    Where:
        - citations: Total citation count
        - years_since_pub: Years since publication (minimum 0.5 to avoid division issues)
        - journal_weight: Based on journal quartile (Q1=1.0, Q2=0.75, Q3=0.5, Q4=0.25)
        - Recency factor: Exponential decay e^(-0.1 * years) favoring recent papers
    """
    
    DECAY_RATE = 0.1
    MIN_YEARS = 0.5  # Minimum years to avoid division by very small numbers
    
    def __init__(self):
        self.journal_cache: Dict[str, float] = {}
        self.client = httpx.AsyncClient(timeout=15.0)
    
    def get_journal_weight(self, journal_name: Optional[str]) -> float:
        """
        Get the quality weight for a journal.
        
        Uses hardcoded weights for common spine journals,
        with 0.25 as default for unknown journals.
        """
        if not journal_name:
            return JOURNAL_WEIGHTS["default"]
        
        journal_lower = journal_name.lower().strip()
        
        # Check cache first
        if journal_lower in self.journal_cache:
            return self.journal_cache[journal_lower]
        
        # Check hardcoded weights
        for key, weight in JOURNAL_WEIGHTS.items():
            if key in journal_lower or journal_lower in key:
                self.journal_cache[journal_lower] = weight
                return weight
        
        # Default weight for unknown journals
        self.journal_cache[journal_lower] = JOURNAL_WEIGHTS["default"]
        return JOURNAL_WEIGHTS["default"]
    
    def calculate_years_since_publication(
        self,
        pub_date: Optional[datetime],
        reference_date: Optional[datetime] = None
    ) -> float:
        """
        Calculate years since publication.
        
        Returns minimum of MIN_YEARS for very recent papers.
        """
        if not pub_date:
            return 5.0  # Default assumption for missing dates
        
        if reference_date is None:
            reference_date = datetime.now()
        
        delta = reference_date - pub_date
        years = delta.days / 365.25
        
        return max(years, self.MIN_YEARS)
    
    def calculate_recency_factor(self, years_since: float) -> float:
        """
        Calculate exponential recency factor.
        
        Recent papers get higher weight.
        """
        return math.exp(-self.DECAY_RATE * years_since)
    
    def calculate_importance_score(
        self,
        citations: int,
        pub_date: Optional[datetime],
        journal: Optional[str] = None,
        reference_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Calculate the comprehensive importance score for a paper.
        
        Args:
            citations: Number of citations
            pub_date: Publication date
            journal: Journal name
            reference_date: Reference date for calculations (default: now)
        
        Returns:
            Dictionary with score and component breakdown
        """
        years_since = self.calculate_years_since_publication(pub_date, reference_date)
        journal_weight = self.get_journal_weight(journal)
        recency_factor = self.calculate_recency_factor(years_since)
        
        # Citation rate (citations per year)
        citation_rate = citations / years_since
        
        # Final importance score
        importance_score = citation_rate * journal_weight * recency_factor
        
        return {
            "importance_score": round(importance_score, 4),
            "citation_rate": round(citation_rate, 2),
            "years_since_publication": round(years_since, 2),
            "journal_weight": journal_weight,
            "recency_factor": round(recency_factor, 4),
            "raw_citations": citations
        }
    
    def analyze_paper(self, paper_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a paper dict and add importance metrics.
        
        Args:
            paper_dict: Paper dictionary with at minimum:
                - citations (int)
                - pub_date (datetime or ISO string)
                - journal (str, optional)
        
        Returns:
            Updated paper dict with importance_metrics added
        """
        citations = paper_dict.get("citations", 0)
        journal = paper_dict.get("journal", "")
        
        # Parse pub_date if string
        pub_date = paper_dict.get("pub_date")
        if isinstance(pub_date, str):
            try:
                pub_date = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
            except ValueError:
                pub_date = None
        
        metrics = self.calculate_importance_score(
            citations=citations,
            pub_date=pub_date,
            journal=journal
        )
        
        paper_dict["importance_metrics"] = metrics
        paper_dict["importance_score"] = metrics["importance_score"]
        
        return paper_dict
    
    def rank_papers(
        self,
        papers: List[Dict[str, Any]],
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze and rank a list of papers by importance score.
        
        Args:
            papers: List of paper dictionaries
            min_score: Optional minimum score threshold
        
        Returns:
            Sorted list of papers with importance metrics
        """
        analyzed = [self.analyze_paper(p.copy()) for p in papers]
        
        if min_score is not None:
            analyzed = [p for p in analyzed if p["importance_score"] >= min_score]
        
        return sorted(analyzed, key=lambda x: x["importance_score"], reverse=True)
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


class EvidenceClassifier:
    """
    Classifies papers by evidence level based on study type.
    
    Evidence Hierarchy:
        1. RCT / Meta-analysis (weight: 1.0)
        2. Systematic review (weight: 0.9)
        3. Prospective cohort (weight: 0.75)
        4. Retrospective cohort (weight: 0.6)
        5. Case-control (weight: 0.5)
        6. Case series (weight: 0.3)
        7. Case report (weight: 0.2)
        8. Expert opinion / Editorial (weight: 0.1)
    """
    
    EVIDENCE_WEIGHTS = {
        "meta-analysis": 1.0,
        "randomized controlled trial": 1.0,
        "rct": 1.0,
        "systematic review": 0.9,
        "prospective cohort": 0.75,
        "prospective study": 0.75,
        "cohort study": 0.65,
        "retrospective cohort": 0.6,
        "retrospective study": 0.6,
        "case-control": 0.5,
        "case control": 0.5,
        "case series": 0.3,
        "case report": 0.2,
        "expert opinion": 0.1,
        "editorial": 0.1,
        "review": 0.4,
        "narrative review": 0.3,
    }
    
    # Keywords to detect study types from abstract/title
    STUDY_TYPE_KEYWORDS = {
        "meta-analysis": ["meta-analysis", "meta analysis", "metaanalysis"],
        "randomized controlled trial": ["randomized", "randomised", "rct", "controlled trial"],
        "systematic review": ["systematic review", "systematically reviewed"],
        "prospective cohort": ["prospective cohort", "prospective study", "prospectively"],
        "retrospective cohort": ["retrospective cohort", "retrospective study", "retrospective analysis"],
        "case-control": ["case-control", "case control"],
        "case series": ["case series", "consecutive cases", "consecutive patients"],
        "case report": ["case report", "single case"],
    }
    
    def classify_study_type(
        self,
        title: str = "",
        abstract: str = "",
        mesh_terms: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Classify the study type based on title, abstract, and MeSH terms.
        
        Returns:
            Dictionary with study_type, evidence_weight, and confidence
        """
        text = f"{title} {abstract}".lower()
        
        # Check MeSH terms first (most reliable)
        if mesh_terms:
            mesh_lower = [m.lower() for m in mesh_terms]
            for mesh in mesh_lower:
                for study_type, weight in self.EVIDENCE_WEIGHTS.items():
                    if study_type in mesh:
                        return {
                            "study_type": study_type,
                            "evidence_weight": weight,
                            "confidence": 0.9,
                            "source": "mesh_terms"
                        }
        
        # Check keywords in text
        for study_type, keywords in self.STUDY_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    return {
                        "study_type": study_type,
                        "evidence_weight": self.EVIDENCE_WEIGHTS[study_type],
                        "confidence": 0.7,
                        "source": "keyword_match"
                    }
        
        # Default to unknown
        return {
            "study_type": "unknown",
            "evidence_weight": 0.25,
            "confidence": 0.3,
            "source": "default"
        }
    
    def add_evidence_classification(
        self,
        paper_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add evidence classification to a paper dictionary.
        """
        classification = self.classify_study_type(
            title=paper_dict.get("title", ""),
            abstract=paper_dict.get("abstract", ""),
            mesh_terms=paper_dict.get("mesh_terms", [])
        )
        
        paper_dict["evidence_classification"] = classification
        paper_dict["evidence_weight"] = classification["evidence_weight"]
        
        return paper_dict


# Convenience function
def analyze_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Analyze and rank papers with both importance and evidence scores.
    
    Example:
        analyzed = analyze_papers([paper1_dict, paper2_dict])
    """
    analyzer = CitationAnalyzer()
    classifier = EvidenceClassifier()
    
    results = []
    for paper in papers:
        p = paper.copy()
        p = analyzer.analyze_paper(p)
        p = classifier.add_evidence_classification(p)
        
        # Combined score: importance * evidence_weight
        p["combined_score"] = round(
            p["importance_score"] * p["evidence_weight"],
            4
        )
        results.append(p)
    
    return sorted(results, key=lambda x: x["combined_score"], reverse=True)
