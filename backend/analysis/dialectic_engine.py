"""
Dialectical Analysis Engine

Implements thesis-antithesis-synthesis reasoning using Claude API
for comprehensive analysis of contradictory findings in literature.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

import numpy as np
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class PositionType(str, Enum):
    """Types of positions in dialectical analysis."""
    THESIS = "thesis"
    ANTITHESIS = "antithesis"
    SYNTHESIS = "synthesis"


@dataclass
class Position:
    """A position (claim) extracted from literature."""
    text: str
    position_type: PositionType
    supporting_papers: List[str] = field(default_factory=list)
    evidence_strength: float = 0.0
    confidence: float = 0.0


@dataclass
class DialecticalAnalysis:
    """Complete dialectical analysis result."""
    topic: str
    thesis: Position
    antithesis: Position
    synthesis: Position
    paper_clusters: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "topic": self.topic,
            "thesis": asdict(self.thesis),
            "antithesis": asdict(self.antithesis),
            "synthesis": asdict(self.synthesis),
            "paper_clusters": self.paper_clusters,
            "metadata": self.metadata
        }


class EmbeddingClusterer:
    """Clusters papers based on semantic embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        return self.model.encode(texts, show_progress_bar=False)
    
    def cluster_papers(
        self,
        papers: List[Dict[str, Any]],
        n_clusters: int = 2,
        text_field: str = "abstract"
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Cluster papers based on their abstracts/text content.
        
        Args:
            papers: List of paper dictionaries
            n_clusters: Number of clusters (typically 2 for thesis/antithesis)
            text_field: Which field to use for clustering
        
        Returns:
            Dictionary mapping cluster ID to list of papers
        """
        if not papers:
            return {}
        
        texts = [p.get(text_field, p.get("title", "")) for p in papers]
        embeddings = self.embed_texts(texts)
        
        # Adjust n_clusters if we have fewer papers
        actual_clusters = min(n_clusters, len(papers))
        
        if actual_clusters == 1:
            return {0: papers}
        
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(papers[i])
        
        return clusters
    
    def find_contradicting_pairs(
        self,
        papers: List[Dict[str, Any]],
        threshold: float = 0.3
    ) -> List[Tuple[Dict, Dict, float]]:
        """
        Find pairs of papers that may have contradicting positions.
        
        Papers with moderate similarity (not too similar, not too different)
        on similar topics but different conclusions are candidates.
        
        Returns:
            List of (paper1, paper2, similarity_score) tuples
        """
        if len(papers) < 2:
            return []
        
        texts = [p.get("abstract", p.get("title", "")) for p in papers]
        embeddings = self.embed_texts(texts)
        
        similarities = cosine_similarity(embeddings)
        
        pairs = []
        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):
                sim = similarities[i][j]
                # Look for papers that are related but not identical
                if threshold < sim < 0.8:
                    pairs.append((papers[i], papers[j], float(sim)))
        
        # Sort by similarity (descending)
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        return pairs


class PositionExtractor:
    """Extracts positions and claims from papers using Claude API."""
    
    EXTRACTION_PROMPT = """Analyze the following research paper abstract and extract the main position/claim.

Paper Title: {title}
Abstract: {abstract}
Journal: {journal}
Citations: {citations}

Extract:
1. Main Claim: What is the paper's primary argument or finding?
2. Evidence Type: What type of evidence supports this? (RCT, cohort, case series, etc.)
3. Strength: Rate the evidence strength 0.0-1.0
4. Key Findings: List 2-3 specific findings or data points

Respond in JSON format:
{{
    "main_claim": "...",
    "evidence_type": "...",
    "evidence_strength": 0.0,
    "key_findings": ["...", "..."],
    "limitations": ["..."]
}}"""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
    
    def extract_position(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract the main position from a single paper.
        """
        prompt = self.EXTRACTION_PROMPT.format(
            title=paper.get("title", "Unknown"),
            abstract=paper.get("abstract", "No abstract available"),
            journal=paper.get("journal", "Unknown"),
            citations=paper.get("citations", 0)
        )
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.content[0].text
            
            # Parse JSON from response
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            return json.loads(content.strip())
            
        except Exception as e:
            logger.error(f"Error extracting position: {e}")
            return {
                "main_claim": paper.get("title", "Unknown claim"),
                "evidence_type": "unknown",
                "evidence_strength": 0.25,
                "key_findings": [],
                "limitations": [],
                "error": str(e)
            }
    
    def extract_positions_batch(
        self,
        papers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract positions from multiple papers."""
        results = []
        for paper in papers:
            position = self.extract_position(paper)
            position["paper_id"] = paper.get("id", "unknown")
            position["paper_title"] = paper.get("title", "Unknown")
            results.append(position)
        return results


class ContradictionDetector:
    """Detects contradictions between paper positions."""
    
    DETECTION_PROMPT = """Analyze these two research positions and determine if they contradict each other.

Position A (from "{paper_a_title}"):
{position_a}

Position B (from "{paper_b_title}"):
{position_b}

Analyze:
1. Are these positions on the same topic/question?
2. Do they reach different or opposing conclusions?
3. If contradicting, what specifically is the disagreement?

Respond in JSON:
{{
    "same_topic": true/false,
    "contradicts": true/false,
    "contradiction_type": "direct" | "partial" | "methodological" | "none",
    "explanation": "...",
    "thesis_position": "...",
    "antithesis_position": "..."
}}"""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
    
    def detect_contradiction(
        self,
        paper_a: Dict[str, Any],
        position_a: Dict[str, Any],
        paper_b: Dict[str, Any],
        position_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect if two positions contradict each other."""
        
        prompt = self.DETECTION_PROMPT.format(
            paper_a_title=paper_a.get("title", "Paper A"),
            paper_b_title=paper_b.get("title", "Paper B"),
            position_a=json.dumps(position_a, indent=2),
            position_b=json.dumps(position_b, indent=2)
        )
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.content[0].text
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            result["paper_a_id"] = paper_a.get("id")
            result["paper_b_id"] = paper_b.get("id")
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting contradiction: {e}")
            return {
                "same_topic": False,
                "contradicts": False,
                "contradiction_type": "none",
                "explanation": f"Error: {e}",
                "error": str(e)
            }


class SynthesisGenerator:
    """Generates synthesis from thesis and antithesis positions."""
    
    SYNTHESIS_PROMPT = """You are a medical research synthesizer. Given contradicting positions in spine surgery literature, generate a comprehensive synthesis.

TOPIC: {topic}

THESIS (supported by {thesis_count} papers, avg evidence strength: {thesis_strength}):
{thesis}

ANTITHESIS (supported by {antithesis_count} papers, avg evidence strength: {antithesis_strength}):
{antithesis}

Evidence Hierarchy Applied:
- RCT/Meta-analysis weight: 1.0
- Cohort study weight: 0.75
- Case-control weight: 0.5
- Case series/report weight: 0.25

Generate a balanced synthesis that:
1. Acknowledges both positions
2. Weighs evidence based on study quality
3. Identifies conditions where each position may be valid
4. Suggests how to reconcile the contradiction
5. Recommends areas for future research

Respond in JSON:
{{
    "synthesis_statement": "A comprehensive statement reconciling both views...",
    "key_insights": ["...", "..."],
    "conditions_for_thesis": ["When thesis applies..."],
    "conditions_for_antithesis": ["When antithesis applies..."],
    "reconciliation": "How both can be true...",
    "research_gaps": ["Areas needing more research..."],
    "clinical_recommendation": "Pragmatic guidance for clinicians...",
    "confidence": 0.0
}}"""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
    
    def generate_synthesis(
        self,
        topic: str,
        thesis_papers: List[Dict[str, Any]],
        thesis_positions: List[Dict[str, Any]],
        antithesis_papers: List[Dict[str, Any]],
        antithesis_positions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a synthesis from contradicting positions."""
        
        # Calculate average evidence strength
        thesis_strength = np.mean([
            p.get("evidence_strength", 0.5) for p in thesis_positions
        ]) if thesis_positions else 0.5
        
        antithesis_strength = np.mean([
            p.get("evidence_strength", 0.5) for p in antithesis_positions
        ]) if antithesis_positions else 0.5
        
        # Format thesis and antithesis
        thesis_text = "\n".join([
            f"- {p.get('main_claim', 'Unknown')} (Evidence: {p.get('evidence_type', 'unknown')})"
            for p in thesis_positions
        ])
        
        antithesis_text = "\n".join([
            f"- {p.get('main_claim', 'Unknown')} (Evidence: {p.get('evidence_type', 'unknown')})"
            for p in antithesis_positions
        ])
        
        prompt = self.SYNTHESIS_PROMPT.format(
            topic=topic,
            thesis=thesis_text or "No specific thesis extracted",
            thesis_count=len(thesis_papers),
            thesis_strength=round(thesis_strength, 2),
            antithesis=antithesis_text or "No specific antithesis extracted",
            antithesis_count=len(antithesis_papers),
            antithesis_strength=round(antithesis_strength, 2)
        )
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.content[0].text
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            return json.loads(content.strip())
            
        except Exception as e:
            logger.error(f"Error generating synthesis: {e}")
            return {
                "synthesis_statement": f"Unable to generate synthesis: {e}",
                "key_insights": [],
                "conditions_for_thesis": [],
                "conditions_for_antithesis": [],
                "reconciliation": "",
                "research_gaps": [],
                "clinical_recommendation": "",
                "confidence": 0.0,
                "error": str(e)
            }


class DialecticEngine:
    """
    Main engine for dialectical analysis of literature.
    
    Workflow:
    1. Cluster papers by semantic similarity
    2. Extract positions from each cluster
    3. Detect contradictions between clusters
    4. Generate synthesis weighing evidence
    """
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        self.clusterer = EmbeddingClusterer()
        self.extractor = PositionExtractor(model)
        self.detector = ContradictionDetector(model)
        self.synthesizer = SynthesisGenerator(model)
    
    def analyze(
        self,
        topic: str,
        papers: List[Dict[str, Any]],
        min_papers_per_cluster: int = 1
    ) -> DialecticalAnalysis:
        """
        Perform full dialectical analysis on a set of papers.
        
        Args:
            topic: The research topic/question
            papers: List of paper dictionaries
            min_papers_per_cluster: Minimum papers needed per cluster
        
        Returns:
            DialecticalAnalysis object with thesis, antithesis, and synthesis
        """
        if not papers:
            return self._empty_analysis(topic)
        
        # Step 1: Cluster papers
        logger.info(f"Clustering {len(papers)} papers...")
        clusters = self.clusterer.cluster_papers(papers, n_clusters=2)
        
        if len(clusters) < 2:
            logger.warning("Could not form two clusters, using single-cluster analysis")
            return self._single_cluster_analysis(topic, papers)
        
        # Assign clusters to thesis and antithesis (larger cluster = thesis)
        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
        thesis_papers = sorted_clusters[0][1]
        antithesis_papers = sorted_clusters[1][1]
        
        # Step 2: Extract positions
        logger.info("Extracting positions from thesis cluster...")
        thesis_positions = self.extractor.extract_positions_batch(thesis_papers)
        
        logger.info("Extracting positions from antithesis cluster...")
        antithesis_positions = self.extractor.extract_positions_batch(antithesis_papers)
        
        # Step 3: Generate synthesis
        logger.info("Generating synthesis...")
        synthesis_result = self.synthesizer.generate_synthesis(
            topic=topic,
            thesis_papers=thesis_papers,
            thesis_positions=thesis_positions,
            antithesis_papers=antithesis_papers,
            antithesis_positions=antithesis_positions
        )
        
        # Build result
        thesis = Position(
            text=self._summarize_positions(thesis_positions),
            position_type=PositionType.THESIS,
            supporting_papers=[p.get("id", "") for p in thesis_papers],
            evidence_strength=float(np.mean([p.get("evidence_strength", 0.5) for p in thesis_positions])),
            confidence=0.8
        )
        
        antithesis = Position(
            text=self._summarize_positions(antithesis_positions),
            position_type=PositionType.ANTITHESIS,
            supporting_papers=[p.get("id", "") for p in antithesis_papers],
            evidence_strength=float(np.mean([p.get("evidence_strength", 0.5) for p in antithesis_positions])),
            confidence=0.8
        )
        
        synthesis = Position(
            text=synthesis_result.get("synthesis_statement", ""),
            position_type=PositionType.SYNTHESIS,
            supporting_papers=[p.get("id", "") for p in papers],
            evidence_strength=float(synthesis_result.get("confidence", 0.5)),
            confidence=float(synthesis_result.get("confidence", 0.5))
        )
        
        return DialecticalAnalysis(
            topic=topic,
            thesis=thesis,
            antithesis=antithesis,
            synthesis=synthesis,
            paper_clusters={
                "thesis": [p.get("id", "") for p in thesis_papers],
                "antithesis": [p.get("id", "") for p in antithesis_papers]
            },
            metadata={
                "thesis_positions": thesis_positions,
                "antithesis_positions": antithesis_positions,
                "synthesis_details": synthesis_result,
                "total_papers": len(papers)
            }
        )
    
    def _summarize_positions(self, positions: List[Dict[str, Any]]) -> str:
        """Summarize multiple positions into a single statement."""
        claims = [p.get("main_claim", "") for p in positions if p.get("main_claim")]
        if not claims:
            return "No position extracted"
        return "; ".join(claims[:3])  # Limit to first 3 claims
    
    def _empty_analysis(self, topic: str) -> DialecticalAnalysis:
        """Return empty analysis when no papers provided."""
        empty_position = Position(
            text="No papers available for analysis",
            position_type=PositionType.THESIS,
            supporting_papers=[],
            evidence_strength=0.0,
            confidence=0.0
        )
        return DialecticalAnalysis(
            topic=topic,
            thesis=empty_position,
            antithesis=Position(**{**asdict(empty_position), "position_type": PositionType.ANTITHESIS}),
            synthesis=Position(**{**asdict(empty_position), "position_type": PositionType.SYNTHESIS}),
            metadata={"error": "No papers provided"}
        )
    
    def _single_cluster_analysis(
        self,
        topic: str,
        papers: List[Dict[str, Any]]
    ) -> DialecticalAnalysis:
        """Handle case where only one cluster is formed."""
        positions = self.extractor.extract_positions_batch(papers)
        
        thesis = Position(
            text=self._summarize_positions(positions),
            position_type=PositionType.THESIS,
            supporting_papers=[p.get("id", "") for p in papers],
            evidence_strength=float(np.mean([p.get("evidence_strength", 0.5) for p in positions])),
            confidence=0.7
        )
        
        return DialecticalAnalysis(
            topic=topic,
            thesis=thesis,
            antithesis=Position(
                text="No contradicting position identified",
                position_type=PositionType.ANTITHESIS,
                supporting_papers=[],
                evidence_strength=0.0,
                confidence=0.0
            ),
            synthesis=Position(
                text=f"Literature shows consensus: {thesis.text}",
                position_type=PositionType.SYNTHESIS,
                supporting_papers=[p.get("id", "") for p in papers],
                evidence_strength=thesis.evidence_strength,
                confidence=0.7
            ),
            paper_clusters={"thesis": [p.get("id", "") for p in papers]},
            metadata={"positions": positions, "single_cluster": True}
        )


# Convenience function
async def analyze_dialectically(
    topic: str,
    papers: List[Dict[str, Any]],
    model: str = "claude-3-5-sonnet-20241022"
) -> Dict[str, Any]:
    """
    Convenience function for dialectical analysis.
    
    Example:
        result = await analyze_dialectically(
            "PELD recurrence rates",
            papers=[paper1_dict, paper2_dict, ...]
        )
    """
    engine = DialecticEngine(model=model)
    analysis = engine.analyze(topic, papers)
    return analysis.to_dict()
