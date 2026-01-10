"""
Tests for Citation Analyzer Module
"""

import pytest
from datetime import datetime
from backend.api.citation_analyzer import (
    CitationAnalyzer,
    EvidenceClassifier,
    analyze_papers
)


class TestCitationAnalyzer:
    """Test suite for CitationAnalyzer."""
    
    def setup_method(self):
        self.analyzer = CitationAnalyzer()
    
    def test_journal_weight_q1(self):
        """Test Q1 journal weight."""
        weight = self.analyzer.get_journal_weight("Spine")
        assert weight == 1.0
    
    def test_journal_weight_unknown(self):
        """Test unknown journal weight."""
        weight = self.analyzer.get_journal_weight("Unknown Journal XYZ")
        assert weight == 0.25
    
    def test_years_since_publication(self):
        """Test years calculation."""
        pub_date = datetime(2024, 1, 1)
        ref_date = datetime(2026, 1, 10)
        years = self.analyzer.calculate_years_since_publication(pub_date, ref_date)
        assert 1.9 < years < 2.1
    
    def test_importance_score_calculation(self):
        """Test importance score formula."""
        result = self.analyzer.calculate_importance_score(
            citations=50,
            pub_date=datetime(2024, 1, 1),
            journal="Spine"
        )
        
        assert "importance_score" in result
        assert result["importance_score"] > 0
        assert result["journal_weight"] == 1.0
        assert result["raw_citations"] == 50
    
    def test_analyze_paper_dict(self):
        """Test paper dictionary analysis."""
        paper = {
            "id": "test_1",
            "title": "Test Paper",
            "citations": 100,
            "pub_date": "2023-06-15",
            "journal": "European Spine Journal"
        }
        
        result = self.analyzer.analyze_paper(paper)
        
        assert "importance_metrics" in result
        assert "importance_score" in result
        assert result["importance_score"] > 0
    
    def test_rank_papers(self):
        """Test paper ranking."""
        papers = [
            {"citations": 10, "pub_date": "2020-01-01", "journal": "Unknown"},
            {"citations": 100, "pub_date": "2024-01-01", "journal": "Spine"},
            {"citations": 50, "pub_date": "2022-01-01", "journal": "World Neurosurgery"}
        ]
        
        ranked = self.analyzer.rank_papers(papers)
        
        assert len(ranked) == 3
        # Highest score should be first
        assert ranked[0]["importance_score"] >= ranked[1]["importance_score"]


class TestEvidenceClassifier:
    """Test suite for EvidenceClassifier."""
    
    def setup_method(self):
        self.classifier = EvidenceClassifier()
    
    def test_classify_rct(self):
        """Test RCT classification."""
        result = self.classifier.classify_study_type(
            title="Randomized Controlled Trial of PELD vs Open Surgery"
        )
        assert result["study_type"] == "randomized controlled trial"
        assert result["evidence_weight"] == 1.0
    
    def test_classify_meta_analysis(self):
        """Test meta-analysis classification."""
        result = self.classifier.classify_study_type(
            abstract="This meta-analysis of 20 studies shows..."
        )
        assert result["study_type"] == "meta-analysis"
        assert result["evidence_weight"] == 1.0
    
    def test_classify_retrospective(self):
        """Test retrospective study classification."""
        result = self.classifier.classify_study_type(
            abstract="We performed a retrospective analysis of 150 patients"
        )
        assert "retrospective" in result["study_type"]
        assert 0.5 <= result["evidence_weight"] <= 0.7
    
    def test_classify_unknown(self):
        """Test unknown study type."""
        result = self.classifier.classify_study_type(
            title="Observations on Spine Surgery"
        )
        assert result["study_type"] == "unknown"
        assert result["evidence_weight"] == 0.25


def test_analyze_papers_convenience():
    """Test the convenience function."""
    papers = [
        {
            "id": "p1",
            "title": "Randomized trial of PELD",
            "abstract": "This randomized controlled trial...",
            "citations": 100,
            "pub_date": "2024-01-01",
            "journal": "Spine"
        }
    ]
    
    result = analyze_papers(papers)
    
    assert len(result) == 1
    assert "combined_score" in result[0]
    assert "importance_score" in result[0]
    assert "evidence_weight" in result[0]
