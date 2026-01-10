"""
Literature Fetcher Module

Integrates PubMed, Semantic Scholar, and arXiv APIs for comprehensive
literature discovery in spine surgery research.
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod

import httpx
import xmltodict
import arxiv
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """Unified paper representation across all sources."""
    id: str
    source: str  # 'pubmed', 'semantic_scholar', 'arxiv'
    title: str
    abstract: str
    authors: List[str]
    journal: Optional[str] = None
    pub_date: Optional[datetime] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    arxiv_id: Optional[str] = None
    citations: int = 0
    full_text_url: Optional[str] = None
    mesh_terms: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        if self.pub_date:
            d['pub_date'] = self.pub_date.isoformat()
        return d


class BaseFetcher(ABC):
    """Abstract base class for literature fetchers."""
    
    @abstractmethod
    async def search(self, query: str, max_results: int = 100) -> List[Paper]:
        """Search for papers matching the query."""
        pass
    
    @abstractmethod
    async def get_paper_details(self, paper_id: str) -> Optional[Paper]:
        """Get detailed information for a specific paper."""
        pass


class PubMedFetcher(BaseFetcher):
    """Fetcher for PubMed/NCBI E-utilities API."""
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self):
        self.api_key = os.getenv("NCBI_API_KEY", "")
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def search(self, query: str, max_results: int = 100) -> List[Paper]:
        """Search PubMed for papers matching query."""
        try:
            # Step 1: ESearch to get PMIDs
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "sort": "relevance"
            }
            if self.api_key:
                search_params["api_key"] = self.api_key
            
            search_resp = await self.client.get(
                f"{self.BASE_URL}/esearch.fcgi",
                params=search_params
            )
            search_resp.raise_for_status()
            search_data = search_resp.json()
            
            pmids = search_data.get("esearchresult", {}).get("idlist", [])
            if not pmids:
                return []
            
            # Step 2: EFetch to get paper details
            return await self._fetch_details(pmids)
            
        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return []
    
    async def _fetch_details(self, pmids: List[str]) -> List[Paper]:
        """Fetch detailed information for a list of PMIDs."""
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml"
        }
        if self.api_key:
            fetch_params["api_key"] = self.api_key
        
        fetch_resp = await self.client.get(
            f"{self.BASE_URL}/efetch.fcgi",
            params=fetch_params
        )
        fetch_resp.raise_for_status()
        
        data = xmltodict.parse(fetch_resp.text)
        articles = data.get("PubmedArticleSet", {}).get("PubmedArticle", [])
        
        if isinstance(articles, dict):
            articles = [articles]
        
        papers = []
        for article in articles:
            try:
                paper = self._parse_article(article)
                if paper:
                    papers.append(paper)
            except Exception as e:
                logger.warning(f"Error parsing article: {e}")
                continue
        
        return papers
    
    def _parse_article(self, article: Dict) -> Optional[Paper]:
        """Parse a PubMed article XML into a Paper object."""
        medline = article.get("MedlineCitation", {})
        article_data = medline.get("Article", {})
        
        # Get PMID
        pmid = medline.get("PMID", {})
        if isinstance(pmid, dict):
            pmid = pmid.get("#text", "")
        
        # Get title
        title = article_data.get("ArticleTitle", "")
        if isinstance(title, dict):
            title = title.get("#text", "")
        
        # Get abstract
        abstract_data = article_data.get("Abstract", {}).get("AbstractText", "")
        if isinstance(abstract_data, list):
            abstract = " ".join([
                a.get("#text", str(a)) if isinstance(a, dict) else str(a)
                for a in abstract_data
            ])
        elif isinstance(abstract_data, dict):
            abstract = abstract_data.get("#text", "")
        else:
            abstract = str(abstract_data) if abstract_data else ""
        
        # Get authors
        author_list = article_data.get("AuthorList", {}).get("Author", [])
        if isinstance(author_list, dict):
            author_list = [author_list]
        
        authors = []
        for author in author_list:
            if isinstance(author, dict):
                last = author.get("LastName", "")
                first = author.get("ForeName", "")
                if last or first:
                    authors.append(f"{last}, {first}".strip(", "))
        
        # Get journal
        journal = article_data.get("Journal", {}).get("Title", "")
        
        # Get publication date
        pub_date = None
        journal_issue = article_data.get("Journal", {}).get("JournalIssue", {})
        pub_date_data = journal_issue.get("PubDate", {})
        year = pub_date_data.get("Year")
        month = pub_date_data.get("Month", "1")
        day = pub_date_data.get("Day", "1")
        
        if year:
            try:
                # Convert month name to number if needed
                month_map = {
                    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
                    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
                    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
                }
                if isinstance(month, str) and month in month_map:
                    month = month_map[month]
                pub_date = datetime(int(year), int(month), int(day))
            except (ValueError, TypeError):
                pub_date = datetime(int(year), 1, 1)
        
        # Get MeSH terms
        mesh_list = medline.get("MeshHeadingList", {}).get("MeshHeading", [])
        if isinstance(mesh_list, dict):
            mesh_list = [mesh_list]
        
        mesh_terms = []
        for mesh in mesh_list:
            descriptor = mesh.get("DescriptorName", {})
            if isinstance(descriptor, dict):
                mesh_terms.append(descriptor.get("#text", ""))
            elif descriptor:
                mesh_terms.append(str(descriptor))
        
        # Get DOI
        doi = None
        article_ids = article.get("PubmedData", {}).get("ArticleIdList", {}).get("ArticleId", [])
        if isinstance(article_ids, dict):
            article_ids = [article_ids]
        for aid in article_ids:
            if isinstance(aid, dict) and aid.get("@IdType") == "doi":
                doi = aid.get("#text", "")
                break
        
        return Paper(
            id=f"pubmed_{pmid}",
            source="pubmed",
            title=title,
            abstract=abstract,
            authors=authors,
            journal=journal,
            pub_date=pub_date,
            doi=doi,
            pmid=pmid,
            mesh_terms=mesh_terms
        )
    
    async def get_paper_details(self, paper_id: str) -> Optional[Paper]:
        """Get details for a specific PMID."""
        pmid = paper_id.replace("pubmed_", "")
        papers = await self._fetch_details([pmid])
        return papers[0] if papers else None
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class SemanticScholarFetcher(BaseFetcher):
    """Fetcher for Semantic Scholar API."""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self):
        self.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        self.client = httpx.AsyncClient(timeout=30.0, headers=headers)
    
    async def search(self, query: str, max_results: int = 100) -> List[Paper]:
        """Search Semantic Scholar for papers."""
        try:
            params = {
                "query": query,
                "limit": min(max_results, 100),
                "fields": "paperId,title,abstract,authors,venue,year,citationCount,externalIds,openAccessPdf"
            }
            
            resp = await self.client.get(
                f"{self.BASE_URL}/paper/search",
                params=params
            )
            resp.raise_for_status()
            data = resp.json()
            
            papers = []
            for item in data.get("data", []):
                paper = self._parse_paper(item)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            logger.error(f"Semantic Scholar search error: {e}")
            return []
    
    def _parse_paper(self, item: Dict) -> Optional[Paper]:
        """Parse Semantic Scholar paper data."""
        paper_id = item.get("paperId", "")
        
        authors = [
            a.get("name", "") for a in item.get("authors", [])
        ]
        
        external_ids = item.get("externalIds", {}) or {}
        
        year = item.get("year")
        pub_date = datetime(year, 1, 1) if year else None
        
        pdf_info = item.get("openAccessPdf") or {}
        
        return Paper(
            id=f"s2_{paper_id}",
            source="semantic_scholar",
            title=item.get("title", ""),
            abstract=item.get("abstract", "") or "",
            authors=authors,
            journal=item.get("venue", ""),
            pub_date=pub_date,
            doi=external_ids.get("DOI"),
            pmid=external_ids.get("PubMed"),
            arxiv_id=external_ids.get("ArXiv"),
            citations=item.get("citationCount", 0) or 0,
            full_text_url=pdf_info.get("url")
        )
    
    async def get_paper_details(self, paper_id: str) -> Optional[Paper]:
        """Get details for a specific Semantic Scholar paper ID."""
        clean_id = paper_id.replace("s2_", "")
        try:
            resp = await self.client.get(
                f"{self.BASE_URL}/paper/{clean_id}",
                params={
                    "fields": "paperId,title,abstract,authors,venue,year,citationCount,externalIds,openAccessPdf,references,citations"
                }
            )
            resp.raise_for_status()
            return self._parse_paper(resp.json())
        except Exception as e:
            logger.error(f"Error fetching paper {paper_id}: {e}")
            return None
    
    async def get_citations(self, paper_id: str, limit: int = 100) -> List[Paper]:
        """Get papers that cite the given paper."""
        clean_id = paper_id.replace("s2_", "")
        try:
            resp = await self.client.get(
                f"{self.BASE_URL}/paper/{clean_id}/citations",
                params={
                    "limit": limit,
                    "fields": "paperId,title,abstract,authors,venue,year,citationCount"
                }
            )
            resp.raise_for_status()
            data = resp.json()
            
            papers = []
            for item in data.get("data", []):
                citing_paper = item.get("citingPaper", {})
                paper = self._parse_paper(citing_paper)
                if paper:
                    papers.append(paper)
            
            return papers
        except Exception as e:
            logger.error(f"Error fetching citations for {paper_id}: {e}")
            return []
    
    async def get_references(self, paper_id: str, limit: int = 100) -> List[Paper]:
        """Get papers referenced by the given paper."""
        clean_id = paper_id.replace("s2_", "")
        try:
            resp = await self.client.get(
                f"{self.BASE_URL}/paper/{clean_id}/references",
                params={
                    "limit": limit,
                    "fields": "paperId,title,abstract,authors,venue,year,citationCount"
                }
            )
            resp.raise_for_status()
            data = resp.json()
            
            papers = []
            for item in data.get("data", []):
                cited_paper = item.get("citedPaper", {})
                if cited_paper:
                    paper = self._parse_paper(cited_paper)
                    if paper:
                        papers.append(paper)
            
            return papers
        except Exception as e:
            logger.error(f"Error fetching references for {paper_id}: {e}")
            return []
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class ArxivFetcher(BaseFetcher):
    """Fetcher for arXiv papers using the arxiv library."""
    
    def __init__(self):
        self.client = arxiv.Client()
    
    async def search(self, query: str, max_results: int = 100) -> List[Paper]:
        """Search arXiv for papers."""
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            for result in self.client.results(search):
                paper = Paper(
                    id=f"arxiv_{result.entry_id.split('/')[-1]}",
                    source="arxiv",
                    title=result.title,
                    abstract=result.summary,
                    authors=[a.name for a in result.authors],
                    journal=result.journal_ref or "arXiv",
                    pub_date=result.published,
                    doi=result.doi,
                    arxiv_id=result.entry_id.split('/')[-1],
                    full_text_url=result.pdf_url,
                    keywords=[c.term for c in result.categories] if hasattr(result, 'categories') else []
                )
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            logger.error(f"arXiv search error: {e}")
            return []
    
    async def get_paper_details(self, paper_id: str) -> Optional[Paper]:
        """Get details for a specific arXiv paper."""
        clean_id = paper_id.replace("arxiv_", "")
        papers = await self.search(f"id:{clean_id}", max_results=1)
        return papers[0] if papers else None
    
    async def close(self):
        """No cleanup needed for arxiv library."""
        pass


class LiteratureAggregator:
    """Aggregates results from multiple literature sources."""
    
    def __init__(self):
        self.pubmed = PubMedFetcher()
        self.semantic_scholar = SemanticScholarFetcher()
        self.arxiv = ArxivFetcher()
    
    async def search_all(
        self,
        query: str,
        max_results_per_source: int = 50,
        sources: Optional[List[str]] = None
    ) -> List[Paper]:
        """
        Search all sources and aggregate results.
        
        Args:
            query: Search query string
            max_results_per_source: Maximum results from each source
            sources: List of sources to search ('pubmed', 'semantic_scholar', 'arxiv')
                    If None, searches all sources.
        
        Returns:
            List of Paper objects from all sources
        """
        if sources is None:
            sources = ["pubmed", "semantic_scholar", "arxiv"]
        
        tasks = []
        if "pubmed" in sources:
            tasks.append(self.pubmed.search(query, max_results_per_source))
        if "semantic_scholar" in sources:
            tasks.append(self.semantic_scholar.search(query, max_results_per_source))
        if "arxiv" in sources:
            tasks.append(self.arxiv.search(query, max_results_per_source))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_papers = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Search error: {result}")
                continue
            all_papers.extend(result)
        
        # Deduplicate by DOI where possible
        seen_dois = set()
        unique_papers = []
        for paper in all_papers:
            if paper.doi:
                if paper.doi not in seen_dois:
                    seen_dois.add(paper.doi)
                    unique_papers.append(paper)
            else:
                unique_papers.append(paper)
        
        return unique_papers
    
    async def close(self):
        """Close all fetcher clients."""
        await self.pubmed.close()
        await self.semantic_scholar.close()
        await self.arxiv.close()


# Convenience function for quick searches
async def fetch_papers(
    query: str,
    max_results: int = 100,
    sources: Optional[List[str]] = None
) -> List[Paper]:
    """
    Convenience function to fetch papers from all sources.
    
    Example:
        papers = await fetch_papers("PELD annular closure devices", max_results=50)
    """
    aggregator = LiteratureAggregator()
    try:
        return await aggregator.search_all(query, max_results // 3, sources)
    finally:
        await aggregator.close()
