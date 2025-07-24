import aiohttp
import re
import asyncio


class WikipediaService:
    def __init__(self):
        self.base_url = "https://en.wikipedia.org/api/rest_v1/page/summary"
        self.search_url = "https://en.wikipedia.org/w/api.php"

    async def get_evidence(self, text: str) -> str:
        """Get relevant evidence from Wikipedia"""
        try:
            # Extract key medical terms
            medical_terms = self._extract_medical_terms(text)

            if not medical_terms:
                return ""

            # Search for the most relevant term
            main_term = medical_terms[0]
            summary = await self._get_page_summary(main_term)

            if summary:
                # Return first 200 characters
                return summary[:300] + "..." if len(summary) > 300 else summary

            return ""

        except Exception as e:
            print(f"Wikipedia service error: {e}")
            return ""

    def _extract_medical_terms(self, text: str) -> list:
        """Extract potential medical terms for search"""
        # Simple extraction - look for capitalized medical terms
        medical_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Capitalized terms
            r'\b(?:COVID-19|HIV|AIDS|DNA|RNA)\b',  # Common abbreviations
        ]

        terms = []
        for pattern in medical_patterns:
            matches = re.findall(pattern, text)
            terms.extend(matches)

        # Filter for likely medical terms
        medical_keywords = ['disease', 'syndrome', 'virus', 'bacteria', 'treatment']
        medical_terms = [term for term in terms if any(keyword in term.lower() for keyword in medical_keywords)]

        return medical_terms[:3]  # Return top 3

    async def _get_page_summary(self, term: str) -> str:
        """Get Wikipedia page summary"""
        try:
            url = f"{self.base_url}/{term.replace(' ', '_')}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('extract', '')

        except Exception as e:
            print(f"Error fetching Wikipedia summary: {e}")

        return ""


# =============================================================================
# PUBMED SERVICE (services/pubmed_service.py)
# =============================================================================

import aiohttp
import xml.etree.ElementTree as ET


class PubMedService:
    def __init__(self):
        self.search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    async def get_evidence(self, text: str) -> str:
        """Get evidence from PubMed"""
        try:
            # Extract medical terms for search
            search_terms = self._extract_search_terms(text)

            if not search_terms:
                return ""

            # Search PubMed
            pmids = await self._search_pubmed(search_terms)

            if pmids:
                # Get summary for first result
                summary = await self._get_article_summary(pmids[0])
                return summary

            return ""

        except Exception as e:
            print(f"PubMed service error: {e}")
            return ""

    def _extract_search_terms(self, text: str) -> str:
        """Extract terms suitable for PubMed search"""
        # Simple approach - extract medical keywords
        medical_terms = re.findall(r'\b(?:cancer|diabetes|covid|vaccine|treatment|therapy|drug|disease)\b',
                                   text.lower())
        return " AND ".join(set(medical_terms[:3]))  # Max 3 terms

    async def _search_pubmed(self, search_terms: str) -> list:
        """Search PubMed for relevant articles"""
        try:
            params = {
                'db': 'pubmed',
                'term': search_terms,
                'retmax': '3',
                'retmode': 'xml'
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(self.search_url, params=params, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        root = ET.fromstring(content)
                        pmids = [id_elem.text for id_elem in root.findall('.//Id')]
                        return pmids

        except Exception as e:
            print(f"PubMed search error: {e}")

        return []

    async def _get_article_summary(self, pmid: str) -> str:
        """Get article summary"""
        try:
            params = {
                'db': 'pubmed',
                'id': pmid,
                'retmode': 'xml'
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(self.summary_url, params=params, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        # Simple extraction - in real implementation, parse XML properly
                        if 'abstract' in content.lower():
                            return f"PubMed article {pmid} found with relevant medical information."

        except Exception as e:
            print(f"PubMed summary error: {e}")

        return ""