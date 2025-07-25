import aiohttp
import ssl
from aiohttp import TCPConnector
import xml.etree.ElementTree as ET
import re

class PubMedService:
    def __init__(self):
        self.search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    async def get_evidence(self, text: str) -> str:
        try:
            search_terms = self._extract_search_terms(text)
            if not search_terms:
                return ""

            pmids = await self._search_pubmed(search_terms)
            if pmids:
                summary = await self._get_article_summary(pmids[0])
                return summary

            return ""
        except Exception as e:
            print(f"PubMed service error: {e}")
            return ""

    def _extract_search_terms(self, text: str) -> str:
        medical_terms = re.findall(r'\b(?:cancer|diabetes|covid|vaccine|treatment|therapy|drug|disease)\b',
                                   text.lower())
        return " AND ".join(set(medical_terms[:3]))

    async def _search_pubmed(self, search_terms: str) -> list:
        try:
            params = {
                'db': 'pubmed',
                'term': search_terms,
                'retmax': '3',
                'retmode': 'xml'
            }

            sslcontext = ssl.create_default_context()
            sslcontext.check_hostname = False
            sslcontext.verify_mode = ssl.CERT_NONE

            connector = TCPConnector(ssl=sslcontext)

            async with aiohttp.ClientSession(connector=connector) as session:
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
        try:
            params = {
                'db': 'pubmed',
                'id': pmid,
                'retmode': 'xml'
            }

            sslcontext = ssl.create_default_context()
            sslcontext.check_hostname = False
            sslcontext.verify_mode = ssl.CERT_NONE

            connector = TCPConnector(ssl=sslcontext)

            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(self.summary_url, params=params, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        if 'abstract' in content.lower():
                            return f"PubMed article {pmid} found with relevant medical information."

        except Exception as e:
            print(f"PubMed summary error: {e}")

        return ""
