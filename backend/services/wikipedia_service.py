import aiohttp
import re
import asyncio

class WikipediaService:
    def __init__(self):
        self.base_url = "https://en.wikipedia.org/api/rest_v1/page/summary"
        self.search_url = "https://en.wikipedia.org/w/api.php"

    async def get_evidence(self, text: str) -> str:
        try:
            medical_terms = self._extract_terms(text)
            if not medical_terms:
                return ""
            summary = await self._get_page_summary(medical_terms[0])
            return summary[:300] + "..." if summary else ""
        except Exception as e:
            print(f"Wikipedia service error: {e}")
            return ""

    def _extract_terms(self, text: str) -> list:
        patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            r'\b(?:COVID-19|HIV|DNA|RNA|AIDS)\b'
        ]
        terms = []
        for p in patterns:
            terms += re.findall(p, text)
        return list(set(terms))[:3]

    async def _get_page_summary(self, term: str) -> str:
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
