import aiohttp
import asyncio
import logging
import nest_asyncio
from tqdm.notebook import tqdm
from time import sleep
from typing import List

nest_asyncio.apply()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CepAPI:
    def __init__(self, base_url: str, limit: int):
        self.base_url = base_url
        self.limit = limit
        self.rate_limit = {"max_retrys": 5, "sleep_requests": 30}

        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
        }
        self.data = []

    def _client_conn(self, limit: int) -> None:
        return aiohttp.TCPConnector(limit=self.limit, ttl_dns_cache=300)

    def _get_url(self, cep: str) -> str:
        if (
            cep is not None
            and cep != "99999999"
            and cep != "00000000"
            and cep != ""
        ):
            url = self.base_url % cep
            if url is not None:
                return url
        else:
            logger.error("Invalid zip code")

    def _create_url(self, cep_list: List) -> List:
        return [self._get_url(cep) for cep in cep_list]

    async def _requests(self, session, url: str, retry: str = 0) -> None:
        try:
            async with session.get(url, headers=self.headers) as resp:
                resp.raise_for_status()
                if resp.status == 200:
                    if (resp.status == 429) and (
                        retry < self.rate_limit["max_retrys"]
                    ):
                        logger.error(
                            f"{resp.status} Too many requests. "
                            + f"sleep for {self.rate_limit['sleep_requests']}s and retry."
                        )
                        sleep(self.rate_limit["sleep_requests"])
                        return await self._requests(
                            session, url, retry=retry + 1
                        )
                    response = await resp.json()
                    if response is not None:
                        self.data.append(response)
                    else:
                        logger.error(
                            "The get() function no return a response object"
                        )
                else:
                    logger.error("Status: {}".format(resp.status))
                    response = dict()
                    self.data.append(response)
                return self.data
        except aiohttp.ClientResponseError as e:
            logger.error(f"HTTP client reply error: {e}")

    async def _get_requests(self, urls: List) -> None:
        timeout = aiohttp.ClientTimeout(total=8 * 60)
        conn = self._client_conn(self.limit)
        try:
            async with aiohttp.ClientSession(
                timeout=timeout, connector=conn
            ) as session:
                urls = tqdm(urls)
                for url in urls:
                    if url is not None:
                        response = await self._requests(session, url, retry=0)
                        await asyncio.sleep(0.05)
                    else:
                        logger.error("Invalid URL")
                await conn.close()
                return response
        except asyncio.TimeoutError as e:
            logger.error(f"Timeout during request: {e}")
