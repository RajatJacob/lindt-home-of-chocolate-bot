import numpy as np
from redis_client import redis_cache
import requests
from pydantic import BaseModel
from urllib.parse import urlparse, parse_qs, urlunparse, urlencode
from bs4 import BeautifulSoup
import logging
from langchain.text_splitter import HTMLHeaderTextSplitter
import chromadb
from chromadb.utils import embedding_functions

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection(
    'bot', embedding_function=sentence_transformer_ef)


SEARCH_URL = "https://en.wikipedia.org/w/index.php?title=Special:Search&profile=advanced&fulltext=1&ns0=1"


class SearchResult(BaseModel):
    title: str
    url: str
    summary: str


def get_search_url(query: str):
    url = urlparse(SEARCH_URL)
    query_params = parse_qs(url.query)
    query_params['search'] = [query]
    modified_url = urlunparse(url._replace(
        query=urlencode(query_params, doseq=True)))

    return modified_url


@redis_cache
def requests_get(*args, **kwargs):
    response = requests.get(*args, **kwargs)
    return response.text


def get_search_results(query: str):
    logging.info(f'Fetching content from "{query}"')
    url = get_search_url(query)
    soup = BeautifulSoup(requests_get(url), 'html.parser')
    elements = soup.select('ul.mw-search-results li td')
    for element in elements:
        link = element.find('a')
        if not link:
            continue
        title = link.get('title')
        url = "https://en.wikipedia.org" + link.get('href')
        summary = '\n'.join([
            res.text for res in element.select('.searchresult')
        ])
        yield SearchResult(title=title, url=url, summary=summary)


def get_closest_match(query: str):
    results = list(get_search_results(query))
    for result in results:
        if result.title.lower() == query.lower():
            return result
    return results[0]


@redis_cache
def get_page_content(url: str):
    logging.info(f'Fetching content from "{url}"')
    soup = BeautifulSoup(requests_get(url), 'html.parser')
    return str('\n'.join(
        filter(bool, soup.select_one('#mw-content-text').text.split('\n'))
    ))


def get_documents_from_page_content(url: str):
    HEADERS_TO_SPLIT_ON = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ]
    logging.info(f'Tokenizing HTML for "{url}"')
    text_splitter = HTMLHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON)
    return text_splitter.split_text(str(requests_get(url)))


def search(query: str, should_return_documents=False):
    closest = get_closest_match(query)
    if should_return_documents:
        return get_documents_from_page_content(closest.url)
    return get_page_content(closest.url)


def search_and_vectorize(query: str):
    logging.info(f'Searching and vectorizing "{query}"')
    results = chroma_collection.query(query_texts=[query], n_results=10)
    distances = np.squeeze(results['distances'])
    if len(distances) > 0 and np.min(distances) < 1:
        return results
    logging.info(f'Getting text for "{query}"')
    text = search(query, should_return_documents=False)
    logging.info(f'Getting docs for "{query}"')
    docs = search(query, should_return_documents=True)
    ids = [f"{query}_text", *[f"{query}_{doc}" for doc in range(len(docs))]]
    logging.info(f'Adding "{query}" to the collection')
    chroma_collection.add(
        documents=[text, *[doc.page_content for doc in docs]],
        ids=ids
    )
    logging.info(f'Done adding "{query}"')
    results = chroma_collection.query(query_texts=[query], n_results=3)
    return results
