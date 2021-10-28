"""
Multi-threaded Web Crawler
Concurrency using futures, also BFS
"""
# Solution: https://leetcode.com/problems/web-crawler-multithreaded/discuss/739683/Concise-and-Beautiful-Python

# """
# This is HtmlParser's API interface.
# You should not implement it, or speculate about its implementation
# """
#class HtmlParser(object):
#    def getUrls(self, url):
#        """
#        :type url: str
#        :rtype List[str]
#        """

from concurrent import futures
from collections import deque
from typing import List
"""
We implement a classic BFS but the entries in our queue are future objects instead of primitive values. A pool of at most
max_workers threads is used to execute getUrl calls asynchronously. Calling result() on our futures blocks until the
task is completed or rejected.
"""
class Solution:
    def crawl(self, startUrl: str, htmlParser: 'HtmlParser') -> List[str]:
        hostname = lambda url: url.split("/")[2]
        # Set up BFS: visited set
        visited = {startUrl}

        with futures.ThreadPoolExecutor(max_workers=16) as executor:
            # Initialize BFS queue (inside futures condition)
            tasks = deque([executor.submit(htmlParser.getUrls, startUrl)])
            # While BFS queue is not empty
            while tasks:
                for url in tasks.popleft().result():
                    if url not in visited and hostname(startUrl) == hostname(url):
                        visited.add(url)
                        tasks.append(executor.submit(htmlParser.getUrls, url))
            return list(visited)