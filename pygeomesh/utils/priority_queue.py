import heapq as hq
from typing import Any
from dataclasses import dataclass, field

@dataclass
class PriorityItem:
    x : Any = field(compare=False)
    priority: float

    def __lt__(self, other):
        return self.priority < other.priority

class PriorityQueue:
    def __init__(self):
        self.data = []

    @property
    def front(self) -> PriorityItem:
        return self.data[0]

    def empty(self) -> bool:
        return len(self.data) == 0

    def get(self) -> PriorityItem:
        return hq.heappop(self.data)

    def pop(self) -> PriorityItem:
        return self.get()
    
    def push(self, x, w) -> None:
        item = PriorityItem(x, w)
        hq.heappush(self.data, item)
