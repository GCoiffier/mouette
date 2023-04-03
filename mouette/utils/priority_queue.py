import heapq as hq
from typing import Any
from dataclasses import dataclass, field

@dataclass
class PriorityItem:
    """
    Dataclass representing an item inside the priority queue.
    An item is a pair element/priority.
    """
    x : Any = field(compare=False)
    priority: float

    def __lt__(self, other):
        return self.priority < other.priority

class PriorityQueue:
    """
    Implementation of a priority queue using a heap from the heapq package.

    Attributes:
        data (list): the list representing the queue
    """

    def __init__(self):
        """
        Initializes an empty queue.
        """
        self.data = []

    def empty(self) -> bool:
        """
        Tests if the queue contains any element.

        Returns:
            bool: True if the queue is empty
        """
        return len(self.data) == 0

    @property
    def front(self) -> PriorityItem:
        """
        The first element of the queue. Is accessed but not removed from the queue

        Returns:
            PriorityItem: The first element of the queue
        """
        return self.data[0]

    def get(self) -> PriorityItem:
        """
        Returns the first element of the queue. The element is removed from the queue.

        Returns:
            PriorityItem: The first element of the queue
        
        Raises:
            IndexError: if the queue is empty
        """
        return hq.heappop(self.data)

    def pop(self) -> PriorityItem:
        """
        Returns the first element of the queue. Same method as `get`.

        Returns:
            PriorityItem: The first element of the queue

        Raises:
            IndexError: if the queue is empty
        """
        return self.get()
    
    def push(self, x, w : float) -> None:
        """
        Inserts an element inside the queue.

        Args:
            x: Element to insert in the queue
            w: Priority of the element.
        """
        item = PriorityItem(x, w)
        hq.heappush(self.data, item)
