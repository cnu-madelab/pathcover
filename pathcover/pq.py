import heapq
import itertools

from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)

REMOVED = '<removed-task>'

class PriorityQueue:

    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.counter = itertools.count()     # unique sequence count

    def get_len(self):
        return len(self.entry_finder)

    def is_included(self, task):
        return task in self.entry_finder

    def get_priority(self, task):
        return self.entry_finder[task][0]

    def add_task(self, task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heapq.heappush(self.pq, entry)

    def remove_task(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = REMOVED

    def pop_task(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = heapq.heappop(self.pq)
            if task is not REMOVED:
                del self.entry_finder[task]
                return task, priority
        raise KeyError('pop from an empty priority queue')

def main():
    pq = PriorityQueue()

    pq.add_task(0, 1)
    pq.add_task(1, 0)

    x = pq.pop_task()
    y = pq.pop_task()

    print(x, y)

if __name__ == "__main__":
    main()
