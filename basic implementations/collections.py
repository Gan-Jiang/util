import queue
import collections

#Queue O(1) insert, O(1) remove, FIFO.
data = queue.Queue()
data.put(3)
data.get()
data.qsize()
data.empty()


#stack, O(1) insert, O(1) remove, LILO. Can just use list.


#PriorityQueue, O(1) find min/max, O(logn) insert, delete/extract min/max, lookup/delete O(n)
#lowest valued entries are retrieved first.
q = queue.PriorityQueue()
q.put((100, 'a not agent task'))
q.put((5, 'a highly agent task'))
q.put((10, 'an important task'))
q.get()

#Deque, O(1) insert and remove
data = collections.deque()
#append, appendleft, pop, popleft, clear, count(x), support indexing, extendleft, remove(value), reverse, rotate(n): rotate n step.

#collections.defaultdict(int)

#collections.OrderedDict(), Ordered dictionaries are just like regular dictionaries but they remember the order that items were inserted.
# When iterating over an ordered dictionary, the items are returned in the order their keys were first added.
#a = collections.OrderedDict()
#a.popitem(last = True), last = True: LILO, otherwise FIFO.

#Counter, Counter([iterable-or-mapping]), can counter list or string, dict, etc.
#Setting a count to zero does not remove an element from a counter. Use del to remove it entirely
counter = collections.Counter()
#c = Counter(a=4, b=2, c=0, d=-2)
#c.elements() -> ['b', 'b', 'a', 'a', 'a', 'a']
#Counter('abracadabra').most_common(3), return most common 3 items with its count.
#c.substract(d)
#c & d, c | d, get the min or max.
#c.clear()
#c += Counter(), remove all 0 and negative counters.