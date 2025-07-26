import multiprocessing as mp
from itertools import chain
from collections import Counter


class CUSTOMGSP:

    def __init__(self, raw_transactions):
        self.freq_patterns = []
        self._pre_processing(raw_transactions)

    def _pre_processing(self, raw_transactions):
        """ Prepare the data by setting up transactions and unique candidates. """
        self.max_size = max(len(item) for item in raw_transactions)
        self.transactions = [tuple(item) for item in raw_transactions]
        counts = Counter(chain.from_iterable(raw_transactions))
        self.unique_candidates = [tuple([k]) for k, c in counts.items()]

    def is_subsequence(self, s, l):
        """ Check if a subsequence s exists in list l, order preserved but not contiguous. """
        it = iter(l)  
        return all(item in it for item in s)  

    def _calc_frequency(self, results, item, minsup):
        """ Calculate the frequency of an item in transactions. """
        frequency = sum(1 for t in self.transactions if self.is_subsequence(item, t))
        if frequency >= minsup:
            results[item] = frequency

    def _support(self, items, minsup):
        """ Calculate support for a set of items using multiprocessing. """
        results = mp.Manager().dict()
        pool = mp.Pool(processes=mp.cpu_count())

        for item in items:
            pool.apply_async(self._calc_frequency, args=(results, item, minsup))
        pool.close()
        pool.join()

        return dict(results)

    def _generate_candidates(self, prev_patterns, k):
        """ Generate new candidate sequences by combining frequent patterns from the previous iteration. """
        prev_patterns = list(prev_patterns)
        candidates = set()
        for i in range(len(prev_patterns)):
            for j in range(len(prev_patterns)):
                # Combine patterns that share all but the last item
                if prev_patterns[i][1:] == prev_patterns[j][:-1]:
                    candidates.add(prev_patterns[i] + (prev_patterns[j][-1],))
        return list(candidates)

    def search(self, minsup):
        """ Run the GSP algorithm to find frequent sequential patterns. """
        assert minsup > 0

        # Initialize with unique 1-sequences
        candidates = self.unique_candidates

        # Calculate initial support and filter for the 1-sequences
        self.freq_patterns.append(self._support(candidates, minsup))

        k_items = 1  # Start with 1-sequences

        # Iteratively generate candidates and filter by support
        while len(self.freq_patterns[k_items - 1]) > 0 and (k_items + 1 <= self.max_size):
            
            k_items += 1  # Increment sequence length
            # Generate candidate sequences for the next length
            items = list(self.freq_patterns[k_items - 2].keys())
            candidates = self._generate_candidates(items, k_items)

            # Calculate support and filter
            self.freq_patterns.append(self._support(candidates, minsup))


        return self.freq_patterns[:-1]
