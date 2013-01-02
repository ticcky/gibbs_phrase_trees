import sys
from collections import defaultdict
import pdbonerror
import operator
import random


class Vocabulary(dict):
    def __init__(self):
        super(Vocabulary, self).__init__()
        self.rev_map = {}

    def get_word_id(self, word):
        """Return words id."""
        if not word in self:
            self[word] = len(self)
            self.rev_map[self[word]] = word
        return self[word]

    def rev(self, wid):
        if wid == -1:
            return "ROOT"
        return self.rev_map[wid]


class Tree(object):
    edges = None
    children = None
    parents = None
    posidmap = None
    vocab = None

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        res = []
        for node in self.parents.keys():
            res += [self.vocab.rev(self.posidmap[node])]

        return " - ".join(res)


class TreeBank(object):
    def __init__(self):
        self.vocab = Vocabulary()
        self.bank = []

    def __len__(self):
        return len(self.bank)

    def __iter__(self):
        for tree in self.bank:
            yield tree

    def __getitem__(self, item):
        return self.bank[item]

    def load(self, fname):
        with open(fname, "r+b") as f_in:
            curr_edges = []
            curr_children = defaultdict(lambda: [])
            curr_parents = {}
            curr_posidmap = {0: -1}

            for ln in f_in:
                ln = ln.strip()

                if len(ln) == 0:
                    self.bank += [Tree(edges=curr_edges, children=curr_children, parents=curr_parents, posidmap=curr_posidmap, vocab=self.vocab)]
                    if len(self.bank) > 5000:
                        break
                    curr_edges = []
                    curr_parents = {}
                    curr_children = defaultdict(lambda: [])
                    curr_posidmap = {0: -1}
                else:
                    word_pos, word, parent_pos = ln.split("\t")
                    word_pos = int(word_pos)
                    parent_pos = int(parent_pos)
                    word_id = self.vocab.get_word_id(word)
                    curr_posidmap[word_pos] = word_id
                    if parent_pos != 0:
                        curr_edges += [(parent_pos, word_pos, )]
                    curr_children[parent_pos] += [word_pos]
                    curr_parents[word_pos] = parent_pos

def in_order(tree):
    if len(tree) == 1:
        yield tree[0]
    else:
        for node in tree[1][0]:
            for value in in_order(node): yield value
        yield tree[0]
        for node in tree[1][1]:
            for value in in_order(node): yield value


class GibbsTrees(object):
    def __init__(self, treebank):
        self.treebank = treebank
        self.tree_groups = [[0] + [0 for _ in tree.parents.keys()] for tree in self.treebank]
        self.counts = None
        self.counts_all = 0
        self.counts_sizes = None

    def __iter__(self):
        return iter(xrange(len(self.treebank)))

    def print_top(self, n=10):
        res = self.counts.items()
        res.sort(key=lambda x: -x[1])

        for tree, cnt in res[:n]:
            print cnt, " ".join([self.treebank.vocab.rev(x) for x in in_order(tree)])

    def randomize_groups(self):
        for tree_id, tree in enumerate(self.treebank):
            root_children = tree.children[0]
            groups = self.tree_groups[tree_id]
            for word in tree.parents.keys():
                if word in root_children or random.random() > 0.5:
                    groups = self.split_group(tree, groups, word)
            self.tree_groups[tree_id] = groups

    def count(self):
        self.counts = defaultdict(lambda: 0)
        self.counts_sizes = defaultdict(lambda: 0)
        for tree, groups in zip(self.treebank, self.tree_groups):
            self._update_count(tree, groups)

    def get_subtrees(self, tree, groups):
        group_sets = defaultdict(lambda: set())

        for node_pos in tree.parents.keys():
            group_sets[groups[node_pos]].add(node_pos)

        return group_sets

    def get_tree_size(self, tree):
        return len(tree)

    def update_tree_count(self, tree_id, how):
        self._update_count(self.treebank[tree_id], self.tree_groups[tree_id], how)

    def get_all_subtree_count(self):
        return self.counts_all

    def get_subtree_count(self, tree, subtree):
        treehash = self.hash_tree(tree, subtree)
        return self.counts[treehash]

    def get_subtree_len_count(self, tree, subtree):
        return self.counts[self.get_tree_size(subtree)]

    def _update_count(self, tree, groups, how=1):
        subtrees = self.get_subtrees(tree, groups)
        for subtree in subtrees.values():
            treehash = self.hash_tree(tree, subtree)
            self.counts[treehash] += how
            self.counts_all += how

            # self.counts_sizes[self.get_tree_size(subtree)] += how

            if how < 0 and self.counts[treehash] == 0:
                del self.counts[treehash]


    def merge_group(self, tree, groups, word_pos):
        group_id1 = groups[tree.parents[word_pos]]
        group_id2 = groups[word_pos]
        groups = [x if x != group_id2 else group_id1 for x in groups]

        return groups

    def split_group(self, tree, groups, word_pos):
        new_gid = max(groups) + 1

        queue = [word_pos]
        while len(queue) > 0:
            curr_word = queue.pop()
            groups[curr_word] = new_gid
            queue = [item for item in tree.children[curr_word]] + queue

        return groups

    def get_root(self, tree, subtree=None):
        if subtree is None:
            subtree = tree.parets.keys()

        hasparent = set()
        everyone = set()
        for node in subtree:
            if tree.parents[node] in subtree:
                hasparent.add(node)
            everyone.add(node)

        parent = everyone - hasparent
        if len(parent) > 1:
            parent = {0}

        return parent.pop()

    def hash_tree(self, tree, subtree=None):
        if subtree is None:
            subtree = tree.parents.keys()

        #hash = [item[] for item in subtree]
        root = self.get_root(tree, subtree)

        res = []
        queue = [root]

        def _mktree(node):
            node_id = tree.posidmap[node]
            if len(tree.children[node]) > 0:
                return (node_id, (tuple(_mktree(x) for x in tree.children[node] if x < node), tuple(_mktree(x) for x in tree.children[node] if x > node), ))
            else:
                return (node_id, )

        #while len(queue) > 0:
        #    curr = queue.pop()
        #    for child in tree.children[curr]:
        #        if child in subtree:
        #            queue = [child] + queue

        #   res += [tree.posidmap[curr]]

        res = _mktree(root)

        return res #tuple(res)


class GibbsSampling(object):
    P_TREE_GEN = 0.5
    P_TREE_STOP = 0.5

    ALFA = 1
    BETA = 1

    def __init__(self, gibbs_trees):
        self.gibbs_trees = gibbs_trees

    def p0trees(self, tree, subtrees):
        res = 1.0
        for subtree in subtrees.values():
            res *= self.p0tree(tree, subtree)

        return res

    def p0tree(self, tree, subtree=None):
        if subtree is None:
            subtree = tree.parents.keys()

        # get root
        root = self.gibbs_trees.get_root(tree, subtree)

        res = 1.0
        queue = [root]


        while len(queue) > 0:
            curr_node = queue.pop()
            #print 'doing', curr_node, 'with children', tree.children[curr_node], res,
            if len(tree.children[curr_node]) > 0:
                queue = tree.children[curr_node] + queue
                res *= self.P_TREE_GEN ** len(tree.children[curr_node])
            else:
                res *= self.P_TREE_STOP


        #res *= (1.0 / (len(self.gibbs_trees.treebank.vocab))) ** len(subtree)


        return res

    def ptree(self, tree, groups):
        curr_subtrees = self.gibbs_trees.get_subtrees(tree, groups)

        p_new = 1.0
        for subtree in curr_subtrees.values():
            p = self.ALFA * self.p0tree(tree, subtree) + self.gibbs_trees.get_subtree_count(tree, subtree)
            p /= self.ALFA + self.gibbs_trees.get_all_subtree_count()

            #p *= self.BETA * self.p0tree(tree, subtree) + self.gibbs_trees.get_subtree_len_count(tree, subtree)
            #p /= self.BETA + self.gibbs_trees.get_all_subtree_count()

            p_new *= p

        return p_new


    def do(self, iterations):
        for iter in range(iterations):
            print >> sys.stderr, "Doing %d. iteration" % iter
            self.do_iteration()

    def do_iteration(self):
        for tree_id in self.gibbs_trees:
            # discount current tree
            self.gibbs_trees.update_tree_count(tree_id, -1)  #tree, groups, -1)

            # compute probability of the current tree
            curr_tree, curr_groups = self.gibbs_trees.treebank[tree_id], self.gibbs_trees.tree_groups[tree_id]

            p_curr = self.ptree(curr_tree, curr_groups)

            # randomly select change one edge to change
            sel_node = random.randint(1, len(curr_tree.parents.keys()))

            # do we join/cut the connecting edge?
            sel_node_parent = curr_tree.parents[sel_node]
            if curr_groups[sel_node] == curr_groups[sel_node_parent]:
                new_groups = self.gibbs_trees.split_group(curr_tree, curr_groups, sel_node)
            else:
                new_groups = self.gibbs_trees.merge_group(curr_tree, curr_groups, sel_node)

            p_new = self.ptree(curr_tree, new_groups)


            if p_new + p_curr > 0.0:
                boundary = p_new / (p_new + p_curr)
            else:
                boundary = 0.5

            if random.random() < boundary:
                self.gibbs_trees.tree_groups[tree_id] = new_groups
                #print 'doing', boundary
            else:
                #print 'not doing', boundary
                pass

            #print boundary

            # add new tree
            self.gibbs_trees.update_tree_count(tree_id, 1)  #, groups, 1)

            if tree_id % 1000 == 0:
                print "tree count", len(self.gibbs_trees.counts), "tree id", tree_id

            #if tree_id % 1000 == 0:
            #    self.gibbs_trees.print_top(100)




def main(fname):
    tb = TreeBank()
    tb.load(fname)

    gt = GibbsTrees(tb)
    gt.randomize_groups()
    gt.count()
    """gt.count()
    print gt.counts
    print gt.tree_groups[0]
    gt.split_group(0, 10)
    print gt.counts
    #print gt.tree_groups[0]
    gt.merge_group(0, 10)
    print gt.counts
    print gt.tree_groups[0]
    #print gt.treebank[0]
    """

    gs = GibbsSampling(gt)
    gs.do(10)

    gt.print_top(100)
    #import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main(sys.argv[1])