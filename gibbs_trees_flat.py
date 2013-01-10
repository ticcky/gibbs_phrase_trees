#/usr/bin/env python

"""Find common phrases in a treebank by Gibbs sampling."""

__author__ = "Lukas Zilka"
__email__ = "lukas@zilka.me"
__version__ = "1.0"
__license__ = "Apache 2.0"

import sys
from collections import defaultdict
import pdbonerror
import random


class Vocabulary(dict):
    """Translates words from/to number ids"""

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
        """Return word for given word id."""
        if wid == -1:
            return "ROOT"
        return self.rev_map[wid]


class Tree(object):
    """Tree for sentence.
    Essentially just a named tuple that holds the information about
    the tree together."""

    edges = None
    children = None
    parents = None
    posidmap = None
    vocab = None

    def __init__(self, **kwargs):
        """Set self properties according to the passed keyword args."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        res = []
        for node in self.parents.keys():
            res += [self.vocab.rev(self.posidmap[node])]

        return " - ".join(res)


class TreeBank(object):
    """Loads and stores corpus of trees."""
    def __init__(self):
        self.vocab = Vocabulary()
        self.bank = []

        # n-grams for estimating word probabilities
        self.ngrams = defaultdict(lambda: 0)
        self.total_words = 0

    def __len__(self):
        """Count trees in the treebank."""
        return len(self.bank)

    def __iter__(self):
        """Iterate over the trees."""
        for tree in self.bank:
            yield tree

    def __getitem__(self, item):
        """Get the tree at given index."""
        return self.bank[item]

    def load(self, fname, limit=None):
        """Load trees from the given file."""
        with open(fname, "r+b") as f_in:
            # initialize new tree
            curr_edges = []
            curr_children = defaultdict(lambda: [])
            curr_parents = {}
            curr_posidmap = {0: -1}  # maps: word_position -> word_id
            last_word_id = None
            self.ngrams[(-1, )] += 1  # add count of root

            for ln in f_in:
                ln = ln.strip()

                # if it is the separating line, store the current tree and
                # start a new one
                if len(ln) == 0:
                    self.bank += [
                        Tree(edges=curr_edges,
                             children=curr_children,
                             parents=curr_parents,
                             posidmap=curr_posidmap,
                             vocab=self.vocab)
                        ]

                    # in case a limit on number of trees has been set, check it
                    if limit is not None and len(self.bank) > limit:
                        break

                    # start a new tree
                    curr_edges = []
                    curr_parents = {}
                    curr_children = defaultdict(lambda: [])
                    curr_posidmap = {0: -1}
                    last_word_id = None
                    self.ngrams[(-1, )] += 1
                else:
                    # save the new word in the current tree
                    word_pos, word, parent_pos = ln.split("\t")
                    word_pos = int(word_pos)
                    parent_pos = int(parent_pos)
                    word_id = self.vocab.get_word_id(word)

                    curr_posidmap[word_pos] = word_id
                    if parent_pos != 0:  # we do not want to save edges from root
                        curr_edges += [(parent_pos, word_pos, )]
                    curr_children[parent_pos] += [word_pos]
                    curr_parents[word_pos] = parent_pos

                    self.ngrams[(word_id, last_word_id)] += 1  # save 2-gram
                    self.ngrams[(word_id, )] += 1  # save 1-gram
                    self.total_words += 1

                    last_word_id = word_id


def in_order(tree):
    """Iterate over the tree in-order.

    Expects tree in the tuple format.
    I.e. (node_id, [left_son_tree1, ...], [right_son_tree1, ...])"""
    if len(tree) == 1:
        yield tree[0]
    else:
        for node in tree[1][0]:
            for value in in_order(node): yield value
        yield tree[0]
        for node in tree[1][1]:
            for value in in_order(node): yield value


class GibbsTrees(object):
    """Operates over the trees: count their partitions, splits them,
    merges them, ..."""
    def __init__(self, treebank):
        # bank of trees to operate on
        self.treebank = treebank

        # for each tree create an array that defines its partitioning
        # i.e. each item in the array says which partition the given
        # node (defined by element index) belongs to; initially all belong
        # to partition 0
        self.tree_groups = [
            [0] + [0 for _ in tree.parents.keys()] for tree in self.treebank
        ]

        # counts of sub-trees
        self.counts = None
        self.counts_all = 0

    def __iter__(self):
        """Iterate over indexes of trees in treebank."""
        return iter(xrange(len(self.treebank)))

    def iter_treegroups(self):
        for tree, group in zip(self.treebank, self.tree_groups):
            yield tree, group

    def save_to_file(self, filename):
        """Sort sub-trees according to their occurence and save that to file."""
        with open(filename, "w+b") as f_in:
            res = self.counts.items()
            res.sort(key=lambda x: -x[1])

            #cnts = defaultdict(lambda: 0)

            for tree, cnt in res:
                print >> f_in, cnt,
                print >> f_in, "\t",
                print >> f_in, " ".join([self.treebank.vocab.rev(x) for x in in_order(tree)]),
                #print >> f_in, " ".join([self.treebank.vocab.rev(x) for x in tree]),
                print >> f_in, "\t",
                print >> f_in, tree

                #cnts[len(list(in_order(tree)))] += 1

            #print cnts.items()

    def print_top(self, n=10):
        res = self.counts.items()
        res.sort(key=lambda x: -x[1])

        cnts = defaultdict(lambda: 0)

        for tree, cnt in res[:n]:
            print cnt, " ".join([self.treebank.vocab.rev(x) for x in in_order(tree)]),
            #print cnt, " ".join([self.treebank.vocab.rev(x) for x in tree]),
            print "\t",
            print tree

            #cnts[len(list(in_order(tree)))] += 1

        print cnts.items()

    def randomize_groups(self):
        """Randomly partition trees in the treebank into subtrees."""
        for tree_id, tree in enumerate(self.treebank):
            # for each tree, and each its node, with probability 0.5 split the tree
            # at that node
            root_children = tree.children[0]  # we want to split on all root children
            groups = self.tree_groups[tree_id]

            for word in tree.parents.keys():
                if word in root_children or random.random() > 0.5:
                    groups, _ = self.split_group(tree, groups, word)

            # set the new partitioning of this tree
            self.tree_groups[tree_id] = groups

    def count(self):
        """Go through all trees and count their subtrees."""
        self.counts = defaultdict(lambda: 0)

        for tree, groups in zip(self.treebank, self.tree_groups):
            self._update_count(tree, groups, 1)

    def get_subtrees(self, tree, groups):
        """Get array of nodes belonging to each of tree's partitions."""
        group_sets = defaultdict(lambda: set())

        for pos, group in enumerate(groups):
            if pos == 0: continue
            group_sets[group].add(pos)

        return dict(group_sets)

    def update_tree_count(self, tree_id, how):
        """Update counts of subtrees of the given tree."""
        self._update_count(self.treebank[tree_id], self.tree_groups[tree_id], how)

    def update_subtree_count(self, tree_id, group_id, how):
        self._update_count(self.treebank[tree_id], self.tree_groups[tree_id], how, which=group_id)

    def get_all_subtree_count(self):
        """Get the number of all subtrees in the treebank."""
        return self.counts_all

    def get_subtree_count(self, tree, subtree):
        """Get the number of the given subtrees."""
        treehash = self.hash_tree(tree, subtree)
        return self.counts[treehash]

    def _update_count(self, tree, groups, how=1, which=None):
        """Add/Subtract counts for subtrees of the given tree."""
        subtrees = self.get_subtrees(tree, groups)
        for group_id, subtree in subtrees.items():
            if which is not None and group_id != which:
                continue

            # convert tree to equivalence class representation
            treehash = self.hash_tree(tree, subtree)

            # update counts
            self.counts[treehash] += how
            self.counts_all += how

    def merge_group(self, tree, groups, word_pos):
        """Merge subtrees given by word_pos and its parent.

        Return:
         tuple(new groups,
              (id of my parents group, i.e. the one that survives, id of my old group)
          )"""
        group_id1 = groups[tree.parents[word_pos]]  # what's my parents group?
        group_id2 = groups[word_pos]  # what's my group?

        # put all nodes in my group to the group of my parent
        groups = [x if x != group_id2 else group_id1 for x in groups]

        return groups, (group_id1, group_id2,)

    def split_group(self, tree, groups, word_pos):
        """Split tree at the word given by word_pos into subtrees.

        Returns: tuple(new groups, new group id)
        """

        new_gid = max(groups) + 1  # get id for new group
        res_groups = list(groups)  # copy current group assignment

        # in BFS fashion update the group assignment from word_pos down
        queue = [word_pos]
        while len(queue) > 0:
            curr_word = queue.pop()
            res_groups[curr_word] = new_gid  # update group

            # add word's children to BFS queue to process later
            queue = [item for item in tree.children[curr_word]] + queue

        return res_groups, new_gid

    def get_root(self, tree, subtree):
        """Find root of the given subtree."""

        res = None
        # root is the node that does not have a parent, so find it
        for node in subtree:
            if not tree.parents[node] in subtree:
                if res is None:
                    res = node  # don't break yet, because we might find out that
                                # the parent should actually be ROOT (i.e. node
                                # with number 0)
                else:
                    res = 0  # in case this subtree has 2 parents, it implies
                             # that the root should be the ROOT of the tree
                    break

        return res

    def _mktree(self, tree, subtree, node):
        """Convert subtree starting at the given node to its tuple representation."""
        node_id = tree.posidmap[node]
        if len(tree.children[node]) > 0:
            left_subtree = tuple(
                self._mktree(tree, subtree, x) for x in tree.children[node]
                if x < node and x in subtree)

            right_subtree = tuple(
                self._mktree(tree, subtree, x)
                for x in tree.children[node]
                if x > node and x in subtree)

            return (node_id, (left_subtree,right_subtree,))
        else:
            return (node_id, )


    def hash_tree(self, tree, subtree):
        """Find equivalence class representation of the given subtree."""


        return tuple(sorted([tree.posidmap[node] for node in subtree]))

    #def hash_tree(self, tree, subtree=None):
    #    return tuple(tree.posidmap[node] for node in subtree)



class GibbsSampling(object):
    """Proceed with gibbs sampling on the trees."""
    p_tree_gen = 0.5
    p_tree_stop = 1.0 - p_tree_gen

    alfa = None

    def __init__(self, gibbs_trees, tree_decoding, alfa):
        # tree partitioning toolkit on which to operate
        self.gibbs_trees = gibbs_trees

        # count collector for decoding
        self.tree_decoding = tree_decoding

        # alfa for the boxed formula
        self.alfa = alfa

    def p0tree(self, tree, subtree):
        """Compute prior for the given subtree."""

        res = 1.0
        for node in subtree:
            #bigram_cnt = self.gibbs_trees.treebank.ngrams[(tree.posidmap[node], tree.posidmap[tree.parents[node]])]
            #unigram_cnt = self.gibbs_trees.treebank.ngrams[(tree.posidmap[tree.parents[node]], )]
            #unigram_cnt = self.gibbs_trees.treebank.ngrams[(tree.posidmap[node], )]
            total_words = self.gibbs_trees.treebank.total_words
            res *= 1.0 / total_words
            #res *= 1.0 / unigram_cnt
            res *= self.p_tree_gen

            #res *= float(unigram_cnt) / total_words
            #res *= self.p_tree_gen

        # res *= (1.0 / (len(self.gibbs_trees.treebank.vocab))) ** len(subtree)
        #print res, subtree
        res /= self.p_tree_gen  # because we multiply one more time in the loop
        res *= self.p_tree_stop

        return res

    def ptree(self, tree, subtree):
        """Compute conditional probability for this subtree,
        given the rest of the treebankd."""

        p = self.alfa * self.p0tree(tree, subtree) + self.gibbs_trees.get_subtree_count(tree, subtree)
        p /= self.alfa + self.gibbs_trees.get_all_subtree_count()

        return p

    def do(self, iterations, collect_start):
        """Do the given number of iterations of gibbs sampling."""
        for i in range(iterations):
            print >> sys.stderr, "Doing %d. iteration" % i
            self.do_iteration()

            if i >= collect_start:
                self.tree_decoding.collect_groups()

    def do_iteration(self):
        """Do one iteration over the trees."""
        # randomize the order which we will go through the trees
        seq = list(self.gibbs_trees)
        random.shuffle(seq)

        # go through all trees
        for tree_id in seq:

            # discount current tree
            self.gibbs_trees.update_tree_count(tree_id, -1)

            # get subtrees of the current tree
            curr_tree = self.gibbs_trees.treebank[tree_id]
            curr_groups = self.gibbs_trees.tree_groups[tree_id]
            curr_subtrees = self.gibbs_trees.get_subtrees(curr_tree, curr_groups)

            # randomly select a place in the tree to change (= to merge/split)
            # do it by randomly going through all nodes in the tree, and picking the first one
            # on which we can proceed with some action
            possible_picks = curr_tree.parents.keys()
            random.shuffle(possible_picks)

            # find a node on which we can make a small change (because we cannot do this on every node)
            while len(possible_picks) > 0:
                sel_node = possible_picks.pop()

                # do we join/cut the connecting edge?
                sel_node_parent = curr_tree.parents[sel_node]
                is_not_boundary_node = curr_groups[sel_node] == curr_groups[sel_node_parent]
                is_not_roots_child = sel_node_parent != 0

                if not (is_not_boundary_node or is_not_roots_child):
                    continue

                # if selected node is not on the boundary of two groups, we have to split it
                if is_not_boundary_node:
                    # compute old group's probability
                    old_group_id = curr_groups[sel_node]
                    oldp = self.ptree(curr_tree, curr_subtrees[old_group_id])

                    # split
                    new_groups, new_group_id = self.gibbs_trees.split_group(curr_tree, curr_groups, sel_node)
                    new_subtrees = self.gibbs_trees.get_subtrees(curr_tree, new_groups)

                    # compute new group1's probability
                    newp = self.ptree(curr_tree, new_subtrees[old_group_id])

                    # compute new group2's probability
                    newp *= self.ptree(curr_tree, new_subtrees[new_group_id])

                # if selected node is not roots child but is boundary node, we can merge it
                elif is_not_roots_child:
                    # merge
                    new_groups, (old_group_id1, old_group_id2) = self.gibbs_trees.merge_group(curr_tree, curr_groups, sel_node)
                    new_subtrees = self.gibbs_trees.get_subtrees(curr_tree, new_groups)

                    # compute old group1's probability
                    oldp = self.ptree(curr_tree, curr_subtrees[old_group_id1])

                    # compute old group2's probability
                    oldp *= self.ptree(curr_tree, curr_subtrees[old_group_id2])

                    # compute new group probability
                    newp = self.ptree(curr_tree, new_subtrees[old_group_id1])
                    #newp = self.ptree_whole(curr_tree, new_subtrees)

                else:
                    # if we are root's child and the boundary node, we canot do anything, so move along
                    continue

                # normalize the probabilities and set the boundary for making decision on this change
                if oldp + newp > 0.0:
                    boundary = newp / (newp + oldp)
                else:
                    boundary = 0.5

                #if tree_id < 10:
                #    print tree_id, "%.5f" % boundary, "split" if (is_not_boundary_node) else "merge"

                # randomly, according to the computed probability either commit the change or not
                rand = random.random()
                if rand < boundary:
                    self.gibbs_trees.tree_groups[tree_id] = new_groups  # commit

                # if we got up until this point, we found a suitable node to make a change at, so
                # end the search
                break

            # update the counts of subtrees in the current tree
            self.gibbs_trees.update_tree_count(tree_id, 1)  #, groups, 1)


class TreeDecoding(object):
    """Provides splitting of the trees in the treebank into phrases. It is based
    on collecting counts of phrase partitionings. Phrase partitionings are based
    on tree partitionings."""
    def __init__(self, gibbs_trees):
        self.gibbs_trees = gibbs_trees

        # counts of partitionings of the sentence of a given tree:
        #  split_counts[tree_id][phrase_partitioning_tuple] = count
        self.split_counts = defaultdict(lambda: defaultdict(lambda: 0))

        self.counts = defaultdict(lambda: 0)

    def collect_groups(self):
        """Collect the counts from current tree partitioning."""
        for tree_id, (tree, groups) in enumerate(self.gibbs_trees.iter_treegroups()):
            self.split_counts[tree_id][self.get_partitioning(tree, groups)] += 1

    def get_partitioning(self, tree, groups):
        """Based on tree partitioning, create a phrase partitioning. I.e. split
        the phrase into several linear segments."""
        subtrees = self.gibbs_trees.get_subtrees(tree, groups)

        res = []
        for subtree in subtrees.values():
            res += [tuple(sorted(subtree))]

        return tuple(sorted(res))

    def save_to_file(self, filename):
        """Save most likely partitionings of each phrase to the output file."""
        with open(filename, "w") as f_out:
            for tree_id, splits in self.split_counts.items():
                tree = self.gibbs_trees.treebank[tree_id]
                max_split = sorted(splits.items(), key=lambda x: -x[1])[0][0]

                for group in max_split:
                    for word in group:
                        word_id = tree.posidmap[word]
                        print >> f_out, self.gibbs_trees.treebank.vocab.rev(word_id),
                    print >> f_out, ""
                print >> f_out, ""


def main(fname):
    # parameters
    n_trees = 1000
    alfa = 1.0
    iters = 1000
    exp_id = int(sys.argv[2])
    params = "e%d_n%s_alfa%.6f_i%d" % (exp_id, str(n_trees), alfa, iters,)

    tb = TreeBank()
    tb.load(fname, n_trees)

    gt = GibbsTrees(tb)
    gt.randomize_groups()
    gt.count()

    td = TreeDecoding(gt)

    gs = GibbsSampling(gt, td, alfa=alfa)
    gs.do(iters, collect_start=iters/3)
    #gt.save_to_file("trees_out_dev_%s.txt" % params)

    td.save_to_file("phrases_%s.txt" % params)

    #import ipdb; ipdb.set_trace()

    #gt.print_top(40)
    #import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main(sys.argv[1])