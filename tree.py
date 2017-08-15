import os
import sys
import json
import yaml
import graphviz
import argparse
from collections import defaultdict


# Inverse the foreign key dictionary for reverse references.
def reverse_foreign_keys(fks):
    rfks = {}

    for c_table, fks in fks.items():
        for fk in fks:
            fk = dict(fk)
            r_table = fk.pop('referred_table')
            fk['constrained_table'] = c_table
            rfks.setdefault(r_table, [])
            rfks[r_table].append(fk)

    return rfks


class Node():
    def __init__(self, table, source=None, predicate=None,
                 reverse=None, nullable=False, depth=0):
        """Represents a relation between two tables.

            `table` - the table this node represents
            `source` - a reference to the source table node.
            `predicate` - the predicate to join these two tables.
            `reverse` - denotes whether this node was derived from a
            forward relationship a reverse relationship.
            `nullable` - flags whether the relationship is nullable. this
            determines the type of join to use.
            `depth` - the depth of this node relative to the root
        """

        self.table = table
        self.source = source
        self.predicate = predicate
        self.reverse = reverse
        self.nullable = nullable
        self.depth = depth

        self.children = []

    def remove_child(self, table):
        "Removes a child node for a given model."
        for i, node in enumerate(list(self.children)):
            if node.table == table:
                return self.children.pop(i)


def normalize_predicate(preds):
    if not preds:
        return

    npreds = []

    for expr in preds:
        pred = None
        expr = tuple(expr)

        # 2 or 3-tuple
        if len(expr) == 3:
            pred = expr
        elif len(expr) == 2:
            pred = expr + ('=',)
        else:
            raise ValueError('invalid predicate format')

        npreds.append(pred)

    return tuple(npreds)


def invert_predicate(preds):
    if not preds:
        return

    ipreds = []

    for pred in preds:
        left, right, equ = pred
        ipreds.append((right, left, equ))

    return tuple(ipreds)


# The task is to find all paths to all tables in the schema (if possible).
# By default, the shortest path between two tables is used.
# Cycles are not allowed and thus relationships that go back through
# a table seen in the path is prevented.
# The algorithm traverses each foreign key, both forward and reverse.
class Tree():
    def __init__(self, schema, table, excluded_joins=None,
                 required_joins=None, excluded_tables=None):

        self.schema = schema
        self.root_table = table
        self.excluded_tables = excluded_tables or set()

        self.columns = schema['columns']
        self.foreign_keys = schema['foreign_keys']
        self.rev_foreign_keys = reverse_foreign_keys(self.foreign_keys)

        # Build the routes that are allowed/preferred
        self.required_joins = self._build_routes(
            required_joins or (),
            allow_redundant_targets=False)

        # Build the routes that are excluded
        self.excluded_joins = self._build_routes(excluded_joins or ())

        self._nodes = {}
        self._alt_joins = defaultdict(set)
        self._table_fields = defaultdict(set)

    def _build_routes(self, routes, allow_redundant_targets=True):
        """Routes provide a means of specifying JOINs between two tables.

            routes - a collection of dicts defining source->target mappings
                 with optional `predicate` specifier and `symmetrical`
                 attribute.

            allow_redundant_targets - whether two routes in this collection
                 are allowed to have the same target - this should NOT
                 be allowed for required routes.
        """

        joins = {}
        targets_seen = set()

        for route in routes:
            source = route.get('source')
            target = route.get('target')
            predicate = normalize_predicate(route.get('predicate'))
            symmetrical = route.get('symmetrical')

            if not allow_redundant_targets:
                if target in targets_seen:
                    raise ValueError('Table {} cannot be the target of '
                                     'more than one route in this list'
                                     .format(target))
                else:
                    targets_seen.add(target)

            # The `joins` hash defines pairs which are explicitly joined
            # via the specified field.  If no field is defined, then the
            # join field is implied or does not matter; the route is reduced
            #  to a straight lookup.
            joins[(source, target)] = predicate

            if symmetrical:
                if not allow_redundant_targets:
                    if source in targets_seen:
                        raise ValueError('Table {} cannot be the target of '
                                         'more than one route in this list'
                                         .format(source))
                    else:
                        targets_seen.add(source)

                joins[(target, source)] = invert_predicate(predicate)

        return joins

    def join_allowed(self, source, target, predicate=None):
        join = (source, target)

        # No cycles unless the left and right columns
        # of the predicate differ which would imply a self-join.
        if target == source:
            if not predicate:
                return False

            for (left, right, _) in predicate:
                if left == right:
                    return False

        # Prevent join to excluded tables.
        if target in self.excluded_tables:
            return False

        # Never go back through the root.
        if target == self.root_table:
            return False

        # Apply excluded joins if any.
        if join in self.excluded_joins:
            _predicate = self.excluded_joins[join]

            # All joins between these two tables are prevented.
            if not _predicate:
                return False

            # Join through a specific predicate is prevented.
            elif _predicate and _predicate == predicate:
                return False

        # Check if the join is allowed by a required rule.
        for (_source, _target), _predicate in self.required_joins.items():
            # A join to the target must come from an explicit table.
            if _target == target:
                if _source != source:
                    return False

                # If a predicate is supplied, it must join through this
                # predicate.
                if predicate and _predicate and _predicate != predicate:
                    return False

        return True

    def add_node(self, source, table, predicate, reverse, nullable, depth):
        prev = self._nodes.get(table)

        # Join is not better or preferred.
        if prev and prev['depth'] <= depth:
            self._alt_joins[table].add((source.table, predicate))
            return

        # Remove previous node from path.
        if prev:
            prev['source'].remove_child(table)

        node = Node(table,
                    source=source,
                    predicate=predicate,
                    reverse=reverse,
                    nullable=nullable,
                    depth=depth)

        for (left, right, _) in predicate:
            self._table_fields[source.table].add(left)
            self._table_fields[table].add(right)

        # Update
        self._nodes[table] = {
            'source': source,
            'depth': depth,
            'node': node,
        }

        node = self.find_relations(node, depth)
        source.children.append(node)

    def find_relations(self, source, depth=0):
        depth += 1

        table = source.table

        columns = {c['name']: c for c in self.columns[table]}
        foreign_keys = self.foreign_keys.get(table, ())
        rev_foreign_keys = self.rev_foreign_keys.get(table, ())

        # Foreign keys defined on the table.
        for fk in foreign_keys:
            src_columns = fk['constrained_columns']
            ref_table = fk['referred_table']
            ref_columns = fk['referred_columns']

            # Table, column, equality triples.
            predicate = tuple([
                (l, r, '=')
                for (l, r) in zip(src_columns, ref_columns)
            ])

            if not self.join_allowed(table, ref_table, predicate):
                continue

            # If any of the source column is nullable then the
            # foreign key is optional.
            nullable = any([
                columns[c]['nullable']
                for c in src_columns
            ])

            # TODO: check for unique constraint for exclusivity?
            self.add_node(source,
                          table=ref_table,
                          predicate=predicate,
                          reverse=False,
                          nullable=nullable,
                          depth=depth)

        # Foreign keys pointing to the table.
        for fk in rev_foreign_keys:
            ref_table = fk['constrained_table']
            ref_columns = fk['constrained_columns']
            src_columns = fk['referred_columns']

            # Table, column, equal
            predicate = tuple([
                (l, r, '=')
                for (l, r) in zip(src_columns, ref_columns)
            ])

            if not self.join_allowed(table, ref_table, predicate):
                continue

            # Reverse key is also nullable.
            nullable = True

            # TODO: check for unique constraint for exclusivity?
            self.add_node(source,
                          table=ref_table,
                          predicate=predicate,
                          reverse=True,
                          nullable=nullable,
                          depth=depth)

        return source

    def build(self):
        node = Node(self.root_table)

        self.find_relations(node)

        self._nodes[self.root_table] = {
            'source': None,
            'depth': 0,
            'node': node,
        }

        self.root_node = node

        return node

    def _path_to_table(self, table, node, path=None):
        "Returns a list representing the path of nodes to the model."
        if path is None:
            path = []

        if node.table == table:
            return path

        for child in node.children:
            mpath = self._path_to_table(table, child, path + [child])
            # TODO why is this condition here?
            if mpath:
                return mpath

    def _node_path(self, table):
        "Returns a list of nodes thats defines the path of traversal."
        return self._path_to_table(table, self.root_node)

    def get_joins(self, table):
        "Returns a list of joins from the root to the target table."
        path = self._node_path(table)

        joins = []

        for i, node in enumerate(path):
            joins.append({
                'left_table': node.source.table,
                'right_table': node.table,
                'predicate': [
                    {
                        'left_column': left,
                        'right_column': right,
                        'operator': op,
                    } for (left, right, op) in node.predicate
                ],
                'reverse': node.reverse,
                'nullable': node.nullable,
            })

        return joins


def to_graph(graph, traverser, node, alt_joins=False):
    add_nodes(graph, traverser, node)

    # Add excluded tables.
    for table in traverser.excluded_tables:
        graph.node(table,
                   label=table,
                   fontcolor='#999999',
                   style='filled',
                   fillcolor='#eeeeee',
                   shape='record')

    # Alternate join paths.
    # TODO: add additional fields to nodes above, prior to adding.
    if alt_joins:
        for target, sources in traverser._alt_joins.items():
            for (source, predicate) in sources:
                for (left, right, _) in predicate:
                    source_field = '{}:{}'.format(source, left)
                    target_field = '{}:{}'.format(target, right)
                    graph.edge(source_field,
                               target_field,
                               style='dashed',
                               color='gray')


def node_label(traverser, node):
    fields = [node.table] + list(sorted(traverser._table_fields[node.table]))
    return '|'.join(['<{}> {}'.format(f, f) for f in fields])


def add_nodes(graph, traverser, node):
    label = node_label(traverser, node)

    graph.node(node.table,
               label=label,
               shape='record')

    if node.children:
        for child in node.children:
            add_nodes(graph, traverser, child)

            for (left, right, _) in child.predicate:
                source_field = '{}:{}'.format(node.table, left)
                target_field = '{}:{}'.format(child.table, right)
                graph.edge(source_field, target_field)


def to_index(traverser):
    index = {}
    for dst in traverser._nodes:
        index[dst] = traverser.get_joins(dst)
    return index


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('schema', help='Schema file')
    parser.add_argument('--table', help='Root table')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--graphviz', help='Output a graphviz output')

    args = parser.parse_args()

    schema = None
    config = None

    # Read schema.
    with open(args.schema) as f:
        if args.schema.endswith('.json'):
            schema = json.load(f)
        else:
            schema = yaml.load(f)

    # Read or init config.
    if args.config:
        with open(args.config) as f:
            if args.config.endswith('.json'):
                config = yaml.json(f)
            else:
                config = yaml.load(f)
    elif args.table:
        config = {
            'table': args.table,
        }
    else:
        print('error: a config file or table must be specified\n')
        parser.print_help()
        sys.exit(1)

    tree = Tree(schema, **config)
    node = tree.build()

    # Output the tree.
    json.dump(to_index(tree), sys.stdout)

    # Generate and write graphviz file.
    if args.graphviz:
        filename, ext = os.path.splitext(args.graphviz)

        # Initialize graph.
        graph = graphviz.Digraph(
            format=ext[1:],
            graph_attr={
                'overlap': 'false',
                'splines': 'compound',
                'nodesep': '1',
            },
        )

        # Build the graphviz graph from the tree.
        to_graph(
            graph,
            tree,
            node,
            alt_joins=False,
        )

        # Output the visual.
        graph.render(filename=filename, cleanup=True)
