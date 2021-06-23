from __future__ import absolute_import
import io
import psycopg2
from ann_benchmarks.algorithms.base import BaseANN


# For best performance:
# - Use socket connection
# - Use prepared statements (todo)
# - Use binary format (todo)
# - Increase work_mem? (todo)
# - Increase shared_buffers? (todo)
# - Try different drivers (todo)
class Pgvector(BaseANN):
    def __init__(self, metric, lists):
        self._metric = metric

        self._lists = lists
        self._probes = None

        self._opclass = {'angular': 'vector_cosine_ops', 'euclidean': 'vector_l2_ops'}[metric]
        self._op = {'angular': '<=>', 'euclidean': '<->'}[metric]
        self._table = 'vectors_%s_%d' % (metric, lists)
        self._query = 'SELECT id FROM %s ORDER BY vec %s %%s LIMIT %%s' % (self._table, self._op)

        self._conn = psycopg2.connect('user=postgres dbname=vector_bench')
        self._conn.autocommit = True
        self._cur = self._conn.cursor()
        self._cur.execute('CREATE EXTENSION IF NOT EXISTS vector')

    def fit(self, X):
        self._cur.execute('CREATE TEMPORARY TABLE %s (id integer, vec vector(%d))' % (self._table, X.shape[1]))

        f = io.StringIO()
        for i, x in enumerate(X):
            f.write(str(i))
            f.write('\t')
            f.write('[' + ','.join(str(v) for v in x.tolist()) + ']')
            f.write('\n')
        f.seek(0)
        self._cur.copy_from(f, self._table, columns=('id', 'vec'))

        self._cur.execute('CREATE INDEX ON %s USING ivfflat (vec %s) WITH (lists = %d)' % (self._table, self._opclass, self._lists))
        self._cur.execute('ANALYZE %s' % self._table)

    def set_query_arguments(self, probes):
        self._probes = probes
        self._cur.execute('SET ivfflat.probes = %s', (str(probes),))

    def query(self, v, n):
        self._cur.execute(self._query, (v.tolist(), n))
        res = self._cur.fetchall()
        return [r[0] for r in res]

    def __str__(self):
        return 'Pgvector(lists=%d, probes=%d)' % (self._lists, self._probes)
