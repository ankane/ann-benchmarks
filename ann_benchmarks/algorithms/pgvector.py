from __future__ import absolute_import
import io
from pgvector.psycopg import register_vector
import psycopg
from psycopg import sql
from ann_benchmarks.algorithms.base import BaseANN


# For best performance:
# - Use socket connection
# - Use prepared statements
# - Use binary format
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
        self._query = 'SELECT id FROM %s ORDER BY vec %s %%b LIMIT %%b' % (self._table, self._op)

        self._conn = psycopg.connect('dbname=pgvector_bench user=postgres')
        self._conn.autocommit = True
        self._conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        register_vector(self._conn)
        self._cur = self._conn.cursor()

    def fit(self, X):
        print('Creating table')
        self._cur.execute('DROP TABLE IF EXISTS %s' % (self._table,));
        self._cur.execute('CREATE TABLE %s (id integer, vec vector(%d))' % (self._table, X.shape[1]))

        with self._cur.copy("COPY %s (id, vec) FROM STDIN" % (self._table,)) as copy:
            for i, x in enumerate(X):
                copy.write(str(i))
                copy.write('\t')
                copy.write('[' + ','.join(str(v) for v in x.tolist()) + ']')
                copy.write('\n')

        print('Creating index')
        self._cur.execute("SET maintenance_work_mem = '%s'" % ('256MB',))
        self._cur.execute('CREATE INDEX ON %s USING ivfflat (vec %s) WITH (lists = %d)' % (self._table, self._opclass, self._lists))
        self._cur.execute('ANALYZE %s' % self._table)

    def set_query_arguments(self, probes):
        self._probes = probes
        self._cur.execute("SET ivfflat.probes = '%s'" % (str(probes),))

    def query(self, v, n):
        res = self._cur.execute(self._query, (v, n), prepare=True).fetchall()
        return [r[0] for r in res]

    def __str__(self):
        return 'Pgvector(lists=%d, probes=%d)' % (self._lists, self._probes)
