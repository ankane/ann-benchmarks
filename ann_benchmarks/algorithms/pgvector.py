from __future__ import absolute_import
import csv
import psycopg2
from ann_benchmarks.algorithms.base import BaseANN


class Pgvector(BaseANN):
    def __init__(self, metric, lists):
        self._metric = metric

        self._lists = lists
        self._probes = None

        self._opclass = {'angular': 'vector_cosine_ops', 'euclidean': 'vector_l2_ops'}[metric]
        self._op = {'angular': '<=>', 'euclidean': '<->'}[metric]

        self._conn = psycopg2.connect('user=postgres dbname=vector_bench')
        self._conn.autocommit = True
        self._cur = self._conn.cursor()

    def fit(self, X):
        self._cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
        self._cur.execute('DROP TABLE IF EXISTS tst')
        self._cur.execute('CREATE TABLE tst (id integer, vec vector(' + str(X.shape[1]) + '))')

        with open('/tmp/data.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for i, x in enumerate(X):
                csvwriter.writerow([i, '[' + ','.join(str(v) for v in x.tolist()) + ']'])
        self._cur.execute('COPY tst (id, vec) FROM \'/tmp/data.csv\' WITH (FORMAT csv)')

        self._cur.execute('CREATE INDEX ON tst USING ivfflat (vec ' + self._opclass + ') WITH (lists = ' + str(self._lists) + ')')

    def set_query_arguments(self, probes):
        self._probes = probes
        self._cur.execute('SET ivfflat.probes = %s', (str(probes),))

    def query(self, v, n):
        self._cur.execute('SELECT id FROM tst ORDER BY vec ' + self._op + ' %s LIMIT %s', ('[' + ','.join(str(v2) for v2 in v.tolist()) + ']', n))
        res = self._cur.fetchall()
        return [r[0] for r in res]

    def __str__(self):
        return 'Pgvector(lists=%d, probes=%d)' % (self._lists, self._probes)
