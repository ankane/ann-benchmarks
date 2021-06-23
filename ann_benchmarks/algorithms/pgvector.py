from __future__ import absolute_import
import io
import asyncio
import asyncpg
from pgvector.asyncpg import register_vector
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
        self._loop = asyncio.get_event_loop()
        self._loop.run_until_complete(self.init_async(metric, lists))

    def fit(self, X):
        self._loop.run_until_complete(self.fit_async(X))

    def set_query_arguments(self, probes):
        self._loop.run_until_complete(self.set_query_arguments_async(probes))

    def query(self, v, n):
        return self._loop.run_until_complete(self.query_async(v, n))

    async def init_async(self, metric, lists):
        self._metric = metric

        self._lists = lists
        self._probes = None

        self._opclass = {'angular': 'vector_cosine_ops', 'euclidean': 'vector_l2_ops'}[metric]
        self._op = {'angular': '<=>', 'euclidean': '<->'}[metric]
        self._table = 'vectors_%s_%d' % (metric, lists)
        self._query = 'SELECT id FROM %s ORDER BY vec %s $1 LIMIT $2' % (self._table, self._op)

        self._conn = await asyncpg.connect(database='vector_bench')
        await self._conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        await register_vector(self._conn)

    async def fit_async(self, X):
        await self._conn.execute('CREATE TEMPORARY TABLE %s (id integer, vec vector(%d))' % (self._table, X.shape[1]))

        records = enumerate(X)
        await self._conn.copy_records_to_table(self._table, records=records, columns=('id', 'vec'))

        await self._conn.execute('CREATE INDEX ON %s USING ivfflat (vec %s) WITH (lists = %d)' % (self._table, self._opclass, self._lists))
        await self._conn.execute('ANALYZE %s' % self._table)
        self._stmt = await self._conn.prepare(self._query)

    async def set_query_arguments_async(self, probes):
        self._probes = probes
        await self._conn.execute('SET ivfflat.probes = %s' % probes)

    async def query_async(self, v, n):
        res = await self._stmt.fetch(v, n)
        return [r[0] for r in res]

    def __str__(self):
        return 'Pgvector(lists=%d, probes=%d)' % (self._lists, self._probes)
