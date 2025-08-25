#include "job.h"
#include <omp.h>

#include "tsp.h"

#ifdef NO_CACHE_COHERENCE // See close_queue() for details
static int waiting_threads = 0;
#endif

inline void reset_queue(job_queue_t *q)
{
	q->begin = 0;
	q->end = 0;
}

void init_queue(job_queue_t *q, unsigned long max_size, int (*repopulate_queue)(void *), void *repopulate_queue_par)
{
	q->max_size = max_size;
	q->status = QUEUE_OK;
	q->repopulate_queue = repopulate_queue;
	q->repopulate_queue_par = repopulate_queue_par;
	reset_queue(q);

	q->buffer = (job_queue_node_t *)malloc(sizeof(job_queue_node_t) * q->max_size);
	LOG("Trying to allocate %lu bytes for the queue (max_size = %lu)\n", sizeof(job_queue_node_t) * q->max_size, q->max_size);
	assert(q->buffer != NULL);
}

static void close_queue(job_queue_t *q)
{
	q->status = QUEUE_CLOSED;
}

void add_job(job_queue_t *q, job_t j)
{
#pragma omp critical(job_queue_add)
	{
		assert(q->end < (int)q->max_size);
		q->buffer[q->end].tsp_job.len = j.len;
		memcpy(&q->buffer[q->end].tsp_job.path, j.path, sizeof(path_t));
		q->end++;
	}
}

int get_job(job_queue_t *q, job_t *j)
{
	int job_found = 0;

#pragma omp critical(job_queue_get)
	{
		if (q->begin < q->end)
		{
			// Jobs available in current batch
			int index = q->begin++;
			memcpy(j, &q->buffer[index].tsp_job, sizeof(job_t));
			job_found = 1;
		}
		else if (q->status == QUEUE_OK)
		{
			// Need to repopulate queue
			q->status = QUEUE_WAIT;
			reset_queue(q);
			int jobs_added = q->repopulate_queue(q->repopulate_queue_par);
			if (jobs_added && q->begin < q->end)
			{
				q->status = QUEUE_OK;
				int index = q->begin++;
				memcpy(j, &q->buffer[index].tsp_job, sizeof(job_t));
				job_found = 1;
			}
			else
			{
				q->status = QUEUE_CLOSED;
				job_found = 0;
			}
		}
		else
		{
			// Queue is closed
			job_found = 0;
		}
	}

	return job_found;
}

void free_queue(job_queue_t *q)
{
	free(q->buffer);
}
