#include "exec.h"
#include <omp.h>

void do_work(tsp_t_pointer tsp)
{
#pragma omp parallel num_threads(tsp->nb_threads)
	{
		int thread_id = omp_get_thread_num();

		tsp_thread_par_t *par = (tsp_thread_par_t *)malloc(sizeof(tsp_thread_par_t));
		assert(par != NULL);
		par->tsp = tsp;
		par->thread_id = thread_id;

		worker_openmp(par);

		free(par);
	}
}

tsp_t_pointer init_execution(int cluster_id, int nb_clusters, int nb_partitions, int nb_threads, int nb_towns, int seed)
{
	tsp_t_pointer ret = init_tsp(cluster_id, nb_clusters, nb_partitions, nb_threads, nb_towns, seed);
	return ret;
}

void start_execution(tsp_t_pointer tsp)
{
	do_work(tsp);
}

void end_execution(tsp_t_pointer tsp)
{
	LOG("Cluster exiting (%d clusters). Partitions processed: %d. Min: %d\n", tsp->nb_clusters, tsp->processed_partitions, tsp->min_distance);
	free_tsp(tsp);
}
