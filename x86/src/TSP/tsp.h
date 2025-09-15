#ifndef __TSP_H
#define __TSP_H

#include "job.h"

typedef struct
{
	int nb_towns;
	struct
	{
		int to_city;
		int dist;
	} info[MAX_TOWNS][MAX_TOWNS];
} distance_matrix_t;

typedef struct
{
	int max_hops;
	int cluster_id;
	int nb_clusters;
	int nb_threads;
	int nb_partitions;
	char PADDING1[PADDING(5 * sizeof(int))];

	distance_matrix_t *distance;
	char PADDING2[PADDING(sizeof(distance_matrix_t))];

	int min_distance;
	int processed_partitions;
	char PADDING4[PADDING(sizeof(int) * 2)];
	job_queue_t queue;
} tsp_t;

typedef tsp_t *tsp_t_pointer;

typedef struct
{
	tsp_t *tsp;
	int thread_id;
} tsp_thread_par_t;

tsp_t_pointer init_tsp(int cluster_id, int nb_clusters, int nb_partitions, int nb_threads, int nb_towns, int seed);
void free_tsp(tsp_t_pointer tsp);

void *worker(void *tsp_worker_par);
void worker_openmp(tsp_thread_par_t *p);
int tsp_get_shortest_path(tsp_t_pointer tsp);
int tsp_update_minimum_distance(tsp_t_pointer tsp, int length);

// Internal functions
int init_max_hops(tsp_t_pointer tsp);
int present(int city, int hops, path_t *path);

// callback
extern void new_minimun_distance_found(tsp_t_pointer tsp);
extern partition_interval_t get_next_partition(tsp_t_pointer tsp);

#endif
