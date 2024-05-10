#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#define INFINITY 99999

// Define structure for edges
typedef struct {
    int node;
    int weight;
} Edge;

// Function to read graph representation from a file
void read_graph(char *filename, int *num_nodes, Edge **adj_list) {
    // Open file for reading
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Unable to open file %s\n", filename);
        exit(1);
    }

    // Variables for reading from file
    int start_node, end_node, weight;
    int max_node = 0; // Keep track of maximum node number encountered
    int num_edges = 0;
    int max_edges = 1000000;
    Edge *edges = (Edge *)malloc(max_edges * sizeof(Edge));

    // Read edges from file
    while (fscanf(file, "%d %d %d", &start_node, &end_node, &weight) != EOF) {
        if (num_edges >= max_edges) {
            max_edges *= 2;
            edges = (Edge *)realloc(edges, max_edges * sizeof(Edge));
        }
        edges[num_edges].node = end_node;
        edges[num_edges].weight = weight;
        num_edges++;
        if (end_node > max_node) {
            max_node = end_node;
        }
    }

    fclose(file);

    // Calculate number of nodes
    *num_nodes = max_node + 1;

    // Allocate memory for adjacency list
    *adj_list = (Edge *)malloc(num_edges * sizeof(Edge));
    for (int i = 0; i < num_edges; i++) {
        (*adj_list)[i].node = edges[i].node;
        (*adj_list)[i].weight = edges[i].weight;
    }

    free(edges);
}

// Function to initialize distance matrix
void initialize_distance_matrix(int num_nodes, int **distance_matrix, Edge *adj_list) {
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_nodes; j++) {
            distance_matrix[i][j] = INFINITY;
        }
    }

    for (int i = 0; i < num_nodes; i++) {
        distance_matrix[i][i] = 0;
    }

    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_nodes; j++) {
            for (int k = 0; k < num_nodes; k++) {
                if (i == j) {
                    distance_matrix[i][j] = 0;
                } else if (i == adj_list[k].node) {
                    distance_matrix[i][adj_list[k].node] = adj_list[k].weight;
                }
            }
        }
    }
}

// Function to compute shortest paths using Dijkstra's algorithm
void dijkstra(int num_nodes, int source, int *distance_matrix, Edge *adj_list) {
    int *visited = (int *)malloc(num_nodes * sizeof(int));
    int *distance = (int *)malloc(num_nodes * sizeof(int));
    int min_distance, next_node;

    for (int i = 0; i < num_nodes; i++) {
        visited[i] = 0;
        distance[i] = distance_matrix[source * num_nodes + i];
    }

    visited[source] = 1;

    for (int i = 0; i < num_nodes; i++) {
        min_distance = INFINITY;

        for (int j = 0; j < num_nodes; j++) {
            if (distance[j] < min_distance && !visited[j]) {
                min_distance = distance[j];
                next_node = j;
            }
        }

        visited[next_node] = 1;

        for (int j = 0; j < num_nodes; j++) {
            if (!visited[j]) {
                if (min_distance + distance_matrix[next_node * num_nodes + j] < distance[j]) {
                    distance[j] = min_distance + distance_matrix[next_node * num_nodes + j];
                }
            }
        }
    }

    for (int i = 0; i < num_nodes; i++) {
        distance_matrix[source * num_nodes + i] = distance[i];
    }

    free(visited);
    free(distance);
}

// Function to parallelize computation of shortest paths
void parallel_shortest_paths(int num_nodes, int source, int k, int *distance_matrix, Edge *adj_list) {
    // Initialize local distance and visited arrays
    int *local_distance = (int *)malloc(num_nodes * sizeof(int));
    int *local_visited = (int *)malloc(num_nodes * sizeof(int));
    int *local_min_distance = (int *)malloc(num_nodes * sizeof(int));
    int *local_next_node = (int *)malloc(num_nodes * sizeof(int));

    // Initialize local distance and visited arrays
    for (int i = 0; i < num_nodes; i++) {
        local_distance[i] = distance_matrix[source * num_nodes + i];
        local_visited[i] = 0;
    }

    local_visited[source] = 1;

    // Parallelize Dijkstra's algorithm
    for (int i = 0; i < num_nodes; i++) {
        local_min_distance[i] = INFINITY;
        int global_min_distance = INFINITY;
        int global_next_node = -1;

        #pragma omp parallel for thread_num(3)
        for (int j = 0; j < num_nodes; j++) {
            if (local_distance[j] < local_min_distance[i] && !local_visited[j]) {
                #pragma omp critical
                {
                    if (local_distance[j] < global_min_distance) {
                        global_min_distance = local_distance[j];
                        global_next_node = j;
                    }
                }
            }
        }

        // Reduce local minimum distances and next nodes
        MPI_Allreduce(&global_min_distance, &local_min_distance[i], 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&global_next_node, &local_next_node[i], 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        // Update local minimum distances and next nodes
        local_visited[local_next_node[i]] = 1;

        #pragma omp parallel for thread_num(3)
        for (int j = 0; j < num_nodes; j++) {
            if (!local_visited[j]) {
                if (local_min_distance[i] + distance_matrix[local_next_node[i] * num_nodes + j] < local_distance[j]) {
                    local_distance[j] = local_min_distance[i] + distance_matrix[local_next_node[i] * num_nodes + j];
                }
            }
        }

        // Reduce local distances
        MPI_Allreduce(local_distance, distance_matrix + source * num_nodes, num_nodes, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    }

    free(local_distance);
    free(local_visited);
    free(local_min_distance);
    free(local_next_node);
}

int main(int argc, char *argv[]) {
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Read graph from file
    int num_nodes;
    Edge *adj_list;
    read_graph("Email-Enron.txt", &num_nodes, &adj_list);
    //read_graph("Email-EuAll.txt", &num_nodes, &adj_list);

    // Allocate memory for distance matrix
    int *distance_matrix = (int *)malloc(num_nodes * num_nodes * sizeof(int));

    // Initialize distance matrix
    initialize_distance_matrix(num_nodes, &distance_matrix, adj_list);

    // Define source node
    int source = 0;

    // Define value of K
    int k = 5;

    // Parallelize computation of shortest paths
    parallel_shortest_paths(num_nodes, source, k, distance_matrix, adj_list);

    // Clean up
    free(distance_matrix);
    free(adj_list);

    MPI_Finalize();

    return 0;
}
