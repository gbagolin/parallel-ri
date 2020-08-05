#ifndef MALLOCUTILITY_H_
#define MALLOCUTILITY_H_

#include "CheckError.cuh"
#include "timer.h"

TIMEHANDLE load_s, load_s_q, make_mama_s, match_s, total_s;
double load_t = 0;
double load_t_q = 0;
double make_mama_t = 0;
double total_t = 0;
double match_t = 0;


int *d_test;
bool *d_printToConsole;
long *d_matchCount;
int *d_comparatorType;
long *d_steps;
long *d_triedcouples;
long *d_matchedcouples;

//mama
int *d_nof_sn;
int *d_edges_sizes;
int *d_source;
int *d_target;
void *d_attr;
int *d_offset_attr;
int *d_flat_edges_indexes;
int *d_map_node_to_state;
int *d_map_state_to_node;
int *d_parent_state;
int *d_parent_type;

int *d_r_nof_nodes;
int *d_r_flatten_in_adj_list;
int *d_r_offset_in_adj_list;
int *d_r_in_adj_sizes;
int *d_r_flatten_out_adj_list;
int *d_r_offset_out_adj_list;
int *d_r_out_adj_sizes;
void *d_r_flatten_nodes_attr;
int *d_r_offset_nodes_attr;
void *d_r_out_adj_attrs = NULL;

//query
int *d_q_nof_nodes;
int *d_q_flatten_in_adj_list;
int *d_q_offset_in_adj_list;
int *d_q_in_adj_sizes;
int *d_q_flatten_out_adj_list;
int *d_q_offset_out_adj_list;
int *d_q_out_adj_sizes;
void *d_q_flatten_nodes_attr;
int *d_q_offset_nodes_attr;

int comparatorType;


void in_out_malloc() {

    SAFE_CALL(cudaMalloc(&d_test, sizeof(int)));

    SAFE_CALL(cudaMalloc(&d_printToConsole, sizeof(bool)));

    SAFE_CALL(cudaMalloc(&d_matchCount, sizeof(long)));

    SAFE_CALL(cudaMalloc(&d_comparatorType, sizeof(int)));

    SAFE_CALL(cudaMalloc(&d_matchedcouples, sizeof(long)));

}

void in_out_memcpy(int *test,
                   int *comparatorType,
                   long *matchedcouples,
                   bool *printToConsole,
                   long *matchCount) {

    SAFE_CALL(cudaMemcpy(d_test, test, sizeof(int),
                         cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMemcpy(d_printToConsole, printToConsole, sizeof(bool),
                         cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMemcpy(d_matchCount, matchCount, sizeof(long),
                         cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMemcpy(d_comparatorType, comparatorType, sizeof(int),
                         cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMemcpy(d_matchedcouples, matchedcouples, sizeof(long),
                         cudaMemcpyHostToDevice));
}

void mama_malloc(MatchingMachine &matchingMachine) {


    SAFE_CALL(cudaMalloc(&d_nof_sn, sizeof(int)));
    SAFE_CALL(cudaMemcpy(d_nof_sn, &matchingMachine.nof_sn, sizeof(int),
                         cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMalloc(&d_edges_sizes, sizeof(int) * matchingMachine.nof_sn));
    SAFE_CALL(cudaMemcpy(d_edges_sizes, matchingMachine.edges_sizes, sizeof(int) * matchingMachine.nof_sn,
                         cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMalloc(&d_source, sizeof(int) * matchingMachine.total_count));
    SAFE_CALL(cudaMemcpy(d_source, matchingMachine.source, sizeof(int) * matchingMachine.total_count,
                         cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMalloc(&d_target, sizeof(int) * matchingMachine.total_count));
    SAFE_CALL(cudaMemcpy(d_target, matchingMachine.target, sizeof(int) * matchingMachine.total_count,
                         cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMalloc(&d_attr, sizeof(char) * (matchingMachine.length_string + 1)));
    SAFE_CALL(cudaMemcpy(d_attr, matchingMachine.attr, sizeof(char) * matchingMachine.length_string,
                         cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMalloc(&d_offset_attr, sizeof(int) * (matchingMachine.total_count + 1)));
    SAFE_CALL(cudaMemcpy(d_offset_attr, matchingMachine.offset_attr, sizeof(int) * (matchingMachine.total_count + 1),
                         cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMalloc(&d_flat_edges_indexes, sizeof(int) * matchingMachine.nof_sn));
    SAFE_CALL(cudaMemcpy(d_flat_edges_indexes, matchingMachine.flat_edges_indexes, sizeof(int) * matchingMachine.nof_sn,
                         cudaMemcpyHostToDevice));


    SAFE_CALL(cudaMalloc(&d_map_node_to_state, sizeof(int) * matchingMachine.nof_sn));
    SAFE_CALL(cudaMemcpy(d_map_node_to_state, matchingMachine.map_node_to_state, sizeof(int) * matchingMachine.nof_sn,
                         cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMalloc(&d_map_state_to_node, sizeof(int) * matchingMachine.nof_sn));
    SAFE_CALL(cudaMemcpy(d_map_state_to_node, matchingMachine.map_state_to_node, sizeof(int) * matchingMachine.nof_sn,
                         cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMalloc(&d_parent_state, sizeof(int) * matchingMachine.nof_sn));
    SAFE_CALL(cudaMemcpy(d_parent_state, matchingMachine.parent_state, sizeof(int) * matchingMachine.nof_sn,
                         cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMalloc(&d_parent_type, sizeof(int) * matchingMachine.nof_sn));
    SAFE_CALL(cudaMemcpy(d_parent_type, matchingMachine.parent_type, sizeof(int) * matchingMachine.nof_sn,
                         cudaMemcpyHostToDevice));
}


//reference


void reference_malloc(Graph &reference) {

    SAFE_CALL(cudaMalloc(&d_r_nof_nodes, sizeof(int) * 1));

    SAFE_CALL(cudaMalloc(&d_r_flatten_in_adj_list, sizeof(int) * reference.length_in_adj_list));

    SAFE_CALL(cudaMalloc(&d_r_offset_in_adj_list, sizeof(int) * (reference.nof_nodes + 1)));

    SAFE_CALL(cudaMalloc(&d_r_in_adj_sizes, sizeof(int) * (reference.nof_nodes)));

    SAFE_CALL(cudaMalloc(&d_r_flatten_out_adj_list, sizeof(int) * reference.length_out_adj_list));

    SAFE_CALL(cudaMalloc(&d_r_offset_out_adj_list, sizeof(int) * (reference.nof_nodes + 1)));

    SAFE_CALL(cudaMalloc(&d_r_out_adj_sizes, sizeof(int) * (reference.nof_nodes)));

    SAFE_CALL(cudaMalloc(&d_r_flatten_nodes_attr, sizeof(char) * reference.length_nodes_attrs));

    SAFE_CALL(cudaMalloc(&d_r_offset_nodes_attr, sizeof(int) * (reference.nof_nodes + 1)));

}

void reference_memcpy(Graph &reference){

    SAFE_CALL(cudaMemcpy(d_r_nof_nodes, &reference.nof_nodes, sizeof(int) * 1, cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMemcpy(d_r_flatten_in_adj_list, reference.flatten_in_adj_list,
                         sizeof(int) * reference.length_in_adj_list,
                         cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMemcpy(d_r_offset_in_adj_list, reference.offset_in_adj_list, sizeof(int) * (reference.nof_nodes + 1),
                         cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMemcpy(d_r_in_adj_sizes, reference.in_adj_sizes, sizeof(int) * (reference.nof_nodes),
                         cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMemcpy(d_r_flatten_out_adj_list, reference.flatten_out_adj_list,
                         sizeof(int) * reference.length_out_adj_list,
                         cudaMemcpyHostToDevice));

    SAFE_CALL(
            cudaMemcpy(d_r_offset_out_adj_list, reference.offset_out_adj_list, sizeof(int) * (reference.nof_nodes + 1),
                       cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMemcpy(d_r_out_adj_sizes, reference.out_adj_sizes, sizeof(int) * (reference.nof_nodes),
                         cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMemcpy(d_r_flatten_nodes_attr, reference.flatten_nodes_attr,
                         sizeof(char) * reference.length_nodes_attrs,
                         cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMemcpy(d_r_offset_nodes_attr, reference.offset_nodes_attr, sizeof(int) * (reference.nof_nodes + 1),
                         cudaMemcpyHostToDevice));
}

void query_malloc(Graph &query) {

    SAFE_CALL(cudaMalloc(&d_q_nof_nodes, sizeof(int) * 1));
    SAFE_CALL(cudaMemcpy(d_q_nof_nodes, &query.nof_nodes, sizeof(int), cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMalloc(&d_q_flatten_in_adj_list, sizeof(int) * query.length_in_adj_list));
    SAFE_CALL(cudaMemcpy(d_q_flatten_in_adj_list, query.flatten_in_adj_list, sizeof(int) * query.length_in_adj_list,
                         cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMalloc(&d_q_offset_in_adj_list, sizeof(int) * (query.nof_nodes + 1)));
    SAFE_CALL(cudaMemcpy(d_q_offset_in_adj_list, query.offset_in_adj_list, sizeof(int) * (query.nof_nodes + 1),
                         cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMalloc(&d_q_in_adj_sizes, sizeof(int) * (query.nof_nodes)));
    SAFE_CALL(
            cudaMemcpy(d_q_in_adj_sizes, query.in_adj_sizes, sizeof(int) * (query.nof_nodes), cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMalloc(&d_q_flatten_out_adj_list, sizeof(int) * query.length_out_adj_list));
    SAFE_CALL(cudaMemcpy(d_q_flatten_out_adj_list, query.flatten_out_adj_list, sizeof(int) * query.length_out_adj_list,
                         cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMalloc(&d_q_offset_out_adj_list, sizeof(int) * (query.nof_nodes + 1)));
    SAFE_CALL(cudaMemcpy(d_q_offset_out_adj_list, query.offset_out_adj_list, sizeof(int) * (query.nof_nodes + 1),
                         cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMalloc(&d_q_out_adj_sizes, sizeof(int) * (query.nof_nodes)));
    SAFE_CALL(
            cudaMemcpy(d_q_out_adj_sizes, query.out_adj_sizes, sizeof(int) * (query.nof_nodes),
                       cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMalloc(&d_q_flatten_nodes_attr, sizeof(char) * query.length_nodes_attrs));
    SAFE_CALL(cudaMemcpy(d_q_flatten_nodes_attr, query.flatten_nodes_attr, sizeof(char) * query.length_nodes_attrs,
                         cudaMemcpyHostToDevice));

    SAFE_CALL(cudaMalloc(&d_q_offset_nodes_attr, sizeof(int) * (query.nof_nodes + 1)));
    SAFE_CALL(cudaMemcpy(d_q_offset_nodes_attr, query.offset_nodes_attr, sizeof(int) * (query.nof_nodes + 1),
                         cudaMemcpyHostToDevice));
}

void reference_free() {
    SAFE_CALL(cudaFree(d_r_nof_nodes));
    SAFE_CALL(cudaFree(d_r_flatten_in_adj_list));
    SAFE_CALL(cudaFree(d_r_offset_in_adj_list));
    SAFE_CALL(cudaFree(d_r_in_adj_sizes));
    SAFE_CALL(cudaFree(d_r_flatten_out_adj_list));
    SAFE_CALL(cudaFree(d_r_offset_out_adj_list));
    SAFE_CALL(cudaFree(d_r_out_adj_sizes));
    SAFE_CALL(cudaFree(d_r_flatten_nodes_attr));
    SAFE_CALL(cudaFree(d_r_offset_nodes_attr));
    SAFE_CALL(cudaFree(d_r_out_adj_attrs));
}

void query_free() {
    SAFE_CALL(cudaFree(d_q_nof_nodes));
    SAFE_CALL(cudaFree(d_q_flatten_in_adj_list));
    SAFE_CALL(cudaFree(d_q_offset_in_adj_list));
    SAFE_CALL(cudaFree(d_q_in_adj_sizes));
    SAFE_CALL(cudaFree(d_q_flatten_out_adj_list));
    SAFE_CALL(cudaFree(d_q_offset_out_adj_list));
    SAFE_CALL(cudaFree(d_q_out_adj_sizes));
    SAFE_CALL(cudaFree(d_q_flatten_nodes_attr));
    SAFE_CALL(cudaFree(d_q_offset_nodes_attr));
}

void mama_free() {
    SAFE_CALL(cudaFree(d_nof_sn));
    SAFE_CALL(cudaFree(d_edges_sizes));
    SAFE_CALL(cudaFree(d_source));
    SAFE_CALL(cudaFree(d_target));
    SAFE_CALL(cudaFree(d_attr));
    SAFE_CALL(cudaFree(d_offset_attr));
    SAFE_CALL(cudaFree(d_flat_edges_indexes));
    SAFE_CALL(cudaFree(d_map_node_to_state));
    SAFE_CALL(cudaFree(d_map_state_to_node));
    SAFE_CALL(cudaFree(d_parent_state));
    SAFE_CALL(cudaFree(d_parent_type));
}


#endif