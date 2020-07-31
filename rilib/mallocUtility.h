#ifndef MALLOCUTILITY_H_
#define MALLOCUTILITY_H_


void reference_malloc(Graph &reference) {
    cudaMalloc(&d_r_nof_nodes, sizeof(int) * 1);
    cudaMemcpy(d_r_nof_nodes, &reference.nof_nodes, sizeof(int) * 1, cudaMemcpyHostToDevice);

    cudaMalloc(&d_r_flatten_in_adj_list, sizeof(int) * reference.length_in_adj_list);
    cudaMemcpy(d_r_flatten_in_adj_list, reference.flatten_in_adj_list, sizeof(int) * reference.length_in_adj_list,
               cudaMemcpyHostToDevice);

    cudaMalloc(&d_r_offset_in_adj_list, sizeof(int) * (reference.nof_nodes + 1));
    cudaMemcpy(d_r_offset_in_adj_list, reference.offset_in_adj_list, sizeof(int) * (reference.nof_nodes + 1),
               cudaMemcpyHostToDevice);

    cudaMalloc(&d_r_in_adj_sizes, sizeof(int) * (reference.nof_nodes));
    cudaMemcpy(d_r_in_adj_sizes, reference.in_adj_sizes, sizeof(int) * (reference.nof_nodes), cudaMemcpyHostToDevice);

    cudaMalloc(&d_r_flatten_out_adj_list, sizeof(int) * reference.length_out_adj_list);
    cudaMemcpy(d_r_flatten_out_adj_list, reference.flatten_out_adj_list, sizeof(int) * reference.length_out_adj_list,
               cudaMemcpyHostToDevice);

    cudaMalloc(&d_r_offset_out_adj_list, sizeof(int) * (reference.nof_nodes + 1));
    cudaMemcpy(d_r_offset_out_adj_list, reference.offset_out_adj_list, sizeof(int) * (reference.nof_nodes + 1),
               cudaMemcpyHostToDevice);

    cudaMalloc(&d_r_out_adj_sizes, sizeof(int) * (reference.nof_nodes));
    cudaMemcpy(d_r_out_adj_sizes, reference.out_adj_sizes, sizeof(int) * (reference.nof_nodes), cudaMemcpyHostToDevice);

    cudaMalloc(&d_r_flatten_nodes_attr, sizeof(char) * reference.length_nodes_attrs);
    cudaMemcpy(d_r_flatten_nodes_attr, reference.flatten_nodes_attr, sizeof(char) * reference.length_nodes_attrs,
               cudaMemcpyHostToDevice);

    cudaMalloc(&d_r_offset_nodes_attr, sizeof(int) * (reference.nof_nodes + 1));
    cudaMemcpy(d_r_offset_nodes_attr, reference.offset_nodes_attr, sizeof(int) * (reference.nof_nodes + 1),
               cudaMemcpyHostToDevice);
}

void query_malloc(Graph &query) {
    cudaMalloc(&d_q_nof_nodes, sizeof(int) * 1);
    cudaMemcpy(d_q_nof_nodes, &query.nof_nodes, sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_q_flatten_in_adj_list, sizeof(int) * query.length_in_adj_list);
    cudaMemcpy(d_q_flatten_in_adj_list, query.flatten_in_adj_list, sizeof(int) * query.length_in_adj_list,
               cudaMemcpyHostToDevice);

    cudaMalloc(&d_q_offset_in_adj_list, sizeof(int) * (query.nof_nodes + 1));
    cudaMemcpy(d_q_offset_in_adj_list, query.offset_in_adj_list, sizeof(int) * (query.nof_nodes + 1),
               cudaMemcpyHostToDevice);

    cudaMalloc(&d_q_in_adj_sizes, sizeof(int) * (query.nof_nodes));
    cudaMemcpy(d_q_in_adj_sizes, query.in_adj_sizes, sizeof(int) * (query.nof_nodes), cudaMemcpyHostToDevice);

    cudaMalloc(&d_q_flatten_out_adj_list, sizeof(int) * query.length_out_adj_list);
    cudaMemcpy(d_q_flatten_out_adj_list, query.flatten_out_adj_list, sizeof(int) * query.length_out_adj_list,
               cudaMemcpyHostToDevice);

    cudaMalloc(&d_q_offset_out_adj_list, sizeof(int) * (query.nof_nodes + 1));
    cudaMemcpy(d_q_offset_out_adj_list, query.offset_out_adj_list, sizeof(int) * (query.nof_nodes + 1),
               cudaMemcpyHostToDevice);

    cudaMalloc(&d_q_out_adj_sizes, sizeof(int) * (query.nof_nodes));
    cudaMemcpy(d_q_out_adj_sizes, query.out_adj_sizes, sizeof(int) * (query.nof_nodes), cudaMemcpyHostToDevice);

    cudaMalloc(&d_q_flatten_nodes_attr, sizeof(char) * query.length_nodes_attrs);
    cudaMemcpy(d_q_flatten_nodes_attr, query.flatten_nodes_attr, sizeof(char) * query.length_nodes_attrs,
               cudaMemcpyHostToDevice);

    cudaMalloc(&d_q_offset_nodes_attr, sizeof(int) * (query.nof_nodes + 1));
    cudaMemcpy(d_q_offset_nodes_attr, query.offset_nodes_attr, sizeof(int) * (query.nof_nodes + 1),
               cudaMemcpyHostToDevice);
}

void mama_malloc(MatchingMachine &mama) {

    MaMaEdge *d_flat_edges;


    cudaMalloc(&d_nof_sn, sizeof(int));
    cudaMemcpy(d_nof_sn, &mama.nof_sn, sizeof(int),
               cudaMemcpyHostToDevice);

    cudaMalloc(&d_edges_sizes, sizeof(int) * mama.nof_sn);
    cudaMemcpy(d_edges_sizes, mama.edges_sizes, sizeof(int) * mama.nof_sn,
               cudaMemcpyHostToDevice);

    /*
    cudaMalloc((void **)d_flat_edges, sizeof(MaMaEdge) * mama.nof_sn);
    cudaMemcpy(d_flat_edges, mama.flat_edges, sizeof(MaMaEdge) * mama.nof_sn, cudaMemcpyHostToDevice);

    for (int i = 0; i < mama.nof_sn; i++) {
        cudaMalloc((void **)&source[i], sizeof(int));
        cudaMemcpy(source[i], mama.flat_edges[i].source, sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc((void **)&target[i], sizeof(int));
        cudaMemcpy(target[i], mama.flat_edges[i].target, sizeof(int), cudaMemcpyHostToDevice);

        cudaMalloc((void ***)&attr[i], sizeof(char));
        cudaMemcpy(source[i], mama.flat_edges[i].attr, sizeof(int), cudaMemcpyHostToDevice);
    }

    */


    cudaMalloc(&d_flat_edges_indexes, sizeof(int) * mama.nof_sn);
    cudaMemcpy(d_flat_edges_indexes, mama.flat_edges_indexes, sizeof(int) * mama.nof_sn,
               cudaMemcpyHostToDevice);



    cudaMalloc(&d_map_node_to_state, sizeof(int) * mama.nof_sn);
    cudaMemcpy(d_map_node_to_state, mama.map_node_to_state, sizeof(int) * mama.nof_sn,
               cudaMemcpyHostToDevice);

    cudaMalloc(&d_map_state_to_node, sizeof(int) * mama.nof_sn);
    cudaMemcpy(d_map_state_to_node, mama.map_state_to_node, sizeof(int) * mama.nof_sn,
               cudaMemcpyHostToDevice);

    cudaMalloc(&d_parent_state, sizeof(int) * mama.nof_sn);
    cudaMemcpy(d_parent_state, mama.parent_state, sizeof(int) * mama.nof_sn,
               cudaMemcpyHostToDevice);

    cudaMalloc(&d_parent_type, sizeof(int) * mama.nof_sn);
    cudaMemcpy(d_parent_type, mama.parent_type, sizeof(int) * mama.nof_sn,
               cudaMemcpyHostToDevice);

}


#endif