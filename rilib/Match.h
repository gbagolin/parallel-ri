/*
 * Match.h
 *
 *  Created on: Aug 3, 2012
 *      Author: vbonnici
 */
/*
Copyright (c) 2014 by Rosalba Giugno

This library contains portions of other open source products covered by separate
licenses. Please see the corresponding source files for specific terms.

RI is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef MATCH_H_
#define MATCH_H_

#include "MatchListener.h"
#include "AttributeComparator.h"
#include "IsoGISolver.h"
#include "SubGISolver.h"
#include "InducedSubGISolver.h"
#include "kernel.h"
#include "flatter.h"
//#include "mallocUtility.h"

namespace rilib {

    using namespace rilib;


    void test(int *arr, int length, int count) {
        printf("Case #%d: ", count);
        for (int i = 0; i < length; ++i) {
            printf("%d ", arr[i]);
        }
        printf("\n");
    }

    enum MATCH_TYPE {
        MT_ISO, MT_INDSUB, MT_MONO
    };

    int count = 0;

    void match(
            Graph &reference,
            Graph &query,
            MatchingMachine &matchingMachine,
            MatchListener &matchListener,
            MATCH_TYPE matchType,
            AttributeComparator &nodeComparator,
            AttributeComparator &edgeComparator,
            long *steps,
            long *triedcouples,
            long *matchedcouples,
            GRAPH_FILE_TYPE file_type,
            bool *printToConsole,
            long *matchCount) {

        int comparatorType;
        switch (file_type) {
            case GFT_GFU:
            case GFT_GFD:
                // only nodes have labels and they are strings
                comparatorType = 0;
                //takeNodeLabels = true;
                break;
            case GFT_GFDA:
                comparatorType = 1;
                //takeNodeLabels = true;
                break;
            case GFT_EGFU:
            case GFT_EGFD:
                //labels on nodes and edges, both of them are strings
                comparatorType = 2;
                //takeNodeLabels = true;
                //takeEdgesLabels = true;
                break;
            case GFT_VFU:
                //no labels
                comparatorType = 1;
                break;

        }
        switch (matchType) {
            case MT_ISO:
                IsoGISolver *solver1;
                solver1 = new IsoGISolver(matchingMachine, reference, query, nodeComparator, edgeComparator,
                                          matchListener, 0);
                solver1->solve();

                *steps = solver1->steps;
                *triedcouples = solver1->triedcouples;
                *matchedcouples = solver1->matchedcouples;

                delete solver1;
                break;
            case MT_INDSUB:
                InducedSubGISolver *solver2;
                solver2 = new InducedSubGISolver(matchingMachine, reference, query, nodeComparator, edgeComparator,
                                                 matchListener, 1);

                solver2->solve();

                *steps = solver2->steps;
                *triedcouples = solver2->triedcouples;
                *matchedcouples = solver2->matchedcouples;

                delete solver2;

                break;
            case MT_MONO:
                flatterGraph(&reference);

                cudaDeviceReset();

                bool *d_printToConsole;
                long *d_matchCount;
                int *d_comparatorType;
                long *d_steps;
                long *d_triedcouples;
                long *d_matchedcouples;

                cudaMalloc(&d_printToConsole, sizeof(bool));
                cudaMemcpy(d_printToConsole, printToConsole, sizeof(bool),
                           cudaMemcpyHostToDevice);

                cudaMalloc(&d_matchCount, sizeof(long));
                cudaMemcpy(d_matchCount, matchCount, sizeof(long),
                           cudaMemcpyHostToDevice);

                cudaMalloc(&d_comparatorType, sizeof(int));
                cudaMemcpy(d_comparatorType, &comparatorType, sizeof(int),
                           cudaMemcpyHostToDevice);

                cudaMalloc(&d_steps, sizeof(long));
                cudaMemcpy(d_steps, steps, sizeof(long),
                           cudaMemcpyHostToDevice);

                cudaMalloc(&d_triedcouples, sizeof(long));
                cudaMemcpy(d_triedcouples, triedcouples, sizeof(long),
                           cudaMemcpyHostToDevice);

                cudaMalloc(&d_matchedcouples, sizeof(long));
                cudaMemcpy(d_matchedcouples, matchedcouples, sizeof(long),
                           cudaMemcpyHostToDevice);
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


                cudaMalloc(&d_nof_sn, sizeof(int));
                cudaMemcpy(d_nof_sn, &matchingMachine.nof_sn, sizeof(int),
                           cudaMemcpyHostToDevice);

                cudaMalloc(&d_edges_sizes, sizeof(int) * matchingMachine.nof_sn);
                cudaMemcpy(d_edges_sizes, matchingMachine.edges_sizes, sizeof(int) * matchingMachine.nof_sn,
                           cudaMemcpyHostToDevice);

                cudaMalloc(&d_source, sizeof(int) * matchingMachine.total_count);
                cudaMemcpy(d_source, matchingMachine.source, sizeof(int) * matchingMachine.total_count,
                           cudaMemcpyHostToDevice);

                cudaMalloc(&d_target, sizeof(int) * matchingMachine.total_count);
                cudaMemcpy(d_target, matchingMachine.target, sizeof(int) * matchingMachine.total_count,
                           cudaMemcpyHostToDevice);

                cudaMalloc(&d_attr, sizeof(char) * (matchingMachine.length_string + 1));
                cudaMemcpy(d_attr, matchingMachine.attr, sizeof(char) * matchingMachine.length_string,
                           cudaMemcpyHostToDevice);

                cudaMalloc(&d_offset_attr, sizeof(int) * (matchingMachine.total_count + 1));
                cudaMemcpy(d_offset_attr, matchingMachine.offset_attr, sizeof(int) * (matchingMachine.total_count + 1),
                           cudaMemcpyHostToDevice);

                cudaMalloc(&d_flat_edges_indexes, sizeof(int) * matchingMachine.nof_sn);
                cudaMemcpy(d_flat_edges_indexes, matchingMachine.flat_edges_indexes, sizeof(int) * matchingMachine.nof_sn,
                           cudaMemcpyHostToDevice);


                cudaMalloc(&d_map_node_to_state, sizeof(int) * matchingMachine.nof_sn);
                cudaMemcpy(d_map_node_to_state, matchingMachine.map_node_to_state, sizeof(int) * matchingMachine.nof_sn,
                           cudaMemcpyHostToDevice);

                cudaMalloc(&d_map_state_to_node, sizeof(int) * matchingMachine.nof_sn);
                cudaMemcpy(d_map_state_to_node, matchingMachine.map_state_to_node, sizeof(int) * matchingMachine.nof_sn,
                           cudaMemcpyHostToDevice);

                cudaMalloc(&d_parent_state, sizeof(int) * matchingMachine.nof_sn);
                cudaMemcpy(d_parent_state, matchingMachine.parent_state, sizeof(int) * matchingMachine.nof_sn,
                           cudaMemcpyHostToDevice);

                cudaMalloc(&d_parent_type, sizeof(int) * matchingMachine.nof_sn);
                cudaMemcpy(d_parent_type, matchingMachine.parent_type, sizeof(int) * matchingMachine.nof_sn,
                           cudaMemcpyHostToDevice);

                //reference

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
                /*
                test(&matchingMachine.nof_sn, 1, count++);
                test(matchingMachine.edges_sizes, matchingMachine.nof_sn, count++);
                test(matchingMachine.source, matchingMachine.total_count, count++);
                */
                subsolver<<<1, 1>>>(
                //in_out
                d_printToConsole,
                        d_matchCount,
                        d_comparatorType,
                        d_steps,
                        d_triedcouples,
                        d_matchedcouples,
                        //mama
                        d_nof_sn,
                        d_edges_sizes,
                        d_flat_edges_indexes,
                        d_source,
                        d_target,
                        d_attr,
                        d_offset_attr,
                        d_map_node_to_state,
                        d_map_state_to_node,
                        d_parent_state,
                        d_parent_type,
                        //reference
                        d_r_nof_nodes,
                        d_r_flatten_in_adj_list,
                        d_r_offset_in_adj_list,
                        d_r_in_adj_sizes,
                        d_r_flatten_out_adj_list,
                        d_r_offset_out_adj_list,
                        d_r_out_adj_sizes,
                        d_r_flatten_nodes_attr,
                        d_r_offset_nodes_attr,
                        d_r_out_adj_attrs,
                        //query
                        d_q_nof_nodes,
                        d_q_flatten_in_adj_list,
                        d_q_offset_in_adj_list,
                        d_q_in_adj_sizes,
                        d_q_flatten_out_adj_list,
                        d_q_offset_out_adj_list,
                        d_q_out_adj_sizes,
                        d_q_flatten_nodes_attr,
                        d_q_offset_nodes_attr);


                cudaMemcpy(steps, d_steps, sizeof(long), cudaMemcpyDeviceToHost);
                cudaMemcpy(triedcouples, d_triedcouples, sizeof(long), cudaMemcpyDeviceToHost);
                cudaMemcpy(matchedcouples, d_matchedcouples, sizeof(long), cudaMemcpyDeviceToHost);
                cudaMemcpy(matchCount, d_matchCount, sizeof(long), cudaMemcpyDeviceToHost);

                cudaDeviceReset();
                //solver3->solve();
                /*
                    steps = solver3->steps;
                    *triedcouples = solver3->triedcouples;
                    *matchedcouples = solver3->matchedcouples;

                    delete solver3;
                */
                break;
        }


    }

};


#endif /* MATCH_H_ */
