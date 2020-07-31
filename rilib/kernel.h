/*
 * SubGISolver.h
 *
 *  Created on: Aug 4, 2012
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

#ifndef KERNEL_H_
#define KERNEL_H_


#include "MatchingMachine.h"
#include "AttributeComparator.h"
#include "substitutors.h"
#include "Graph.h"
#include <set>


__device__
bool
nodeSubCheck(int si, int ci, int *map_state_to_node, int *r_out_adj_sizes, int *q_out_adj_sizes, int *r_in_adj_sizes,
             int *q_in_adj_sizes, void *r_nodes_attrs, int *r_offset_nodes_attr, void *q_nodes_attrs,
             int *q_offset_nodes_attr, int comparatorType) {

    if (r_out_adj_sizes[ci] >= q_out_adj_sizes[map_state_to_node[si]]
        && r_in_adj_sizes[ci] >= q_in_adj_sizes[map_state_to_node[si]]) {

        int r_start = r_offset_nodes_attr[ci];
        int r_end = r_offset_nodes_attr[ci + 1];
        int q_start = q_offset_nodes_attr[map_state_to_node[si]];
        int q_end = q_offset_nodes_attr[map_state_to_node[si] + 1];
        void *q_str_attr = getSubString(q_nodes_attrs, q_start, q_end);
        void *r_str_attr = getSubString(r_nodes_attrs, r_start, r_end);
        return nodeComparator(comparatorType, r_str_attr, q_str_attr);
    }
    return false;
}

__device__
bool edgesSubCheck(int si, int ci, int *solution, bool *matched, int *edges_sizes, int *source, int *target, void *attr,
                   int *offset_attr,
                   int *m_flat_edges_indexes, int *r_out_adj_sizes, int *r_out_adj_list, int *r_offset_out_adj_list,
                   int comparatorType) {

    int tmp_source, tmp_target;
    int ii;
    for (int me = 0; me < edges_sizes[si]; me++) {
        //printf("siamo qui dentro");
        tmp_source = solution[source[m_flat_edges_indexes[si] + me]];
        tmp_target = solution[target[m_flat_edges_indexes[si] + me]];

        for (ii = 0; ii < r_out_adj_sizes[tmp_source]; ii++) {
            int index = r_offset_out_adj_list[tmp_source] + ii;
            if (r_out_adj_list[index] == tmp_target) {
//					if(! edgeComparator.compare(rgraph.out_adj_attrs[source][ii],  mama.edges[si][me].attr)){
//						return false;
//					}
//					else{
//						break;
//					}
                int start = offset_attr[m_flat_edges_indexes[si] + me];
                int end = offset_attr[m_flat_edges_indexes[si] + me + 1];
                void *str_attr = getSubString(attr, start, end);
                if (edgeComparator(comparatorType, NULL,
                                   str_attr)) {
                    break;
                }
            }
        }
        if (ii >= r_out_adj_sizes[tmp_source]) {
            return false;
        }
    }
    return true;
}

__global__
void subsolver(
        //in_out
        bool *printToConsole,
        long *matchCount,
        int *type_comparator,
        long *steps,
        long *triedcouples,
        long *matchedcouples,
        //mama
        int *nof_sn,
        int *edges_sizes,
        int *flat_edges_indexes,
        int *source,
        int *target,
        void *attr,
        int *offset_attr,
        int *map_node_to_state,
        int *map_state_to_node,
        int *parent_state,
        int *parent_type,
        //reference
        int *r_nof_nodes,
        int *r_in_adj_list, //flatted
        int *r_offset_in_adj_list,
        int *r_in_adj_sizes,
        int *r_out_adj_list,//flatted
        int *r_offset_out_adj_list,
        int *r_out_adj_sizes,
        void *r_nodes_attrs,//flatted
        int *r_offset_nodes_attr,
        void *r_out_adj_attrs,
        //query
        int *q_nof_nodes,
        int *q_in_adj_list,//flatted
        int *q_offset_in_adj_list,
        int *q_in_adj_sizes,
        int *q_out_adj_list, //flatted
        int *q_offset_q_out_adj_list,
        int *q_out_adj_sizes,
        void *q_nodes_attrs, //flatted
        int *q_offset_nodes_attr

) {
    printf("sono il kernel");
    /*
    printf("Kernel: printToConsole %d\n", *printToConsole);
    printf("Kernel: matchCount %d\n", *matchCount);
    printf("Kernel: type_comparator %d\n", *type_comparator);
    printf("Kernel: steps %d\n", *steps);
    printf("Kernel: triedcouples %d\n", *triedcouples);
    printf("Kernel: matchedcouples %d\n", *matchedcouples);


    printf("Kernel: nof_sn %d\n", *nof_sn);

    printf("Kernel: edges_sizes %d\n", edges_sizes[1]);

    printf("Kernel: source %d\n", source[2]);

    printf("Kernel: r_nof_nodes %d\n", r_nof_nodes[0]);

    printf("Kernel: q_nof_nodes %d\n", q_nof_nodes[0]);

    */

    //printf("ciaoooo\n");

    //printf("qui ci entro");

    //printf("%s\n", (char *)(attr));
    
    int ii;
    int *listAllRef = new int[*r_nof_nodes];
    for (
            ii = 0;
            ii < *
                    r_nof_nodes;
            ii++)
        listAllRef[ii] =
                ii;

    int **candidates = new int *[*nof_sn];                            //indexed by state_id
    int *candidatesIT = new int[*nof_sn];                            //indexed by state_id
    int *candidatesSize = new int[*nof_sn];                            //indexed by state_id
    int *solution = new int[*nof_sn];
//indexed by state_id
    for (
            ii = 0;
            ii < *
                    nof_sn;
            ii++) {
        solution[ii] = -1;
    }


    bool cmatched[10][10];

//printf("%i, %i, %i\n", *nof_sn, *r_nof_nodes, *q_nof_nodes);

    bool matched[1000];

    candidates[0] = listAllRef;
    candidatesSize[0] = *
            r_nof_nodes;
    candidatesIT[0] = -1;

    int psi = -1;
    int si = 0;
    int ci = -1;
    int sip1;


    while (si != -1) {


//steps++;

        if (psi >= si) {
            matched[solution[si]] = false;
        }

        ci = -1;
        candidatesIT[si]++;

        while (candidatesIT[si] < candidatesSize[si]) {
            //printf("ok");
//triedcouples++;

            ci = candidates[si][candidatesIT[si]];
            solution[si] = ci;

//				std::cout<<"[ "<<map_state_to_node[si]<<" , "<<ci<<" ]\n";
//				if(matched[ci]) std::cout<<"fails on alldiff\n";
//				if(!nodeCheck(si,ci, map_state_to_node)) std::cout<<"fails on node label\n";
//				if(!(edgesCheck(si, ci, solution, matched))) std::cout<<"fails on edges \n";

//MT_ISO

            /*
            if (!matched[ci]) {

                if (cmatched[si][ci] == false) {

                    if (nodeSubCheck(si, ci, map_state_to_node, r_out_adj_sizes, q_out_adj_sizes, r_in_adj_sizes,
                                     q_in_adj_sizes, r_nodes_attrs, r_offset_nodes_attr, q_nodes_attrs,
                                     q_offset_nodes_attr,
                                     *type_comparator
                    )) {
                        printf("terzo if");
                    }

                }

            }

            */

            if ((!matched[ci])
                && (cmatched[si][ci] == false)
                &&
                nodeSubCheck(si, ci, map_state_to_node, r_out_adj_sizes, q_out_adj_sizes, r_in_adj_sizes,
                             q_in_adj_sizes, r_nodes_attrs, r_offset_nodes_attr, q_nodes_attrs, q_offset_nodes_attr,
                             *type_comparator
                )
                &&
                edgesSubCheck(si, ci, solution, matched, edges_sizes, source, target, attr, offset_attr,
                              flat_edges_indexes, r_out_adj_sizes,
                              r_out_adj_list, r_offset_out_adj_list, *type_comparator
                )
                    ) {
//printf("sono qui");
                break;
            } else {
                ci = -1;
            }


            candidatesIT[si]++;
        }

        if (ci == -1) {
            psi = si;
            for (
                    int i = 0;
                    i < *
                            r_nof_nodes;
                    i++) {
                cmatched[si][i] = false;
            }
//cmatched[si].clear();
            si--;
        } else {
            cmatched[si][ci] = true;

            (*matchedcouples)++;

            if (si == *nof_sn - 1) {
                matchListener(printToConsole, matchCount, *nof_sn, map_state_to_node, solution
                );
#ifdef FIRST_MATCH_ONLY
                si = -1;
#endif
                psi = si;
            } else {
                matched[solution[si]] = true;
                sip1 = si + 1;
                if (parent_type[sip1] == 2) {
                    candidates[sip1] =
                            listAllRef;
                    candidatesSize[sip1] = *
                            r_nof_nodes;
                } else {
                    if (parent_type[sip1] == 0) {

                        int r_start_in_adj_list = r_offset_in_adj_list[solution[parent_state[sip1]]];
                        int r_end_in_adj_list = r_offset_in_adj_list[solution[parent_state[sip1]] + 1];
                        int arr_len = r_end_in_adj_list - r_start_in_adj_list;
                        int *arr_r_in_adj_list = (int *) malloc(arr_len * sizeof(int));
                        int pos = 0;
                        for (
                                int i = r_start_in_adj_list;
                                i < r_end_in_adj_list;
                                ++i) {
                            arr_r_in_adj_list[pos] = r_in_adj_list[i];
                            pos++;
                        }
                        candidates[sip1] =
                                arr_r_in_adj_list;
                        candidatesSize[sip1] =
                                arr_len;
                       // printf("%d", arr_len == r_in_adj_sizes[solution[parent_state[sip1]]]);
                    } else {//(parent_type[sip1] == MAMA_PARENTTYPE::PARENTTYPE_OUT)
                        printf("sono qui");
                        int r_start_out_adj_list = r_offset_out_adj_list[solution[parent_state[sip1]]];
                        int r_end_out_adj_list = r_offset_out_adj_list[solution[parent_state[sip1]] + 1];
                        int arr_len = r_end_out_adj_list - r_start_out_adj_list;
                        int *arr_r_out_adj_list = (int *) malloc(arr_len * sizeof(int));
                        int pos = 0;
                        for (
                                int i = r_start_out_adj_list;
                                i < r_end_out_adj_list;
                                ++i) {
                            arr_r_out_adj_list[pos] = r_out_adj_list[i];
                            pos++;
                        }
                        candidates[sip1] =
                                arr_r_out_adj_list;
                        candidatesSize[sip1] =
                                arr_len;
                    }
                }
                candidatesIT[si + 1] = -1;

                psi = si;
                si++;
            }
        }
    }

// memory cleanup
    free(matched);
//delete[] cmatched;
    delete[]
            solution;
    delete[]
            candidatesSize;
    delete[]
            candidatesIT;
    delete[]
            candidates;
    delete[]
            listAllRef;


}


#endif 