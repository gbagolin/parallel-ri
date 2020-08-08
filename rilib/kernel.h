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
#define MAX_SIZE 35000

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

    //   printf("node_sub_check\n");
    //  printf("q_nodes_attr : %s\n", q_nodes_attrs);
    //  printf("r_nodes_attr: %s\n", r_nodes_attrs);

    if (r_out_adj_sizes[ci] >= q_out_adj_sizes[map_state_to_node[si]]
        && r_in_adj_sizes[ci] >= q_in_adj_sizes[map_state_to_node[si]]) {

        //  printf("IF node_sub_check\n");

        int r_start = r_offset_nodes_attr[ci];
        int r_end = r_offset_nodes_attr[ci + 1];
        int q_start = q_offset_nodes_attr[map_state_to_node[si]];
        int q_end = q_offset_nodes_attr[map_state_to_node[si] + 1];
        //  printf("q: %d, %d\n", q_start,q_end);
        // printf("r: %d, %d\n", r_start,r_end);
        void *q_str_attr = getSubString(q_nodes_attrs, q_start, q_end);
        // printf("q_str_attr: %s\n", q_str_attr);
        void *r_str_attr = getSubString(r_nodes_attrs, r_start, r_end);
        //  printf("r_str_attr: %s\n", r_str_attr);
        //  printf("END IF node_sub_check\n");
        return nodeComparator(comparatorType, r_str_attr, q_str_attr);
    }
    return false;
}

__device__
bool edgesSubCheck(int si, int ci, int *solution, bool *matched, int *edges_sizes, int *source, int *target, void *attr,
                   int *offset_attr,
                   int *m_flat_edges_indexes, int *r_out_adj_sizes, int *r_out_adj_list, int *r_offset_out_adj_list,
                   int comparatorType) {
    if (comparatorType != 2) {
        return true;
    } else {
        int tmp_source, tmp_target;
        int ii;
        for (int me = 0; me < edges_sizes[si]; me++) {
            // printf("siamo qui dentro");
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

}

__global__
void subsolver(
        //test
        //in_out
        bool *d_printToConsole,
        long *matchCount,
        int *d_type_comparator,
        long *matchedcouples,
        //mama
        int *d_nof_sn,
        int *edges_sizes,
        int *flat_edges_indexes,
        int *source,
        int *target,
        void *attr,
        int *offset_attr,
        int *map_state_to_node,
        int *parent_state,
        int *parent_type,
        //reference
        int *d_r_nof_nodes,
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
        int *q_in_adj_sizes,
        int *q_out_adj_sizes,
        void *q_nodes_attrs, //flatted
        int *q_offset_nodes_attr

) {

    int d[1];
    d[0] = threadIdx.x + blockDim.x * blockIdx.x;
   // extern __shared__ int testing2[];

    __shared__ int r_nof_nodes;
    __shared__ bool printToConsole;
    __shared__ int type_comparator;
    __shared__ int nof_sn;
   


    
    if (threadIdx.x==0){
        r_nof_nodes = *d_r_nof_nodes;
        printToConsole = *d_printToConsole;
        type_comparator = *d_type_comparator;
        nof_sn =*d_nof_sn;
       
    }
    __syncthreads();
    if (d[0] >= r_nof_nodes) {
        return;
    } else {
        int ii;
        
        int **candidates = new int *[nof_sn];                            //indexed by state_id
        int *candidatesIT = new int[nof_sn];                            //indexed by state_id
        int *candidatesSize = new int[nof_sn];                            //indexed by state_id
        int *solution = new int[nof_sn];
//indexed by state_id
        for (
                ii = 0;
                ii < nof_sn;
                ii++) {
            solution[ii] = -1;
        }

        bool *matched = (bool *) malloc(sizeof(bool) * (r_nof_nodes));
        memset(matched, false, sizeof(bool) * (r_nof_nodes));
        
        candidates[0] = d;
        candidatesSize[0] = 1;
        candidatesIT[0] = -1;

        int psi = -1;
        int si = 0;
        int ci = -1;
        int sip1;


        while (si != -1) {

            //    printf("sono dentro il while\n");

//steps++;
            
            if (psi >= si) {
                //printf("psi >= si\n");
                matched[solution[si]] = false;
            }


            ci = -1;
            candidatesIT[si]++;
            //printf("candidatesSize[si]: %d\n", candidatesSize[si]);
            while (candidatesIT[si] < candidatesSize[si]) {
                //    printf("candidatesIT[si] < candidatesSize[si]\n");
                //  printf("ok");
//triedcouples++;
                //  printf("0\n");
                ci = candidates[si][candidatesIT[si]];

                solution[si] = ci;
                //  printf("1\n");
                if ((!matched[ci])
                    && 
                    nodeSubCheck(si, ci, map_state_to_node, r_out_adj_sizes, q_out_adj_sizes, r_in_adj_sizes,
                                 q_in_adj_sizes, r_nodes_attrs, r_offset_nodes_attr, q_nodes_attrs, q_offset_nodes_attr,
                                 type_comparator
                    )
                    &&
                    edgesSubCheck(si, ci, solution, matched, edges_sizes, source, target, attr, offset_attr,
                                  flat_edges_indexes, r_out_adj_sizes,
                                  r_out_adj_list, r_offset_out_adj_list, type_comparator
                    )
                        ) {
                    //printf("2\n");
                    break;
                } else {
                    // printf("3\n");
                    //  printf("ci\n");
                    ci = -1;
                }


                candidatesIT[si]++;

            }
            
            if (ci == -1) {
                psi = si;
               
                si--;
            }
            else{
                atomicAdd((int *) matchedcouples, 1);
                if (si == nof_sn - 1) {
                    matchListener(&printToConsole, matchCount, nof_sn, map_state_to_node, solution);        


                    #ifdef FIRST_MATCH_ONLY
                    si = -1;
                    #endif
                    psi = si;
                } else {
                    matched[solution[si]] = true;
                    sip1 = si + 1;

                    
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
                        //  printf("sono qui dentro\n");
                        
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

                        candidatesSize[sip1] = arr_len;
                    }

                    candidatesIT[si + 1] = -1;

                    psi = si;
                    si++;
                }
            }
        }

// memory cleanup

        free(matched);  
        free(solution);
        free(candidatesSize);
    free(candidatesIT);
        delete[]
                candidates;
        /*
        delete[]
                listAllRef;
        */
    }


}


#endif 