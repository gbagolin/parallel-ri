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

#ifndef PMATCH_H_
#define PMATCH_H_

#include "isosolver.h"
#include "subsolver.h"
#include "indsolver.h"
#include "flatter.h"
#include "mallocUtility.h"
#include "CheckError.cuh"

#define BLOCK_DIM 1024

namespace rilib {

    using namespace rilib;



    void test(int *arr, int length, int count) {
        printf("Case #%d: ", count);
        for (int i = 0; i < length; ++i) {
            printf("%d ", arr[i]);
        }
        printf("\n");
    }



    void parallel_match(
            Graph &reference,
            Graph &query,
            MatchingMachine &matchingMachine,
            MATCH_TYPE matchType,
            long *steps,
            long *triedcouples,
            long *matchedcouples,
            GRAPH_FILE_TYPE file_type,
            bool *printToConsole,
            long *matchCount) {      
        in_out_memcpy(&comparatorType, matchedcouples, printToConsole, matchCount);
        device_memory_s=start_time();
        size_t size = (query.nof_nodes*reference.nof_nodes)*reference.nof_nodes ;
        dim3 DimGrid(reference.nof_nodes / BLOCK_DIM, 1, 1);
        if (reference.nof_nodes % BLOCK_DIM) DimGrid.x++;
        dim3 DimBlock(BLOCK_DIM, 1, 1);     
        if(reference.nof_nodes < 10000)           
        cudaDeviceSetLimit(cudaLimitMallocHeapSize,0.3*size);
        if(reference.nof_nodes < 20000)           
        cudaDeviceSetLimit(cudaLimitMallocHeapSize,0.4*size);
        if(reference.nof_nodes > 20000)           
        cudaDeviceSetLimit(cudaLimitMallocHeapSize,0.7*size);
        device_memory_t += end_time(device_memory_s);

        switch (matchType) {
            case MT_ISO:
                match_s = start_time();
                isosolver<<<DimGrid , DimBlock>>>(
                        //in_out
                        d_printToConsole,
                        d_matchCount,
                        d_comparatorType,
                        d_matchedcouples,
                        //mama
                        d_nof_sn,
                        d_edges_sizes,
                        d_o_edges_sizes,
                        d_i_edges_sizes,
                        d_flat_edges_indexes,
                        d_source,
                        d_target,
                        d_attr,
                        d_offset_attr,
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
                        d_r_offset_out_adj_attrs,
                        d_r_indexes_out_adj_attrs,
                        //query
                        d_q_in_adj_sizes,
                        d_q_out_adj_sizes,
                        d_q_flatten_nodes_attr,
                        d_q_offset_nodes_attr);
                match_t += end_time(match_s);
                break;
     
            case MT_INDSUB:      
                match_s = start_time();
                indsolver<<<DimGrid , DimBlock>>>(
                        //in_out
                        d_printToConsole,
                        d_matchCount,
                        d_comparatorType,
                        d_matchedcouples,
                        //mama
                        d_nof_sn,
                        d_edges_sizes,
                        d_o_edges_sizes,
                        d_i_edges_sizes,
                        d_flat_edges_indexes,
                        d_source,
                        d_target,
                        d_attr,
                        d_offset_attr,
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
                        d_r_offset_out_adj_attrs,
                        d_r_indexes_out_adj_attrs,
                        //query
                        d_q_in_adj_sizes,
                        d_q_out_adj_sizes,
                        d_q_flatten_nodes_attr,
                        d_q_offset_nodes_attr);
                match_t += end_time(match_s);              
                break;
        
            case MT_MONO:    
                match_s = start_time();
                subsolver<<<DimGrid , DimBlock>>>(
                        //in_out
                        d_printToConsole,
                        d_matchCount,
                        d_comparatorType,
                        d_matchedcouples,
                        //mama
                        d_nof_sn,
                        d_edges_sizes,
                        d_o_edges_sizes,
                        d_i_edges_sizes,
                        d_flat_edges_indexes,
                        d_source,
                        d_target,
                        d_attr,
                        d_offset_attr,
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
                        d_r_offset_out_adj_attrs,
                        d_r_indexes_out_adj_attrs,
                        //query
                        d_q_in_adj_sizes,
                        d_q_out_adj_sizes,
                        d_q_flatten_nodes_attr,
                        d_q_offset_nodes_attr);
                match_t += end_time(match_s);
                break;
        }
        cudaDeviceSynchronize();
        SAFE_CALL(cudaMemcpy(matchedcouples, d_matchedcouples, sizeof(long), cudaMemcpyDeviceToHost));
        SAFE_CALL(cudaMemcpy(matchCount, d_matchCount, sizeof(long), cudaMemcpyDeviceToHost));


    }

};


#endif /* MATCH_H_ */
