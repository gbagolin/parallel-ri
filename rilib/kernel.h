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


	void doubleFlatter(void ** ptr, void * retPtr, int * indexes){
		
	}


    __host__ __device__ 
    bool nodeSubCheck(int si, int ci, int* map_state_to_node,int* r_out_adj_sizes,int* q_out_adj_sizes,int* r_in_adj_sizes,int* q_in_adj_sizes,void** r_nodes_attrs,void** q_nodes_attrs, int comparatorType){
		if(			r_out_adj_sizes[ci] >= q_out_adj_sizes[map_state_to_node[si]]
					&& r_in_adj_sizes[ci] >= q_in_adj_sizes[map_state_to_node[si]]){
			return nodeComparator(comparatorType, r_nodes_attrs[ci], q_nodes_attrs[map_state_to_node[si]]);
		}
		return false;
	}
    __host__ __device__ 
	bool edgesSubCheck(int si, int ci, int* solution, bool* matched,int* edges_sizes, MaMaEdge * m_flat_edges, int * m_flat_edges_indexes,int* r_out_adj_sizes,int** r_out_adj_list, void*** r_out_adj_attrs, int comparatorType){
		int source, target;
		int ii;
		for(int me=0; me<edges_sizes[si]; me++){
			//printf("siamo qui dentro"); 
			source = solution[ m_flat_edges[m_flat_edges_indexes[si] + me].source ];
			target = solution[ m_flat_edges[m_flat_edges_indexes[si] + me].target ];

			for(ii=0; ii< r_out_adj_sizes[source]; ii++){
				if(r_out_adj_list[source][ii] == target){
//					if(! edgeComparator.compare(rgraph.out_adj_attrs[source][ii],  mama.edges[si][me].attr)){
//						return false;
//					}
//					else{
//						break;
//					}
					if(edgeComparator(comparatorType, r_out_adj_attrs[source][ii],  m_flat_edges[m_flat_edges_indexes[si] + me].attr)){
						break;
					}
				}
			}
			if(ii >= r_out_adj_sizes[source]){
				return false;
			}
		}
		return true;
	}


	

    void subsolver(
        //printToConsole
        bool* printToConsole,
        long* matchCount, 
        //typeComparator
        int* type_comparator,
        //Mama
        int * nof_sn, 
		void* nodes_attrs, 
		int* edges_sizes, 
        MaMaEdge* flat_edges, 
        int* flat_edges_indexes,
		int* map_node_to_state, 			
		int* map_state_to_node,
		int* parent_state,
		MAMA_PARENTTYPE* parent_type,
        //rgraph
        int* r_nof_nodes,
        int** r_in_adj_list, 
        int* r_in_adj_sizes,
        int** r_out_adj_list, 
        int* r_out_adj_sizes,
        void** r_nodes_attrs,
        void*** r_out_adj_attrs,
        //qgraph
        int* q_nof_nodes,
        int** q_in_adj_list, 
        int* q_in_adj_sizes,
        int** q_out_adj_list, 
        int* q_out_adj_sizes,
        void** q_nodes_attrs,

        long* steps,
        long* triedcouples,
        long* matchedcouples
    ){
        
        int ii; 
        int* listAllRef = new int[*r_nof_nodes];
		for(ii=0; ii<*r_nof_nodes; ii++)
			listAllRef[ii] = ii;

		int** candidates = new int*[*nof_sn];							//indexed by state_id
		int* candidatesIT = new int[*nof_sn];							//indexed by state_id
		int* candidatesSize = new int[*nof_sn];							//indexed by state_id
		int* solution = new int[*nof_sn];							
        //indexed by state_id
		for(ii=0; ii<*nof_sn; ii++){
			solution[ii] = -1;
		}

		std:set<int>* cmatched = new std::set<int>[*nof_sn];

		bool* matched = (bool*) calloc(*r_nof_nodes, sizeof(bool));		//indexed by node_id

		candidates[0] = listAllRef;
		candidatesSize[0] = *r_nof_nodes;
		candidatesIT[0] = -1;

		int psi = -1;
		int si = 0;
		int ci = -1;
		int sip1;
		
		while(si != -1){
			//steps++;

			if(psi >= si){
				matched[solution[si]] = false;
			}

			ci = -1;
			candidatesIT[si]++;
			while(candidatesIT[si] < candidatesSize[si]){
				//triedcouples++;

				ci = candidates[si][candidatesIT[si]];
				solution[si] = ci;

//				std::cout<<"[ "<<map_state_to_node[si]<<" , "<<ci<<" ]\n";
//				if(matched[ci]) std::cout<<"fails on alldiff\n";
//				if(!nodeCheck(si,ci, map_state_to_node)) std::cout<<"fails on node label\n";
//				if(!(edgesCheck(si, ci, solution, matched))) std::cout<<"fails on edges \n";
				
				//MT_ISO
				
					if(	  (!matched[ci])
						&& (cmatched[si].find(ci)==cmatched[si].end())
						&& nodeSubCheck(si,ci, map_state_to_node, r_out_adj_sizes,q_out_adj_sizes,r_in_adj_sizes,q_in_adj_sizes,r_nodes_attrs,q_nodes_attrs,*type_comparator)
						&& edgesSubCheck(si, ci, solution, matched,edges_sizes,flat_edges,flat_edges_indexes,r_out_adj_sizes,r_out_adj_list,r_out_adj_attrs,*type_comparator)
								){
						break;
					}
					else{
						ci = -1;
					}
				
				candidatesIT[si]++;
			}

			if(ci == -1){
				psi = si;
				cmatched[si].clear();
				si--;
			}
			else{
				cmatched[si].insert(ci);
				(*matchedcouples)++;

				if(si == *nof_sn -1){
					matchListener(printToConsole, matchCount, *nof_sn, map_state_to_node, solution);
#ifdef FIRST_MATCH_ONLY
					si = -1;
#endif
					psi = si;
				}
				else{
					matched[solution[si]] = true;
					sip1 = si+1;
					if(parent_type[sip1] == PARENTTYPE_NULL){
						candidates[sip1] = listAllRef;
						candidatesSize[sip1] = *r_nof_nodes;
					}
					else{
						if(parent_type[sip1] == PARENTTYPE_IN){
							candidates[sip1] = r_in_adj_list[solution[parent_state[sip1]]];
							candidatesSize[sip1] = r_in_adj_sizes[solution[parent_state[sip1]]];
						}
						else{//(parent_type[sip1] == MAMA_PARENTTYPE::PARENTTYPE_OUT)
							candidates[sip1] = r_out_adj_list[solution[parent_state[sip1]]];
							candidatesSize[sip1] = r_out_adj_sizes[solution[parent_state[sip1]]];
						}
					}
					candidatesIT[si +1] = -1;

					psi = si;
					si++;
				}
			}
		}
    
    // memory cleanup
    free(matched);
    delete[] cmatched;
    delete[] solution;
    delete[] candidatesSize;
    delete[] candidatesIT;
    delete[] candidates;
    delete[] listAllRef;
	}
	

#endif 