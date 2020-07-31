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

namespace rilib {

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

    int *d_q_nof_nodes;
    int *d_q_flatten_in_adj_list;
    int *d_q_offset_in_adj_list;
    int *d_q_in_adj_sizes;
    int *d_q_flatten_out_adj_list;
    int *d_q_offset_out_adj_list;
    int *d_q_out_adj_sizes;
    void *d_q_flatten_nodes_attr;
    int *d_q_offset_nodes_attr;

    int *d_nof_sn;
    int *d_edges_sizes;
    MaMaEdge * d_flat_edges;
    int *d_flat_edges_indexes;
    int *d_map_node_to_state;
    int *d_map_state_to_node;
    int *d_parent_state;
    int *d_parent_type;

    using namespace rilib;

    enum MATCH_TYPE {
        MT_ISO, MT_INDSUB, MT_MONO
    };


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
                //printf("%d\n",reference.length_nodes_attrs);
                //SubGISolver* solver3;
                //solver3 = new SubGISolver(matchingMachine, reference, query, nodeComparator, edgeComparator, matchListener,2);
                /*
                long steps;
                long triedcouples;
                long matchedcouples;
                */


                subsolver(
                        //printToConsole
                        printToConsole,
                        matchCount,
                        //typeComparator
                        &comparatorType,
                        //Mama
                        &matchingMachine.nof_sn,
                        matchingMachine.edges_sizes,
                        matchingMachine.flat_edges_indexes,
                        matchingMachine.source,
                        matchingMachine.target,
                        matchingMachine.attr,
                        matchingMachine.offset_attr,
                        matchingMachine.map_node_to_state,
                        matchingMachine.map_state_to_node,
                        matchingMachine.parent_state,
                        matchingMachine.parent_type,
                        //rgraph
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
                        //qgraph
                        d_q_nof_nodes,
                        d_q_flatten_in_adj_list,
                        d_q_offset_in_adj_list,
                        d_q_in_adj_sizes,
                        d_q_flatten_out_adj_list,
                        d_q_offset_out_adj_list,
                        d_q_out_adj_sizes,
                        d_q_flatten_nodes_attr,
                        d_q_offset_nodes_attr,
                        steps,
                        triedcouples,
                        matchedcouples
                );
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
