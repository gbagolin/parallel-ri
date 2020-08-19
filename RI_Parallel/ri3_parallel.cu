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

#include <fstream>
#include <cstdlib>
#include <ctime>

#include <stdio.h>
#include <stdlib.h>
#include "c_textdb_driver.h"
#include "timer.h"

#include "enum.h"

#include "Graph.h"
#include "MatchingMachine.h"
#include "MaMaConstrFirst.h"
#include "Parallel_Match.h"


//#define FIRST_MATCH_ONLY  //if setted, the searching process stops at the first found match
#include "flatter.h"
//#define PRINT_MATCHES
//#define CSV_FORMAT
#define N_TEST 30

using namespace rilib;

void usage(char *args0);

int match(MATCH_TYPE matchtype, GRAPH_FILE_TYPE filetype, std::string &referencefile, std::string &queryfile);

int main(int argc, char *argv[]) {

    if (argc != 5) {
        usage(argv[0]);
        return -1;
    }

    MATCH_TYPE matchtype;
    GRAPH_FILE_TYPE filetype;
    std::string reference;
    std::string query;

    std::string par = argv[1];
    if (par == "iso") {
        matchtype = MT_ISO;
    } else if (par == "ind") {
        matchtype = MT_INDSUB;
    } else if (par == "mono") {
        matchtype = MT_MONO;
    } else {
        usage(argv[0]);
        return -1;
    }

    par = argv[2];
    if (par == "gfu") {
        filetype = GFT_GFU;
    } else if (par == "gfd") {
        filetype = GFT_GFD;
    } else if (par == "gfda") {
        filetype = GFT_GFDA;
    } else if (par == "geu") {
        filetype = GFT_EGFU;
    } else if (par == "ged") {
        filetype = GFT_EGFD;
    } else if (par == "vfu") {
        filetype = GFT_VFU;
    } else {
        usage(argv[0]);
        return -1;
    }

    reference = argv[3];
    query = argv[4];

    return match(matchtype, filetype, reference, query);
};

void usage(char *args0) {
    std::cout << "usage " << args0 << " [iso ind mono] [gfu gfd gfda geu ged vfu] reference query\n";
    std::cout << "\tmatch type:\n";
    std::cout << "\t\tiso = isomorphism\n";
    std::cout << "\t\tind = induced subisomorphism\n";
    std::cout << "\t\tmono = monomorphism\n";
    std::cout << "\tgraph input format:\n";
    std::cout << "\t\tgfu = undirect graphs with labels on nodes\n";
    std::cout << "\t\tgfd = direct graphs with labels on nodes\n";
    std::cout << "\t\tgfd = direct graphs with one single label on nodes\n";
    std::cout << "\t\tgeu = undirect graphs with labels both on nodes and edges\n";
    std::cout << "\t\tged = direct graphs with labels both on nodes and edges\n";
    std::cout << "\t\tvfu = VF2Lib undirect unlabeled format\n";
    std::cout << "\treference file contains one or more reference graphs\n";
    std::cout << "\tquery contains the query graph (just one)\n";
};

int match(
        MATCH_TYPE matchtype,
        GRAPH_FILE_TYPE filetype,
        std::string &referencefile,
        std::string &queryfile) {

    total_s = start_time();

    bool takeNodeLabels = false;
    bool takeEdgesLabels = false;
    int rret;
    
    switch (filetype) {
        case GFT_GFU:
        case GFT_GFD:
            comparatorType = 0;
            // only nodes have labels and they are strings

            takeNodeLabels = true;

            break;
        case GFT_GFDA:
            comparatorType = 1;

            takeNodeLabels = true;
            break;
        case GFT_EGFU:
        case GFT_EGFD:
            comparatorType = 2;
            //labels on nodes and edges, both of them are strings

            takeNodeLabels = true;
            takeEdgesLabels = true;
            break;
        case GFT_VFU:
            comparatorType = 1;
            //no labels

            break;
        default:
            return -1;
    }

    TIMEHANDLE tt_start;
    double tt_end;

    //read the query graph
    load_s_q = start_time();
    Graph *query = new Graph();
    rret = read_graph(queryfile.c_str(), query, filetype);
    if(rret==0)
        flatterGraph(query);
    load_t_q += end_time(load_s_q);

    if (rret != 0) {
        std::cout << "error on reading query graph\n";
    }

    make_mama_s = start_time();
    MaMaConstrFirst *mama = new MaMaConstrFirst(*query);
    mama->build(*query);
    make_mama_t += end_time(make_mama_s);

    long steps = 0,                //total number of steps of the backtracking phase
    triedcouples = 0,        //nof tried pair (query node, reference node)
    matchcount = 0,        //nof found matches
    matchedcouples = 0;        //nof mathed pair (during partial solutions)
    long tsteps = 0, ttriedcouples = 0, tmatchedcouples = 0;
    FILE *fd = open_file(referencefile.c_str(), filetype);
    if (fd != NULL) {
        bool printToConsole = false;
        long matchCount = 0;
#ifdef PRINT_MATCHES
        //to print found matches on screen
        printToConsole = true;
#else
        //do not print matches, just count them
        printToConsole = false;
#endif
        int i = 0;
        bool rreaded = true;
        int count = 0;
        do {//for each reference graph inside the input file
#ifdef PRINT_MATCHES
            std::cout << "#" << i << "\n";
#endif
            //read the next reference graph
            load_s = start_time();
            Graph *rrg = new Graph();
            int rret = read_dbgraph(referencefile.c_str(), fd, rrg, filetype);
            rreaded = (rret == 0); 
            if(rreaded)
                flatterGraph(rrg);

            load_t += end_time(load_s);

            if (rreaded) {
      

                count++;
              
                reference_malloc(*rrg);
                reference_memcpy(*rrg);
                query_malloc(*query);
                in_out_malloc();
                mama_malloc(*mama);
                
                //run the matching
                
                parallel_match(
                      *rrg,
                      *query,
                      *mama,
                      matchtype,
                      &tsteps,
                      &ttriedcouples,
                      &tmatchedcouples,
                      filetype,
                      &printToConsole,
                      &matchCount
                );

                //see rilib/Solver.h
//					steps += tsteps;
//					triedcouples += ttriedcouples;
                matchedcouples += tmatchedcouples;
                tmatchedcouples = 0;
                mama_free();
                query_free();
                reference_free();
                SAFE_CALL(cudaFree(d_steps));
                SAFE_CALL(cudaFree(d_triedcouples));
                SAFE_CALL(cudaFree(d_matchedcouples));
                SAFE_CALL(cudaFree(d_matchCount));
                SAFE_CALL(cudaFree(d_printToConsole));
                SAFE_CALL(cudaFree(d_comparatorType));
                device_reset_s = start_time();
                cudaDeviceReset();
                device_reset_t += end_time(device_reset_s);
            }
            
            
            delete rrg;

            i++;
        } while (rreaded && count < N_TEST);

        matchcount = matchCount;
        fclose(fd);
    } else {
        std::cout << "unable to open reference file\n";
        return -1;
    }

    total_t = end_time(total_s);

#ifdef CSV_FORMAT
    std::cout<<referencefile<<"\t"<<queryfile<<"\t";
    std:cout<<load_t_q<<"\t"<<make_mama_t<<"\t"<<load_t<<"\t"<<match_t<<"\t"<<total_t<<"\t"<<steps<<"\t"<<triedcouples<<"\t"<<matchedcouples<<"\t"<<matchcount;
#else
    std::cout << match_t << "\n";

    printf("reference file: %s \n",referencefile.c_str());
    printf("query file: %s \n",queryfile.c_str());
    std::cout << "total time: " << total_t << "\n";
    std::cout << "matching time: " << match_t << "\n";
    std::cout << "number of found matches: " << matchcount << "\n";
    std::cout << "search space size: " << matchedcouples << "\n";
    cout << "Pattern Graph load time " << load_t_q << endl;
    cout << "Target Graphs load time " << load_t << endl;
    cout << "mama time: " << make_mama_t << endl;
    std::cout << "CudaMalloc time "<<cuda_malloc_t<<std::endl;
    std::cout << "CudaFree time "<<cuda_free_t<<std::endl;
    std::cout << "Device Reset time "<<device_reset_t<<std::endl;
    std::cout << "Device memory heap definition time "<<device_memory_t<<std::endl;
#endif
    delete mama;
    delete query;

    return 0;
};





