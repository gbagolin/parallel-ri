/*
 * Graph.h
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

#ifndef GRAPH_H_
#define GRAPH_H_


namespace rilib{
class Graph{
public:
	int id;
	int nof_nodes;

	void** nodes_attrs; //["ciao", "come", "va"]
	void* flatten_nodes_attr; 
	int length_nodes_attrs; //10 
	int * offset_nodes_attr; //[0,4,8,10]

	int* out_adj_sizes;
	int* in_adj_sizes;

	int** out_adj_list; //[
							//[0,1]
							//[1,2]
						//]
	int* flatten_out_adj_list; 
	int length_out_adj_list; //4
	int * offset_out_adj_list; //[0,2,4]

	int** in_adj_list;			//come sopra
	int* flatten_in_adj_list; 
	int length_in_adj_list;
	int * offset_in_adj_list; 

	void*** out_adj_attrs;
	void* flatten_out_adj_attrs; 
	int length_out_adj_attrs;
	int total_count;
	int * offset_out_adj_attrs; 


	Graph(){
		id = -1;
		nof_nodes = 0;
		nodes_attrs = NULL;
		out_adj_sizes = NULL;
		in_adj_sizes = NULL;
		out_adj_list = NULL;
		in_adj_list = NULL;
		out_adj_attrs = NULL;
		flatten_nodes_attr = NULL;
		length_nodes_attrs = 0; 
		offset_nodes_attr = NULL; 
		flatten_out_adj_list = NULL; 
		length_out_adj_list = 0; 
		offset_out_adj_list = NULL; 
		flatten_in_adj_list = NULL; 
		length_in_adj_list = 0; 
		flatten_out_adj_attrs = NULL; 
		length_out_adj_attrs = 0; 
		total_count = 0;
		offset_out_adj_attrs = NULL; 
		offset_in_adj_list = NULL; 
	}
  
  ~Graph() {
    int i, j;
    
    for (i=0; i<nof_nodes; ++i) {
      for (j=0; j<out_adj_sizes[i]; ++j) free(out_adj_attrs[i][j]);
      free(out_adj_attrs[i]);
      free(in_adj_list[i]);
      free(out_adj_list[i]);
      free(nodes_attrs[i]);

    }
    free(out_adj_attrs);
    free(in_adj_list);
    free(out_adj_list);
    free(in_adj_sizes);
    free(out_adj_sizes);
    free(nodes_attrs);
	free(flatten_nodes_attr); 
	//free(length_nodes_attrs); 
	free(offset_nodes_attr); 
	free(flatten_out_adj_list); 
	//free(length_out_adj_list); 
	free(offset_out_adj_list); 
	free(flatten_in_adj_list); 
	//free(length_in_adj_list);  
	free(flatten_out_adj_attrs); 
	//free(length_out_adj_attrs); 
	free(offset_out_adj_attrs);
	free(offset_in_adj_list); 
  }


//	void sort_edges(){
//		for(int i=0;i<nof_nodes;i++){
//			if(out_adj_sizes[i]>1){
//				quicksort_edges(out_adj_list[i], out_adj_attrs[i], 0, out_adj_sizes[i] -1);
//			}
//		}
//	}
//
//
//	void quicksort_edges(int* adj_list, void** adj_attrs, int p, int r){
//		if(p<r){
//			int q=quicksort_edges_partition(adj_list, adj_attrs, p, r);
//			quicksort_edges(adj_list, adj_attrs, p, q-1);
//			quicksort_edges(adj_list, adj_attrs, q+1, r);
//		}
//	}
//
//
//	int quicksort_edges_partition(int* adj_list, void** adj_attrs, int p, int r){
//		int ltmp; void* atmp;
//		int target = adj_list[r];
//		void* attr = adj_attrs[r];
//		int i = p-1;
//		for(int j=p; j<r; j++){
//			//if(adj_list[j] < target || (adj_list[j]==target  && (edgeComparator.compareint(adj_attrs[j], attr)<=0))){
//			if(adj_list[j] < target){
//				i++;
//				ltmp = adj_list[i];
//				adj_list[i] = adj_list[j];
//				adj_list[j] = ltmp;
//				atmp = adj_attrs[i];
//				adj_attrs[i] = adj_attrs[j];
//				adj_attrs[j] = atmp;
//			}
//		}
//		ltmp = adj_list[i+1];
//		adj_list[i+1] = adj_list[r];
//		adj_list[r] = ltmp;
//		atmp = adj_attrs[i+1];
//		adj_attrs[i+1] = adj_attrs[r];
//		adj_attrs[r] = atmp;
//		return i+1;
//	}

	/*
	void print(){
//		int id;
//		int nof_nodes;
//		void** nodes_attrs;
//		int* out_adj_sizes;
//		int* in_adj_sizes;
//		int** out_adj_list;
//		int** in_adj_list;
//		void*** out_adj_attrs;
//		void*** in_adj_attrs;
		std::cout<<"| ReferenceGraph["<<id<<"] nof nodes["<<nof_nodes<<"]\n";
		for(int i=0; i<nof_nodes; i++){
			std::cout<<"| node["<<i<<"]\n";
			std::cout<<"| \tattribute_pointer["<<nodes_attrs[i]<<"]\n";
			std::cout<<"| \tattribute["<<*((std::string*)(nodes_attrs[i]))<<"]\n";
			std::cout<<"| \tout_adjs["<<out_adj_sizes[i]<<"][";
			for(int j=0; j<out_adj_sizes[i]; j++){
				std::cout<<out_adj_list[i][j];
				if(j!=out_adj_sizes[i]-1)
					std::cout<<", ";
			}
			std::cout<<"]\n";
			std::cout<<"| \tin_adjs["<<in_adj_sizes[i]<<"][";
			for(int j=0; j<in_adj_sizes[i]; j++){
				std::cout<<in_adj_list[i][j];
				if(j!=in_adj_sizes[i]-1)
					std::cout<<", ";
			}
			std::cout<<"]\n";
		}
	}
	*/
};
}


#endif /* REFERENCEGRAPH_H_ */
