bool nodeMonoCheck(int si, int ci, int* map_state_to_node){
		if(			rgraph.out_adj_sizes[ci] >= qgraph.out_adj_sizes[map_state_to_node[si]]
					&& rgraph.in_adj_sizes[ci] >= qgraph.in_adj_sizes[map_state_to_node[si]]){
			return nodeComparator.compare(rgraph.nodes_attrs[ci], qgraph.nodes_attrs[map_state_to_node[si]]);
		}
		return false;
}

	
bool edgesMonoCheck(int si, int ci, int* solution, bool* matched){
		int source, target;
		int ii;
		for(int me=0; me<mama.edges_sizes[si]; me++){
			//printf("siamo qui dentro"); 
			source = solution[ mama.flat_edges[mama.flat_edges_indexes[si] + me].source ];
			target = solution[ mama.flat_edges[mama.flat_edges_indexes[si] + me].target ];

			for(ii=0; ii<rgraph.out_adj_sizes[source]; ii++){
				if(rgraph.out_adj_list[source][ii] == target){
//					if(! edgeComparator.compare(rgraph.out_adj_attrs[source][ii],  mama.edges[si][me].attr)){
//						return false;
//					}
//					else{
//						break;
//					}
					if(edgeComparator.compare(rgraph.out_adj_attrs[source][ii],  mama.flat_edges[mama.flat_edges_indexes[si] + me].attr)){
						break;
					}
				}
			}
			if(ii >= rgraph.out_adj_sizes[source]){
				return false;
			}
		}
		return true;
	}

virtual bool nodeISOCheck(int si, int ci, int* map_state_to_node){
		if(			rgraph.out_adj_sizes[ci] >= qgraph.out_adj_sizes[map_state_to_node[si]]
					&& rgraph.in_adj_sizes[ci] >= qgraph.in_adj_sizes[map_state_to_node[si]]){
			return nodeComparator.compare(rgraph.nodes_attrs[ci], qgraph.nodes_attrs[map_state_to_node[si]]);
		}
		return false;
	}

	virtual bool edgesISOCheck(int si, int ci, int* solution, bool* matched){
		int source, target;
		int ii;
		for(int me=0; me<mama.edges_sizes[si]; me++){
			source = solution[ mama.edges[si][me].source ];
			target = solution[ mama.edges[si][me].target ];

			for(ii=0; ii<rgraph.out_adj_sizes[source]; ii++){
				if(rgraph.out_adj_list[source][ii] == target){
//					if(! edgeComparator.compare(rgraph.out_adj_attrs[source][ii],  mama.edges[si][me].attr)){
//						return false;
//					}
//					else{
//						break;
//					}
					if(edgeComparator.compare(rgraph.out_adj_attrs[source][ii],  mama.edges[si][me].attr)){
						break;
					}
				}
			}
			if(ii >= rgraph.out_adj_sizes[source]){
				return false;
			}
		}
		return true;
	}

		virtual bool nodeINDSUBCheck(int si, int ci, int* map_state_to_node){
		if(			rgraph.out_adj_sizes[ci] >= qgraph.out_adj_sizes[map_state_to_node[si]]
					&& rgraph.in_adj_sizes[ci] >= qgraph.in_adj_sizes[map_state_to_node[si]]){
			return nodeComparator.compare(rgraph.nodes_attrs[ci], qgraph.nodes_attrs[map_state_to_node[si]]);
		}
		return false;
	}

	virtual bool edgesINDSUBCheck(int si, int ci, int* solution, bool* matched){
		int source, target;
		int ii;
		for(int me=0; me<mama.edges_sizes[si]; me++){
			source = solution[ mama.edges[si][me].source ];
			target = solution[ mama.edges[si][me].target ];

			for(ii=0; ii<rgraph.out_adj_sizes[source]; ii++){
				if(rgraph.out_adj_list[source][ii] == target){
//					if(! edgeComparator.compare(rgraph.out_adj_attrs[source][ii],  mama.edges[si][me].attr)){
//						return false;
//					}
//					else{
//						break;
//					}
					if(edgeComparator.compare(rgraph.out_adj_attrs[source][ii],  mama.edges[si][me].attr)){
						break;
					}
				}
//				else if(rgraph.out_adj_list[source][ii] > target){
//					return false;
//				}
			}
			if(ii >= rgraph.out_adj_sizes[source]){
				return false;
			}
		}


		int count = 0;
		for(ii=0; ii< rgraph.out_adj_sizes[ci]; ii++){
			if(matched[rgraph.out_adj_list[ci][ii]]){
				count++;
				if(count > mama.o_edges_sizes[si])
					return false;
			}
		}
		count = 0;
		for(ii=0; ii< rgraph.in_adj_sizes[ci]; ii++){
			if(matched[rgraph.in_adj_list[ci][ii]]){
				count++;
				if(count > mama.i_edges_sizes[si])
					return false;
			}

		}
		//if(count != mama.edges_sizes[si])
		//	return false;

		return true;
	}