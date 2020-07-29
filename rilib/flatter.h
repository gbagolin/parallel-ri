#ifndef FLATTER_H_
#define FLATTER_H_

	/*
								[1][0]
		indexes: [0,4,8,10,13,17,23,28]
		offset: [0,3,5,7]
		[
			["ciao" ,"come","va"], 
			["sto", "bene"],
			["grazie","mille"],
			["ciao"]
		]	

		["ciaocomevastobene"]

		*/


void flatterGraph(Graph * graph){

    //flatten node_attrs
    for(int i = 0; i < graph -> nof_nodes; i++){
        graph -> length_nodes_attrs += strlen((char *)graph->nodes_attrs[i]); 
    }

    graph -> length_nodes_attrs++; 

    graph -> flatten_nodes_attr = (char * )malloc(graph -> length_nodes_attrs * sizeof(char)); 
	graph -> indexes_nodes_attr = (int *)malloc((graph -> nof_nodes + 1) * sizeof(int)); 

	//strcpy((char *)graph -> flatten_nodes_attr,""); 
    ((char *)graph -> flatten_nodes_attr)[0] = '\0'; 

	int total_length = 0; 
	for(int i=0; i<graph->nof_nodes; i++){
		strcat((char *)graph -> flatten_nodes_attr, (char*)graph->nodes_attrs[i]);
		graph -> indexes_nodes_attr[i] = total_length; 
		total_length += strlen((char*)graph->nodes_attrs[i]); 
	}
    graph -> indexes_nodes_attr[graph->nof_nodes] = total_length + strlen((char*)graph->nodes_attrs[graph->nof_nodes - 1]); 

    /*
   for(int i = 0; i < graph -> nof_nodes; i++){

       int len = graph -> indexes_nodes_attr[i+1] - graph -> indexes_nodes_attr[i]; 
       char * str = (char *)malloc((len + 1)* sizeof(char));
       int pos = 0; 
       for(int j = graph -> indexes_nodes_attr[i]; j < graph -> indexes_nodes_attr[i+1]; j++){
            str[pos++] = ((char *)graph -> flatten_nodes_attr)[j]; 

        }
        str[len] = '\0'; 
        
        if(strcmp(str,(char*)graph->nodes_attrs[i]) != 0){
            printf("qualcosa non va"); 
        }
        
        free(str); 
    }
    */

   //flatten out_adj_attrs

   
}


#endif