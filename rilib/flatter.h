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

void flatterGraph(Graph *graph)
{
    //flatten node_attrs
    for (int i = 0; i < graph->nof_nodes; i++)
    {
        graph->length_nodes_attrs += strlen((char *)graph->nodes_attrs[i]);
    }

    graph->length_nodes_attrs++;

    graph->flatten_nodes_attr = (char *)malloc(graph->length_nodes_attrs * sizeof(char));
    graph->offset_nodes_attr = (int *)malloc((graph->nof_nodes + 1) * sizeof(int));

    //strcpy((char *)graph -> flatten_nodes_attr,"");
    ((char *)graph->flatten_nodes_attr)[0] = '\0';

    int total_length = 0;
    for (int i = 0; i < graph->nof_nodes; i++)
    {
        strcat((char *)graph->flatten_nodes_attr, (char *)graph->nodes_attrs[i]);
        graph->offset_nodes_attr[i] = total_length;
        total_length += strlen((char *)graph->nodes_attrs[i]);
    }
    graph->offset_nodes_attr[graph->nof_nodes] = total_length + strlen((char *)graph->nodes_attrs[graph->nof_nodes - 1]);

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
    //forse questo non serve, perchè nessun dataset ha archi con label.
    
    //flatten node_attrs
    for (int i = 0; i < graph->nof_nodes; i++)
    {
        for (int j = 0; j < graph->out_adj_sizes[i]; j++){
            if(graph->out_adj_attrs[i][j] != NULL){
                graph->length_out_adj_attrs += strlen((char *)graph->out_adj_attrs[i][j]);
                graph->total_count++;
            }
            
        }
    }
    graph->flatten_out_adj_attrs = (void*) calloc(graph->length_out_adj_attrs + 1,sizeof(char));
    graph->offset_out_adj_attrs = (int*) calloc(graph->total_count + 1,sizeof(int));
    for(int i = 0, int c = 0; i < graph->nof_nodes; i++){
            for(int j = 0; j <graph->out_adj_sizes[i]; j++){
                if(graph->out_adj_attrs[i][j] != NULL){
                    strcat((char * )graph->flatten_out_adj_attrs, (char *)graph->out_adj_attrs[c][j]);
                    graph->offset_out_adj_attrs[c + j + 1] = graph->offset_out_adj_attrs[c + j] + strlen((char *)graph->out_adj_attrs[c][j]);
                }
            }

            c += graph->out_adj_sizes[i];

        }
    

    //in_adj_list

    graph->offset_in_adj_list = (int *)malloc((graph->nof_nodes + 1) * sizeof(int));
    graph->offset_in_adj_list[0] = 0;

    for (int i = 0; i < graph->nof_nodes; i++)
    {
        graph->length_in_adj_list += graph->in_adj_sizes[i];
        graph->offset_in_adj_list[i + 1] = graph->offset_in_adj_list[i] + graph->in_adj_sizes[i];
    }

    graph->flatten_in_adj_list = (int *)malloc(graph->length_in_adj_list * sizeof(int));
    int pos = 0;
    for (int i = 0; i < graph->nof_nodes; i++)
    {
        for (int j = 0; j < graph->in_adj_sizes[i]; j++)
        {
            graph->flatten_in_adj_list[pos++] = graph->in_adj_list[i][j];
        }
    }

    //out_adj_list
    graph->offset_out_adj_list = (int *)malloc((graph->nof_nodes + 1) * sizeof(int));
    graph->offset_out_adj_list[0] = 0;

    for (int i = 0; i < graph->nof_nodes; i++)
    {
        graph->length_out_adj_list += graph->out_adj_sizes[i];
        graph->offset_out_adj_list[i + 1] = graph->offset_out_adj_list[i] + graph->out_adj_sizes[i];
    }

    graph->flatten_out_adj_list = (int *)malloc(graph->length_out_adj_list * sizeof(int));
    pos = 0;
    for (int i = 0; i < graph->nof_nodes; i++)
    {
        for (int j = 0; j < graph->out_adj_sizes[i]; j++)
        {
            graph->flatten_out_adj_list[pos++] = graph->out_adj_list[i][j];
        }
    }
    pos = 0; 

    //test out_adj_sizes
    for (int i = 0; i < graph->nof_nodes; i++)
    {
        for (int j = 0; j < graph->out_adj_sizes[i]; j++)
        {
            if(graph->flatten_out_adj_list[graph -> offset_out_adj_list[i] + j] != graph->out_adj_list[i][j]){
                printf("test failed 1\n");
            }
            else{
               // printf("test passed\n");
            }
        }
    }

    //test in_adj_list
    for (int i = 0; i < graph->nof_nodes; i++)
    {
        for (int j = 0; j < graph->in_adj_sizes[i]; j++)
        {
            if(graph->flatten_in_adj_list[graph -> offset_in_adj_list[i] + j] != graph->in_adj_list[i][j]){
                printf("test failed 2\n");
            }
            else{
                //printf("test passed\n");
            }
        }
    }
    //test out_adj_list
    for (int i = 0; i < graph->nof_nodes; i++)
    {
        for (int j = 0; j < graph->out_adj_sizes[i]; j++)
        {
            if(graph->flatten_out_adj_list[graph -> offset_out_adj_list[i] + j] != graph->out_adj_list[i][j]){
                printf("test failed 2\n");
            }
            else{
                //printf("test passed\n");
            }
        }
    }


}

#endif