


/**
* Get the substring from a flatted array of string
* @param nodes_attrs
* @param start
* @param end
* @return
*/
__device__
void * getSubString(void * nodes_attrs,int start, int end){
    int len = end - start + 1;
    char * sub_string = (char *)malloc(len*sizeof(char));
    int pos = 0;
    for (int i = start; i < end; ++i,++pos) {
        sub_string[pos] = ((char*)nodes_attrs)[i];
    }
    sub_string[pos] = '\0';
    return (void*)sub_string;
}

    __device__
    int strcmp (const char *p1, const char *p2)
    {
    const unsigned char *s1 = (const unsigned char *) p1;
    const unsigned char *s2 = (const unsigned char *) p2;
    unsigned char c1, c2;
    do
        {
        c1 = (unsigned char) *s1++;
        c2 = (unsigned char) *s2++;
        if (c1 == '\0')
            return c1 - c2;
        }
    while (c1 == c2);
    
    return c1 - c2;
    }

    __device__
    bool nodeComparator(int comparatorType, void * attr1, void * attr2){
        char* a=(char*)attr1;
        char* b=(char*)attr2;
        switch (comparatorType)
        {
        case 0:
            return (strcmp(a, b))==0;
            break;
        case 1: 
            return true;
            break; 
        case 2: 
            return (strcmp(a, b))==0;
            break;
        default:
            break;
        }
    }
    __device__
    bool edgeComparator(int comparatorType, void* attr1, void* attr2){
        char* a=(char*)attr1;
        char* b=(char*)attr2;
        switch (comparatorType)
        {
        case 0:
            return true; 
            break;
        case 1: 
            return true;
            break; 
        case 2:     
            return (strcmp(a, b))==0;
            break;
        default:
            break;
        }
    }

__device__ volatile int lock=0;
__device__
void matchListener(bool* printToConsole,long * matchcount, int n, int* qIDs, int* rIDs){
   

        if(*printToConsole){                  
            atomicAdd ( (int*)matchcount , 1);
            bool leave = true;
	        while (leave) {
	        	if (atomicCAS((int*)&lock, 0, 1) == 0) {
                    printf("{");
                    for(int i=0; i<n; i++){
                        printf("(%d,%d)", qIDs[i] , rIDs[i]);
                    }
                    printf("}\n");
			        leave = false;
			        atomicExch((int*)&lock, 0);
		        }
		    
            }
        }
        else{
            atomicAdd ( (int*)matchcount , 1);
        }
		
	}