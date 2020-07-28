
    __host__ __device__     
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

    __host__ __device__ 
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
    __host__ __device__ 
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

void matchListener(bool* printToConsole,long * matchcount, int n, int* qIDs, int* rIDs){
        if(*printToConsole){
            (*matchcount)++;
            std::cout<< "{";
            for(int i=0; i<n; i++){
                std::cout<< "("<< qIDs[i] <<","<< rIDs[i] <<")";
            }
            std::cout<< "}\n";
        }
        else{
            (*matchcount)++; 
        }
		
	}