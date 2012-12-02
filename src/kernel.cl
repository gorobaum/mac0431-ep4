__kernel void shiftRow(__global float* inicial, __global float* final, __global int* rowSize, __global int* columnSize, __global int* shifts) {
    unsigned int row = get_global_id(0);
    unsigned int column = get_global_id(1);
    int i = 0;
    int size = ((*rowSize)*(*columnSize));
    int pos = row*(*columnSize)+column;
    int realShifts = (*shifts)%(*rowSize);
    for ( i = 0; i < realShifts; i++) {
    	pos = (pos + (*columnSize))%size;
    }	
	final[row*(*columnSize)+column] = inicial[pos];
}
