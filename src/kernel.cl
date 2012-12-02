__kernel void shiftRow(__global float* inicial, __global float* final, __global int* rowSize, __global int* columnSize)
{
    unsigned int row = get_global_id(0);
    unsigned int column = get_global_id(1);
    int pos = (row*(*columnSize)+column + (*columnSize)) % ((*rowSize)*(*columnSize));
	final[(row*(*columnSize)+column)] = inicial[pos];
}
