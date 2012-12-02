__kernel void shift(__global float* inicial, __global float* final, __global int* rowSize, __global int* columnSize, __global int* shifts, __global int* rowShift) {
    unsigned int row = get_global_id(0);
    unsigned int column = get_global_id(1);
    int i, offset, size, realShifts, shiftAxis;
    int pos = row*(*columnSize)+column;
    if ((*rowShift)) {
        shiftAxis = (*columnSize);
        realShifts = (*shifts)%(*rowSize);
        size = ((*rowSize)*(*columnSize));
        offset = 0;
    } else {
        shiftAxis = 1;
        realShifts = (*shifts)%(*columnSize);
        size = (*columnSize);
        offset = (*columnSize)*row;
    }
	for ( i = 0; i < realShifts; i++) {
        pos = (pos+shiftAxis)%size+offset;
    }
	final[row*(*columnSize)+column] = inicial[pos];
}
