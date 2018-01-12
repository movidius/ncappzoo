
#ifndef INCLUDE_FP16_H

unsigned half2float(unsigned short h);
unsigned short float2half(unsigned f);
void fp16tofloat(float *dst, unsigned char *src, unsigned nelem);


#endif
