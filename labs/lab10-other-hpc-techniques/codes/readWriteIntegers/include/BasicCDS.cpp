/*
 * BasicCDC.cpp
 *
 *  Created on: April 2018
 *      Author: HECTOR FERRADA
 */

#include "BasicCDS.h"

namespace cds {
	// sets bit i-ht in e (left to right)
	void setBit64(ulong *e, ulong i) {
		e[i>>BW64] |= (maskW63>>(i%W64));
	}

	// cleans bit i-ht in e (left to right)
	void cleanBit64(ulong * e, ulong i) {
		e[i>>BW64] &= ~(maskW63>>(i%W64));
	}

	// print W64 bits of unsigned long int x
	void printBitsUlong(ulong x){
		uint cnt = 0;
		ulong mask = 0x8000000000000000;

		for(cnt=0;cnt<W64;++cnt){
			putchar(((x & mask) == 0) ? '0' : '1');
			mask >>= 1;
		}
	}

	// set the number x as a bitstring sequence in *A. In the range of bits [ini, .. ini+len-1] of *A. Here x has len bits
	void setNum64(ulong *A, ulong ini, uint len, ulong x) {
		ulong i=ini>>BW64, j=ini-(i<<BW64);

		if ((j+len)>W64){
			ulong myMask = ~(~0ul >> j);
			A[i] = (A[i] & myMask) | (x >> (j+len-W64));
			myMask = ~0ul >> (j+len-W64);
			A[i+1] = (A[i+1] & myMask) | (x << (WW64-j-len));
		}else{
			ulong myMask = (~0ul >> j) ^ (~0ul << (W64-j-len)); // XOR: 1^1=0^0=0; 0^1=1^0=1
			A[i] = (A[i] & myMask) | (x << (W64-j-len));
		}
	}

	// return (in a unsigned long integer) the number in A from bits of position 'ini' to 'ini+len-1'
	ulong getNum64(ulong *A, ulong ini, uint len){
		ulong i=ini>>BW64, j=ini-(i<<BW64);
		ulong result = (A[i] << j) >> (W64-len);

		if (j+len > W64)
			result = result | (A[i+1] >> (WW64-j-len));

		return result;
	}
} /* namespace cds */
