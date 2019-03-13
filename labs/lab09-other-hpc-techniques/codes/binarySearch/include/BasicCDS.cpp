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

	// print the last cantBits of unisned int x
	void printBitsNum(uint x, uint cantBits){
		uint cnt, mask = 1 << (cantBits-1);

		for(cnt=1;cnt<=cantBits;++cnt){
			putchar(((x & mask) == 0) ? '0' : '1');
			mask >>= 1;
		}
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

	// return (in a unsigned long integer) the number in A from bits of position 'ini' to 'ini+len-1'
	ulong getNum64(ulong *A, ulong ini, uint len){
		ulong i=ini>>BW64, j=ini-(i<<BW64);
		ulong result = (A[i] << j) >> (W64-len);

		if (j+len > W64)
			result = result | (A[i+1] >> (WW64-j-len));

		return result;
	}

	// Extract n cells: A[sp,...,sp+n-1] and stores values in B[0,...,n-1], where each cell has lenCell bits.
	void extractUlongs(ulong *A, ulong sp, ulong n, uint lenCell, ulong *B) {
		ulong i = sp*(lenCell>>BW64);			// byte i of A
		ulong j = (sp*lenCell)%W64;		// offset j inside byte i in A
		ulong k=0;

		while (k<n){
			B[k] = (A[i] << j) >> (W64-lenCell);
			if (j+lenCell > W64)
				B[k] = B[k] | (A[i+1] >> (WW64-j-lenCell));

			j+=lenCell;
			if(j>=W64){
				j-=W64;
				i++;
			}
			k++;
		}
	}

	// set the number x as a bitstring sequence in *A. In the range of bits [ini, .. ini+len-1] of *A. Here x has len bits
	void setNum32(uint *A, uint ini, uint len, uint x) {
		uint i=ini>>BW32, j=ini-(i<<BW32);

		if ((j+len)>W32){
			uint myMask = ~(~0u >> j);
			A[i] = (A[i] & myMask) | (x >> (j+len-W32));
			myMask = ~0u >> (j+len-W32);
			A[i+1] = (A[i+1] & myMask) | (x << (W64-j-len));
		}else{
			uint myMask = (~0u >> j) ^ (~0u << (W32-j-len));
			A[i] = (A[i] & myMask) | (x << (W32-j-len));
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
} /* namespace cds */
