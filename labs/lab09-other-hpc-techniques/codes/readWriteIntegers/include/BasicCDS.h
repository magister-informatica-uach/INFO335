/*
 * BasicCDS.h
 *
 *  Created on: June/2018
 *      Author: HECTOR FERRADA
 */

#ifndef BASIC_DRF_H_
#define BASIC_DRF_H_

#include <stdio.h>

namespace cds {
	#ifndef uchar // 1 byte
	#define uchar unsigned char
	#endif
	#ifndef suint
	#define suint short unsigned int
	#endif
	#ifndef uint
	#define uint unsigned int
	#endif
	#ifndef ulong
	#define ulong unsigned long
	#endif

	const uint W32 = 32;
	const uint W64 = 64;
	const uint BW32 = 5;	// pow of two for W32
	const uint BW64 = 6;	// pow of two for W64
	const uint WW64 = 128;
	const ulong maskW63 = 0x8000000000000000;

	// reads i-th bit from e (left to right)
	#define readBit64(e,i) ((e[i/W64] >> (W64minusone-i%W64)) & 1)

	// sets bit i-ht in e (left to right)
	void setBit64(ulong *e, ulong i);

	// cleans bit i-ht in e (left to right)
	void cleanBit64(ulong *e, ulong i);

	// print W64 bits of unsigned long int x
	void printBitsUlong(ulong x);

	// set the ulong x as a bitstring sequence in *A. In the range of bits [ini, .. ini+len-1] of *A. We set len bits from x
	void setNum64(ulong *A, ulong ini, uint len, ulong x);

	// return (in a unsigned integer) the number in A from bits of position 'ini' to 'ini+len-1'
	ulong getNum64(ulong *A, ulong ini, uint len);

} /* namespace cds */

#endif /* BASIC_DRF_H_ */
