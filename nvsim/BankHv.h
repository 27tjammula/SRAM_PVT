/*
 * BankHv.h
 *
 *  Created on: Jun 13, 2010
 *      Author: Xiangyu
 */

#ifndef BANKHV_H_
#define BANKHV_H_

#include "Bank.h"
#include "Mat.h"
#include "typedef.h"

class BankHv: public Bank {
public:
	BankHv();
	virtual ~BankHv();

	/* Functions */
	void Initialize(int _numRowMat, int _numColumnMat, long long _capacity,
			long _blockSize, int _associativity, int _numRowPerSet, int _numActiveMatPerRow,
			int _numActiveMatPerColumn, int _muxSenseAmp, bool _internalSenseAmp, int _muxOutputLev1, int _muxOutputLev2,
			int _numRowSubarray, int _numColumnSubarray,
			int _numActiveSubarrayPerRow, int _numActiveSubarrayPerColumn,
			BufferDesignTarget _areaOptimizationLevel, MemoryType _memoryType);
	void CalculateArea();
	void CalculateRC();
	void CalculateLatencyAndPower();
	BankHv & operator=(const BankHv &);

	int numAddressBit;		/* Number of bank address bits */
	int numDataDistributeBit;	/* Number of bank data bits (these bits will be distributed along with the address) */
	int numDataBroadcastBit;	/* Number of bank data bits (these bits will be broadcasted at every node) */

	int levelHorizontal;			/* The number of horizontal levels */
	int levelVertical;				/* The number of vertical levels */
	int * numHorizontalWire;		/* The number of horizontal wires on level x */
	double * lengthHorizontalWire;	/* The length of horizontal wires on level x, Unit: m */
	int * numVerticalWire;			/* The number of vertical wires on level x */
	double * lengthVerticalWire;	/* The length of vertical wires on level 2, Unit: m */
	int *numActiveHorizontalWireGroup;		/* The number of active horizontal wire group on level x */
	int *numActiveVerticalWireGroup;		/* The number of active vertical wire group on level x */
	int *numHorizontalWireGroup;	/* The number of horizontal wire group on level x */
	int *numVerticalWireGroup;		/* The number of vertical wire group on level x */
};

#endif /* BANKHV_H_ */
