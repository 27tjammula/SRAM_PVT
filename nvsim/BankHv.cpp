/*
 * BankHv.cpp
 *
 *  Created on: Jun 13, 2010
 *      Author: Xiangyu
 */
#include "BankHv.h"
#include "formula.h"
#include "global.h"

BankHv::BankHv() {
	// TODO Auto-generated constructor stub
	initialized = false;
	invalid = false;
	numHorizontalWire = NULL;
	lengthHorizontalWire = NULL;
	numVerticalWire = NULL;
	lengthVerticalWire = NULL;
	numActiveHorizontalWireGroup = NULL;
	numActiveVerticalWireGroup = NULL;
	numHorizontalWireGroup = NULL;
	numVerticalWireGroup = NULL;
}

BankHv::~BankHv() {
	// TODO Auto-generated destructor stub
	if (numHorizontalWire)
		delete [] numHorizontalWire;
	if (lengthHorizontalWire)
		delete [] lengthHorizontalWire;
	if (numVerticalWire)
		delete [] numVerticalWire;
	if (lengthVerticalWire)
		delete [] lengthVerticalWire;
	if (numActiveHorizontalWireGroup)
		delete [] numActiveHorizontalWireGroup;
	if (numActiveVerticalWireGroup)
		delete [] numActiveVerticalWireGroup;
	if (numHorizontalWireGroup)
		delete [] numHorizontalWireGroup;
	if (numVerticalWireGroup)
		delete [] numVerticalWireGroup;

}

void BankHv::Initialize(int _numRowMat, int _numColumnMat, long long _capacity,
		long _blockSize, int _associativity, int _numRowPerSet, int _numActiveMatPerRow,
		int _numActiveMatPerColumn, int _muxSenseAmp, bool _internalSenseAmp, int _muxOutputLev1, int _muxOutputLev2,
		int _numRowSubarray, int _numColumnSubarray,
		int _numActiveSubarrayPerRow, int _numActiveSubarrayPerColumn,
		BufferDesignTarget _areaOptimizationLevel, MemoryType _memoryType) {
	if (initialized) {
		/* Reset the class for re-initialization */
		if (numHorizontalWire)
			delete [] numHorizontalWire;
		if (lengthHorizontalWire)
			delete [] lengthHorizontalWire;
		if (numVerticalWire)
			delete [] numVerticalWire;
		if (lengthVerticalWire)
			delete [] lengthVerticalWire;
		if (numActiveHorizontalWireGroup)
			delete [] numActiveHorizontalWireGroup;
		if (numActiveVerticalWireGroup)
			delete [] numActiveVerticalWireGroup;
		if (numHorizontalWireGroup)
			delete [] numHorizontalWireGroup;
		if (numVerticalWireGroup)
			delete [] numVerticalWireGroup;
		initialized = false;
		invalid = false;
		numHorizontalWire = NULL;
		lengthHorizontalWire = NULL;
		numVerticalWire = NULL;
		lengthVerticalWire = NULL;
		numActiveHorizontalWireGroup = NULL;
		numActiveVerticalWireGroup = NULL;
		numHorizontalWireGroup = NULL;
		numVerticalWireGroup = NULL;
	}

	numRowMat = _numRowMat;
	numColumnMat = _numColumnMat;
	capacity = _capacity;
	blockSize = _blockSize;
	associativity = _associativity;
	numRowPerSet = _numRowPerSet;
	internalSenseAmp = _internalSenseAmp;
	areaOptimizationLevel = _areaOptimizationLevel;
	memoryType = _memoryType;
	int numWay = 1;	/* default value for non-cache design */

	/* Calculate the physical signals that are required in routing */
	numAddressBit = (int)(log2((double)capacity / blockSize / associativity) + 0.1);
			/* use double during the calculation to avoid overflow */
	if (memoryType == data) {
		numDataDistributeBit = blockSize;
		numDataBroadcastBit = (int)(log2(associativity));	/* TO-DO: this is not the only way */
	} else if (memoryType == tag) {
		numDataDistributeBit = associativity;		/* TO-DO: it seems that it only supports power of 2 here */
		numDataBroadcastBit = blockSize;
	} else {	/* CAM */
		numDataDistributeBit = 0;
		numDataBroadcastBit = blockSize;
	}

	if (_numActiveMatPerRow > numColumnMat) {
		cout << "[Bank] Warning: The number of active subarray per row is larger than the number of subarray per row!"  << endl;
		cout << _numActiveMatPerRow << " > " << numColumnMat << endl;
		numActiveMatPerRow = numColumnMat;
	} else {
		numActiveMatPerRow = _numActiveMatPerRow;
	}
	if (_numActiveMatPerColumn > numRowMat) {
		cout << "[Bank] Warning: The number of active subarray per column is larger than the number of subarray per column!"  << endl;
		cout << _numActiveMatPerColumn << " > " << numRowMat << endl;
		numActiveMatPerColumn = numRowMat;
	} else {
		numActiveMatPerColumn = _numActiveMatPerColumn;
	}
	muxSenseAmp = _muxSenseAmp;
	muxOutputLev1 = _muxOutputLev1;
	muxOutputLev2 = _muxOutputLev2;

	numRowSubarray = _numRowSubarray;
	numColumnSubarray = _numColumnSubarray;
	if (_numActiveSubarrayPerRow > numColumnSubarray) {
		cout << "[Bank] Warning: The number of active subarray per row is larger than the number of subarray per row!"  << endl;
		cout << _numActiveSubarrayPerRow << " > " << numColumnSubarray << endl;
		numActiveSubarrayPerRow = numColumnSubarray;
	} else {
		numActiveSubarrayPerRow = _numActiveSubarrayPerRow;
	}
	if (_numActiveSubarrayPerColumn > numRowSubarray) {
		cout << "[Bank] Warning: The number of active subarray per column is larger than the number of subarray per column!"  << endl;
		cout << _numActiveSubarrayPerColumn << " > " << numRowSubarray << endl;
		numActiveSubarrayPerColumn = numRowSubarray;
	} else {
		numActiveSubarrayPerColumn = _numActiveSubarrayPerColumn;
	}

	levelHorizontal = (int)(log2(numColumnMat)+0.1);
	levelVertical = (int)(log2(numRowMat)+0.1);
	if (levelHorizontal > 0) {
		numHorizontalWire = new int[levelHorizontal];
		lengthHorizontalWire = new double[levelHorizontal];
		numActiveHorizontalWireGroup = new int[levelHorizontal];
		numHorizontalWireGroup = new int[levelHorizontal];
	}
	if (levelVertical > 0) {
		numVerticalWire = new int[levelVertical];
		lengthVerticalWire = new double[levelVertical];
		numActiveVerticalWireGroup = new int[levelVertical];
		numVerticalWireGroup = new int[levelVertical];
	}

	/* When H > V */
	int h = levelHorizontal;
	int v = levelVertical;
	int rowToActive = numActiveMatPerColumn;
	int columnToActive = numActiveMatPerRow;
	int numAddressBitToRoute = numAddressBit;
	int numDataDistributeBitToRoute = numDataDistributeBit;
	int numDataBroadcastBitToRoute = numDataBroadcastBit;

	int numActiveWireGroup = 1;
	int numWireGroup = 1;

	if (h == 0 && v > 0) {
		/* if there is no horizontal wire, the number of vertical wire starts from 2 */
		numActiveWireGroup = numWireGroup = 2;
		if (rowToActive > 1) {
			numDataDistributeBitToRoute /= 2;
			rowToActive /= 2;
		} else {
			numAddressBitToRoute--;
			numActiveWireGroup /= 2;	/* half of them are shut down */
		}
	}

	while (h > v) {		/* horizontal only */
		if (numDataDistributeBitToRoute + numDataBroadcastBitToRoute == 0 || numAddressBitToRoute == 0) {
			invalid = true;
			initialized = true;
			return;
		}
		numHorizontalWire[levelHorizontal - h] =  numAddressBitToRoute
				+ numDataDistributeBitToRoute + numDataBroadcastBitToRoute;
		numActiveHorizontalWireGroup[levelHorizontal - h] = numActiveWireGroup;
		numHorizontalWireGroup[levelHorizontal - h] = numWireGroup;

		numActiveWireGroup *= 2;	/* for next level, mirrored */
		numWireGroup *= 2;			/* for next level, mirrored */

		/* If there is possibility to reduce the data bits */
		if (columnToActive > 1) {
			numDataDistributeBitToRoute /= 2;
			columnToActive /= 2;
		} else {
			numAddressBitToRoute--;
			numActiveWireGroup /= 2;	/* half of them are shut down */
		}

		h--;
	}

	/* Reduce H to zero */
	while (h > 0) {		/* one horizontal, one vertical */
		/* horizontal first */
		if (numDataDistributeBitToRoute + numDataBroadcastBitToRoute == 0 || numAddressBitToRoute == 0) {
			invalid = true;
			initialized = true;
			return;
		}
		numHorizontalWire[levelHorizontal - h] =  numAddressBitToRoute
				+ numDataDistributeBitToRoute + numDataBroadcastBitToRoute;
		numActiveHorizontalWireGroup[levelHorizontal - h] = numActiveWireGroup;
		numHorizontalWireGroup[levelHorizontal - h] = numWireGroup;

		numActiveWireGroup *= 2;	/* for next level, mirrored */
		numWireGroup *= 2;			/* for next level, mirrored */

		/* If there is possibility to reduce the data bits on vertical */
		if (rowToActive > 1) {
			numDataDistributeBitToRoute /= 2;
			rowToActive /= 2;
		} else {
			numAddressBitToRoute--;
			numActiveWireGroup /= 2;	/* half of them are shut down */
		}

		/* vertical later */
		if (numDataDistributeBitToRoute + numDataBroadcastBitToRoute == 0 || numAddressBitToRoute == 0) {
			invalid = true;
			initialized = true;
			return;
		}
		numVerticalWire[levelVertical - v] = numAddressBitToRoute
				+ numDataDistributeBitToRoute + numDataBroadcastBitToRoute;
		numActiveVerticalWireGroup[levelVertical - v] = numActiveWireGroup;
		numVerticalWireGroup[levelVertical - v] = numWireGroup;

		numActiveWireGroup *= 2;	/* for next level, mirrored */
		numWireGroup *= 2;			/* for next level, mirrored */

		/* If there is possibility to reduce the data bits */
		if (columnToActive > 1) {
			numDataDistributeBitToRoute /= 2;
			columnToActive /= 2;
		} else {
			numAddressBitToRoute--;
			numActiveWireGroup /= 2;	/* half of them are shut down */
		}

		h--;
		v--;
	}

	/* Reduce V to zero */
	while (v > 0) {		/* vertical only */
		if (numDataDistributeBitToRoute + numDataBroadcastBitToRoute == 0 || numAddressBitToRoute == 0) {
			invalid = true;
			initialized = true;
			return;
		}
		numVerticalWire[levelVertical - v] = numAddressBitToRoute
				+ numDataDistributeBitToRoute + numDataBroadcastBitToRoute;
		numActiveVerticalWireGroup[levelVertical - v] = numActiveWireGroup;
		numVerticalWireGroup[levelVertical - v] = numWireGroup;

		numActiveWireGroup *= 2;	/* for next level, mirrored */
		numWireGroup *= 2;			/* for next level, mirrored */

		if (v > 1) {	/* no need to recalculate for the last level */
			/* If there is possibility to reduce the data bits on vertical */
			if (rowToActive > 1) {
				numDataDistributeBitToRoute /= 2;
				rowToActive /= 2;
			} else {
				numAddressBitToRoute--;
				numActiveWireGroup /= 2;	/* half of them are shut down */
			}
		}

		v--;
	}

	/* If this mat is cache data array, determine if the number of cache ways assigned to this mat is legal */
	if (memoryType == data) {
		if (numRowPerSet > (int)pow(2, numDataBroadcastBitToRoute)) {
			/* There is no enough ways to distribute into multiple rows */
			invalid = true;
			initialized = true;
			return;
		}
	}

	/* If this mat is cache tag array, determine if the number of cache ways assigned to this mat is legal */
	if (memoryType == tag) {
		if (numRowPerSet > 1) {
			/* tag array cannot have multiple rows to contain ways in a set, otherwise the bitline has to be shared */
			invalid = true;
			initialized = true;
			return;
		}
		if (numDataDistributeBitToRoute == 0) {
			/* This mat does not contain at least one way */
			invalid = true;
			initialized = true;
			return;
		}
	}

	long matBlockSize;
	/* Determine the number of columns in a mat */
	if (memoryType == data) {		/* Data array */
		/* numDataDistributeBit is the bits in a data word that is assigned to this mat */
		matBlockSize = numDataDistributeBitToRoute;
		numWay = (int)pow(2, numDataBroadcastBitToRoute);
		/* Consider the case if each mat is a cache data array that contains multiple ways */
		int numWayPerRow = numWay / numRowPerSet;	/* At least 1, otherwise it is invalid, and returned already */
		if (numWayPerRow > 1) {		/* multiple ways per row, needs extra mux level */
			/* Do mux level recalculation to contain the multiple ways */
			if (cell->memCellType == DRAM || cell->memCellType == eDRAM) {
				/* for DRAM, mux before sense amp has to be 1, only mux output1 and mux output2 can be used */
				int numWayPerRowInLog = (int)(log2((double)numWayPerRow) + 0.1);
				int extraMuxOutputLev2 = (int)pow(2, numWayPerRowInLog / 2);
				int extraMuxOutputLev1 = numWayPerRow / extraMuxOutputLev2;
				muxOutputLev1 *= extraMuxOutputLev1;
				muxOutputLev2 *= extraMuxOutputLev2;
			} else {
				/* for non-DRAM, all mux levels can be used */
				int numWayPerRowInLog = (int)(log2((double)numWayPerRow) + 0.1);
				int extraMuxOutputLev2 = (int)pow(2, numWayPerRowInLog / 3);
				int extraMuxOutputLev1 = extraMuxOutputLev2;
				int extraMuxSenseAmp = numWayPerRow / extraMuxOutputLev1 / extraMuxOutputLev2;
				muxSenseAmp *= extraMuxSenseAmp;
				muxOutputLev1 *= extraMuxOutputLev1;
				muxOutputLev2 *= extraMuxOutputLev2;
			}
		}
	} else if (memoryType == tag) {	/* Tag array */
		/* numDataBroadcastBit is the tag width, numDataDistributeBit is the number of ways assigned to this mat */
		matBlockSize = numDataBroadcastBitToRoute;
		numWay = numDataDistributeBitToRoute;
	} else {						/* CAM */
		matBlockSize = numDataBroadcastBitToRoute;
		numWay = 1;
	}

	mat.Initialize(numRowSubarray, numColumnSubarray, numAddressBitToRoute, matBlockSize,
			numWay, numRowPerSet, false, numActiveSubarrayPerRow, numActiveSubarrayPerColumn,
			muxSenseAmp, internalSenseAmp, muxOutputLev1, muxOutputLev2, areaOptimizationLevel, memoryType);

	/* Check if mat is under a legal configuration */
	if (mat.invalid) {
		invalid = true;
		initialized = true;
		return;
	}

	/* Reset the mux values for correct printing */
	muxSenseAmp = _muxSenseAmp;
	muxOutputLev1 = _muxOutputLev1;
	muxOutputLev2 = _muxOutputLev2;

	initialized = true;
}

void BankHv::CalculateArea() {
	if (!initialized) {
		cout << "[BankHv] Error: Require initialization first!" << endl;
	} else if (invalid) {
		height = width = area = 1e41;
	} else {
		mat.CalculateArea();
		height = mat.height * numRowMat;
		width = mat.width * numColumnMat;

		int numWireSharingWidth;
		double effectivePitch;
		if (globalWire->wireRepeaterType == repeated_none) {
			numWireSharingWidth = 1;
			effectivePitch = 0;		/* assume that the wire is built on another metal layer, there does not cause silicon area */
			//effectivePitch = globalWire->wirePitch;
		} else {
			numWireSharingWidth = (int)floor(globalWire->repeaterSpacing / globalWire->repeaterHeight);
			effectivePitch = globalWire->repeatedWirePitch;
		}

		for (int i = 0; i < levelHorizontal; i++) {
			height += ceil((double)numHorizontalWire[i] * numHorizontalWireGroup[i] / (int)pow(2,i) / numWireSharingWidth) * effectivePitch;
		}
		for (int i = 0; i < levelVertical; i++) {
			width += ceil((double)numVerticalWire[i] * numVerticalWireGroup[i] / 2 / (int)pow(2,i) / numWireSharingWidth) * effectivePitch;
		}

		/* Determine if the aspect ratio meets the constraint */
		if (memoryType == data)
			if (height / width > CONSTRAINT_ASPECT_RATIO_BANK || width / height > CONSTRAINT_ASPECT_RATIO_BANK) {
				/* illegal */
				invalid = true;
				height = width = area = 1e41;
				return;
			}

		area = height * width;

		/* Calculate the length of each H-tree wire */
		int h = levelHorizontal - 1;
		int v = levelVertical - 1;
		while (v > h) {
			if (v == levelVertical - 1)
				lengthVerticalWire[v] = mat.height / 2;
			else
				lengthVerticalWire[v] = lengthVerticalWire[v + 1] * 2;
			v--;
		}
		while (v >= 0) {
			if (v == levelVertical - 1) {
				lengthVerticalWire[v] = mat.height / 2;
			} else {
				if (h == levelHorizontal - 1) {
					lengthVerticalWire[v] = lengthVerticalWire[v + 1] * 2;
				} else {
					lengthVerticalWire[v] = lengthVerticalWire[v + 1] * 2
							+ ceil((double)numVerticalWire[h + 1] / numWireSharingWidth) * effectivePitch;
				}
			}
			if (h == levelHorizontal - 1) {
				lengthHorizontalWire[h] = mat.width;
				for (int i = v; i < levelVertical; i++)
					lengthHorizontalWire[h] += ceil((double)numVerticalWire[i] / numWireSharingWidth) * effectivePitch / 2;
			} else {
				lengthHorizontalWire[h] = lengthHorizontalWire[h + 1] * 2
						+ ceil((double)numVerticalWire[v] / numWireSharingWidth) * effectivePitch / 2;
			}
			v--;
			h--;
		}
		while (h >=  0) {
			if (h == levelHorizontal - 1)
				lengthHorizontalWire[h] = mat.width;
			else
				lengthHorizontalWire[h] = lengthHorizontalWire[h + 1] * 2;
			h--;
		}
	}
}

void BankHv::CalculateRC() {
	if (!initialized) {
		cout << "[BankHv] Error: Require initialization first!" << endl;
	} else if (!invalid) {
		mat.CalculateRC();
	}
}

void BankHv::CalculateLatencyAndPower() {
	if (!initialized) {
		cout << "[BankHv] Error: Require initialization first!" << endl;
	} else if (invalid) {
		readLatency = writeLatency = 1e41;
		readDynamicEnergy = writeDynamicEnergy = 1e41;
		leakage = 1e41;
	} else {
		double latency;
		double energy;
		double leakageWire;
		int beta = 1;	/* Default value is 1. For fast access mode cache, this value is equal to associativity, which means only 1/beta interconnect wires are activated */

		mat.CalculateLatency(1e41 /* means Inf */);
		mat.CalculatePower();
		readLatency = mat.readLatency;
		writeLatency = mat.writeLatency;
		readDynamicEnergy = mat.readDynamicEnergy * numActiveMatPerRow * numActiveMatPerColumn;
		writeDynamicEnergy = mat.writeDynamicEnergy * numActiveMatPerRow * numActiveMatPerColumn;
		leakage = mat.leakage * numRowMat * numColumnMat;

		/* energy consumption on cells */
		cellReadEnergy = mat.cellReadEnergy * numActiveMatPerRow * numActiveMatPerColumn;
		cellSetEnergy = mat.cellSetEnergy * numActiveMatPerRow * numActiveMatPerColumn;
		cellResetEnergy = mat.cellResetEnergy * numActiveMatPerRow * numActiveMatPerColumn;

		/* for asymmetric RESET/SET only */
		resetLatency = mat.resetLatency;
		setLatency = mat.setLatency;
		resetDynamicEnergy = mat.resetDynamicEnergy * numActiveMatPerRow * numActiveMatPerColumn;
		setDynamicEnergy = mat.setDynamicEnergy * numActiveMatPerRow * numActiveMatPerColumn;

		if (inputParameter->designTarget == cache && inputParameter->cacheAccessMode == fast_access_mode)
			beta = inputParameter->associativity;

		/* horizontal wires */
		for (int i = 0; i < levelHorizontal; i++) {
			globalWire->CalculateLatencyAndPower(lengthHorizontalWire[i], &latency, &energy, &leakageWire);
			/* add latency */
			readLatency += latency * 2;		/* 2 due to in/out */
			writeLatency += latency;		/* only in */
			resetLatency += latency;
			setLatency += latency;
			/* consider number of wires */
			energy *= (numHorizontalWire[i] * numActiveHorizontalWireGroup[i]);
			leakageWire *= (numHorizontalWire[i] * numHorizontalWireGroup[i]);
			/* add power */
			readDynamicEnergy += energy;	/* only address input and data output */
			writeDynamicEnergy += energy / beta;	/* only input */
			resetDynamicEnergy += energy / beta;
			setDynamicEnergy += energy / beta;
			leakage += leakageWire;
		}

		/* vertical wires */
		for (int i = 0; i < levelVertical; i++) {
			globalWire->CalculateLatencyAndPower(lengthVerticalWire[i], &latency, &energy, &leakageWire);
			/* add latency */
			readLatency += latency * 2;		/* 2 due to in/out */
			writeLatency += latency;		/* only in */
			resetLatency += latency;
			setLatency += latency;
			/* consider number of wires */
			energy *= (numVerticalWire[i] * numActiveVerticalWireGroup[i]);
			leakageWire *= (numVerticalWire[i] * numVerticalWireGroup[i]);
			/* add power */
			readDynamicEnergy += energy;	/* only address input and data output */
			writeDynamicEnergy += energy / beta;	/* only input */
			resetDynamicEnergy += energy / beta;
			setDynamicEnergy += energy / beta;
			leakage += leakageWire;
		}
	}
}

BankHv & BankHv::operator=(const BankHv &rhs) {
	Bank::operator=(rhs);
	levelHorizontal = rhs.levelHorizontal;
	levelVertical = rhs.levelVertical;
	return *this;
}
