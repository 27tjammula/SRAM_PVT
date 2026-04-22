/*
 * Bank.cpp
 *
 *  Created on: Jul 9, 2010
 *      Author: Xiangyu
 */

#include "Bank.h"

Bank::Bank() {
	// TODO Auto-generated constructor stub
	initialized = false;
	invalid = false;
}

Bank::~Bank() {
	// TODO Auto-generated destructor stub
}

void Bank::PrintProperty() {
	cout << "Bank Properties:" << endl;
	FunctionUnit::PrintProperty();
}

Bank & Bank::operator=(const Bank &rhs) {
	height = rhs.height;
	width = rhs.width;
	area = rhs.area;
	readLatency = rhs.readLatency;
	writeLatency = rhs.writeLatency;
	readDynamicEnergy = rhs.readDynamicEnergy;
	writeDynamicEnergy = rhs.writeDynamicEnergy;
	resetLatency = rhs.resetLatency;
	setLatency = rhs.setLatency;
	resetDynamicEnergy = rhs.resetDynamicEnergy;
	setDynamicEnergy = rhs.setDynamicEnergy;
	cellReadEnergy = rhs.cellReadEnergy;
	cellSetEnergy = rhs.cellSetEnergy;
	cellResetEnergy = rhs.cellResetEnergy;
	leakage = rhs.leakage;
	initialized = rhs.initialized;
	invalid = rhs.invalid;
	numRowMat = rhs.numRowMat;
	numColumnMat = rhs.numColumnMat;
	capacity = rhs.capacity;
	blockSize = rhs.blockSize;
	associativity = rhs.associativity;
	numRowPerSet = rhs.numRowPerSet;
	numActiveMatPerRow = rhs.numActiveMatPerRow;
	numActiveMatPerColumn = rhs.numActiveMatPerColumn;
	muxSenseAmp = rhs.muxSenseAmp;
	internalSenseAmp = rhs.internalSenseAmp;
	muxOutputLev1 = rhs.muxOutputLev1;
	muxOutputLev2 = rhs.muxOutputLev2;
	areaOptimizationLevel = rhs.areaOptimizationLevel;
	memoryType = rhs.memoryType;
	numRowSubarray = rhs.numRowSubarray;
	numColumnSubarray = rhs.numColumnSubarray;
	numActiveSubarrayPerRow = rhs.numActiveSubarrayPerRow;
	numActiveSubarrayPerColumn = rhs.numActiveSubarrayPerColumn;
	mat = rhs.mat;
	return *this;
}
