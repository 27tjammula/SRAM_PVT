/*
 * global.h
 *
 *  Created on: Mar 18, 2010
 *      Author: xydong
 */

#include "InputParameter.h"
#include "Technology.h"
#include "MemCell.h"
#include "Wire.h"

extern InputParameter *inputParameter;
extern Technology *tech;
extern MemCell *cell;
extern Wire *localWire;		/* The wire type of local interconnects (for example, wire in mat) */
extern Wire *globalWire;	/* The wire type of global interconnects (for example, the ones that connect mats) */
