#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <numeric>
#include <algorithm>
#include <iostream>
#include "PSNWhere_Hungarian.h"

//#define PSN_HUNGARIAN_DEBUG_DISPLAY

#ifndef PSHWHERE_DEFINES_H
#define PSN_P_INF_F (FLT_MAX)
#define PSN_N_INF_F (FLT_MIN)
#endif

bool psnIsOne(float value){ return 1.0f == value ? true : false; }

inline void CPSNWhere_Hungarian::psnMatSum(std::vector<std::vector<float>> &argInputMat, std::vector<float> &vecDest, unsigned int nDirection)
{
	unsigned int numCols = (unsigned int)argInputMat[0].size();
	unsigned int numRows = (unsigned int)argInputMat.size();
	vecDest.clear();

	switch(nDirection)
	{
	case 1: // column sum
		vecDest.resize(numCols, 0);
		for(unsigned int colIdx = 0; colIdx < numCols; colIdx++)
		{
			for(unsigned int rowIdx = 0; rowIdx < numRows; rowIdx++)
			{
				vecDest[colIdx] += argInputMat[rowIdx][colIdx];
			}
		}
		break;
	case 2: // row sum
		vecDest.resize(numRows, 0);
		for(unsigned int rowIdx = 0; rowIdx < numRows; rowIdx++)
		{
			for(unsigned int colIdx = 0; colIdx < numCols; colIdx++)
			{
				vecDest[rowIdx] += argInputMat[rowIdx][colIdx];
			}
		}
		break;
	default:
		// do nothing
		break;
	}
}

CPSNWhere_Hungarian::CPSNWhere_Hungarian(void)
{
	m_numRows = 0;
	m_numCols = 0;
	m_bInit = false;
}


CPSNWhere_Hungarian::~CPSNWhere_Hungarian(void)
{
}


void CPSNWhere_Hungarian::Initialize(std::vector<float> costArray, unsigned int numRows, unsigned int numCols) {
	if (!(numRows * numCols)) { return;	}
	if (costArray.size() != numRows * numCols) { assert("dimension mismatch!!"); }

	m_numRows = numRows;
	m_numCols = numCols;
	m_costMatrix.clear(); m_costMatrix.resize(m_numRows, std::vector<float>(m_numCols, 0));
	//m_costMatrix = cv::Mat(m_numRows, m_numCols, CV_32F);	
	for (unsigned int rowIdx = 0; rowIdx < m_numRows; rowIdx++) {
		for (unsigned int colIdx = 0; colIdx < m_numCols; colIdx++) {
			m_costMatrix[rowIdx][colIdx] = costArray[rowIdx*numCols+colIdx];
			if(_isnanf(m_costMatrix[rowIdx][colIdx])) {
				assert("illegal cost!! 1.#IND is contained");
				return;
			}
		}
	}
	m_matchInfo.rows.clear();
	m_matchInfo.cols.clear();
	m_matchInfo.costMatrix = m_costMatrix;
	m_matchInfo.totalCost = 0;
	m_bInit = true;
}

void CPSNWhere_Hungarian::Initialize(std::vector<std::vector<float>> costMatrix)
{
	if(!costMatrix.size())
	{
		return;
	}

	m_numRows = (unsigned int)costMatrix.size();
	m_numCols = (unsigned int)costMatrix[0].size();
	m_costMatrix.clear(); m_costMatrix.resize(m_numRows, std::vector<float>(m_numCols, 0));
	//m_costMatrix = cv::Mat(m_numRows, m_numCols, CV_32F);
	
	for(unsigned int rowIdx = 0; rowIdx < m_numRows; rowIdx++)
	{
		for(unsigned int colIdx = 0; colIdx < m_numCols; colIdx++)
		{
			m_costMatrix[rowIdx][colIdx] = costMatrix[rowIdx][colIdx];
			if(_isnanf(m_costMatrix[rowIdx][colIdx]))
			{
				assert("illegal cost!! 1.#IND is contained");
				return;
			}
		}
	}

	m_matchInfo.rows.clear();
	m_matchInfo.cols.clear();
	m_matchInfo.costMatrix = m_costMatrix;
	m_matchInfo.totalCost = 0;

	m_bInit = true;
}

void CPSNWhere_Hungarian::Initialize(float *costArray, unsigned int numRows, unsigned int numCols)
{
	if(!(numRows * numCols))
	{
		return;
	}

	m_numRows = numRows;
	m_numCols = numCols;
	m_costMatrix.clear(); m_costMatrix.resize(m_numRows, std::vector<float>(m_numCols, 0));
	//m_costMatrix = cv::Mat(m_numRows, m_numCols, CV_32F);

	for(unsigned int rowIdx = 0; rowIdx < m_numRows; rowIdx++)
	{
		for(unsigned int colIdx = 0; colIdx < m_numCols; colIdx++)
		{
			m_costMatrix[rowIdx][colIdx] = costArray[rowIdx*numCols+colIdx];
			if(_isnanf(m_costMatrix[rowIdx][colIdx]))
			{
				assert("illegal cost!! 1.#IND is contained");
				return;
			}
		}
	}

	m_matchInfo.rows.clear();
	m_matchInfo.cols.clear();
	m_matchInfo.costMatrix = m_costMatrix;
	m_matchInfo.totalCost = 0;

	m_bInit = true;
}

void CPSNWhere_Hungarian::Initialize(float **cost2DArray, unsigned int numRows, unsigned int numCols)
{
	if(!(numRows * numCols))
	{
		return;
	}

	m_numRows = numRows;
	m_numCols = numCols;
	m_costMatrix.clear(); m_costMatrix.resize(m_numRows, std::vector<float>(m_numCols, 0));
	//m_costMatrix = cv::Mat(m_numRows, m_numCols, CV_32F);

	for(unsigned int rowIdx = 0; rowIdx < m_numRows; rowIdx++)
	{
		for(unsigned int colIdx = 0; colIdx < m_numCols; colIdx++)
		{
			m_costMatrix[rowIdx][colIdx] = cost2DArray[rowIdx][colIdx];
			if(_isnanf(m_costMatrix[rowIdx][colIdx]))
			{
				assert("illegal cost!! 1.#IND is contained");
				return;
			}
		}
	}

	m_matchInfo.rows.clear();
	m_matchInfo.cols.clear();
	m_matchInfo.costMatrix = m_costMatrix;
	m_matchInfo.totalCost = 0;

	m_bInit = true;
}

void CPSNWhere_Hungarian::Finalize(void)
{
	if(!m_bInit)
	{
		return;
	}

	m_matchInfo.rows.clear();
	m_matchInfo.cols.clear();
	for(unsigned int rowIdx = 0; rowIdx < m_matchInfo.costMatrix.size(); rowIdx++)
	{
		m_matchInfo.costMatrix[rowIdx].clear();
		m_matchInfo.matchMatrix[rowIdx].clear();
	}
	m_matchInfo.costMatrix.clear();
	m_matchInfo.matchMatrix.clear();
	m_matchInfo.matchCosts.clear();
	m_matchInfo.totalCost = 0.0f;

	m_bInit = false;
}

stMatchInfo* CPSNWhere_Hungarian::Match()
{
	if(!m_bInit)
	{
		return &m_matchInfo;
	}

	// initialize variables
	m_matMatching.clear(); m_matMatching.resize(m_numRows, std::vector<float>(m_numCols, 0));

	// find the number in each column and row that are connected
	std::vector<unsigned int> numX;
	std::vector<unsigned int> numY;
	numX.clear(); numX.resize(m_numRows, 0);
	numY.clear(); numY.resize(m_numCols, 0);
	for(unsigned int colIdx = 0; colIdx < m_numCols; colIdx++)
	{
		for(unsigned int rowIdx = 0; rowIdx < m_numRows; rowIdx++)
		{
			unsigned int addVal = (unsigned int)_finitef(m_costMatrix[rowIdx][colIdx]);
			numX[rowIdx] += addVal;
			numY[colIdx] += addVal;
		}
	}

	// find the columns(vertices) and rows(vertices) that are isolated
	std::vector<unsigned int> xCon;
	std::vector<unsigned int> yCon;
	for(unsigned int idx = 0; idx < m_numRows; idx++){ if(numX[idx] > 0) { xCon.push_back(idx); } }
	for(unsigned int idx = 0; idx < m_numCols; idx++){ if(numY[idx] > 0) { yCon.push_back(idx); } }

	// assemble condensed performance matrix
	m_nSquareSize = std::max(m_numCols, m_numRows);
	m_matPCond.clear(); m_matPCond.resize(m_nSquareSize, std::vector<float>(m_nSquareSize, 0.0f));
	for(unsigned int rowIdx = 0; rowIdx < m_numRows; rowIdx++)
	{
		for(unsigned int colIdx = 0; colIdx < m_numCols; colIdx++)
		{
			m_matPCond[rowIdx][colIdx] = m_costMatrix[rowIdx][colIdx];
		}
	}

	// ensure that a perfect matching exists
	// calculate a form of the Edge matrix
	std::vector<std::vector<float>> matEdge(m_nSquareSize, std::vector<float>(m_nSquareSize, std::numeric_limits<float>::infinity()));
	float fPMaxVal = 0;
	for(unsigned int rowIdx = 0; rowIdx < m_nSquareSize; rowIdx++)
	{
		for(unsigned int colIdx = 0; colIdx < m_nSquareSize; colIdx++)
		{
			if(_finitef(m_matPCond[rowIdx][colIdx]))
			{
				matEdge[rowIdx][colIdx] = 0.0f; 
				if(m_matPCond[rowIdx][colIdx] > fPMaxVal)
				{
					fPMaxVal = m_matPCond[rowIdx][colIdx];
				}
			}
		}		
	}
	
	// find the deficiency(CNUM) in the edge matrix
	unsigned int cNum = this->minLineCover(matEdge);

	// project additional vertices and edges so that a perfect matching exists
	m_nSquareSize += cNum;
	m_matPCond.clear(); m_matPCond.resize(m_nSquareSize, std::vector<float>(m_nSquareSize, fPMaxVal));
	for(unsigned int rowIdx = 0; rowIdx < xCon.size(); rowIdx++)
	{
		for(unsigned int colIdx = 0; colIdx < yCon.size(); colIdx++)
		{
			m_matPCond[rowIdx][colIdx] = m_costMatrix[xCon[rowIdx]][yCon[colIdx]];
		}
	}


	///////////////////////////////////////////////////////
	// MAIN PROGRAM: CONTROLS WHICH STEP IS EXECUTED
	///////////////////////////////////////////////////////
	std::vector<std::vector<float>> matM;
	std::vector<float> rCov, cCov, zR, zC;

	bool bExitFlag = false;
	m_nStepNum = 1;
	while(!bExitFlag)
	{
		switch(m_nStepNum)
		{
		case 1:
			this->step1();
			break;
		case 2:			
			this->step2(rCov, cCov, matM, m_matPCond);
			break;
		case 3:
			this->step3(cCov, matM);
			break;
		case 4:
			this->step4(m_matPCond, rCov, cCov, matM, zR, zC);
			break;
		case 5:
			this->step5(matM, zR, zC, rCov, cCov);
			break;
		case 6:
			this->step6(m_matPCond, rCov, cCov);
			break;
		case 7:
			bExitFlag = true;
			break;
		default:
			// do nothing
			assert("Wrong case!");
			break;
		}
	}

	// remove all the virtual satellites and targets and uncondense
	// the matching to the size the original performance matrix

	// extract mathing information together
	m_matchInfo.rows.clear();
	m_matchInfo.cols.clear();
	m_matchInfo.matchCosts.clear();
	for(unsigned int rowIdx = 0; rowIdx < m_numRows; rowIdx++)
	{
		for(unsigned int colIdx = 0; colIdx < m_numCols; colIdx++)
		{
			m_matMatching[rowIdx][colIdx] = matM[rowIdx][colIdx];
			if(1 == m_matMatching[rowIdx][colIdx])
			{
				m_matchInfo.rows.push_back(rowIdx);
				m_matchInfo.cols.push_back(colIdx);
				m_matchInfo.matchCosts.push_back(this->m_costMatrix[rowIdx][colIdx]);
			}
		}
	}
	m_matchInfo.totalCost = std::accumulate(m_matchInfo.matchCosts.begin(), m_matchInfo.matchCosts.end(), 0.0f);
	m_matchInfo.matchMatrix = m_matMatching;	
	
	return &m_matchInfo;
}


void CPSNWhere_Hungarian::PrintCost(void)
{
	if(!m_bInit)
	{
		return;
	}

	for(unsigned int rowIdx = 0; rowIdx < m_numRows; rowIdx++)
	{
		std::cout << '[';
		for(unsigned int colIdx = 0; colIdx < m_numCols-1; colIdx++)
		{
			std::cout << m_costMatrix[rowIdx][colIdx] << '\t';
		}
		std::cout << m_costMatrix[rowIdx][m_numCols-1];
		std::cout << ']' << std::endl;
	}
}


void CPSNWhere_Hungarian::PrintMatch(void)
{
	if(!m_bInit)
	{
		return;
	}

	for(unsigned int rowIdx = 0; rowIdx < m_numRows; rowIdx++)
	{
		std::cout << '[';
		for(unsigned int colIdx = 0; colIdx < m_numCols-1; colIdx++)
		{
			std::cout << m_matMatching[rowIdx][colIdx] << '\t';
		}
		std::cout << m_matMatching[rowIdx][m_numCols-1];
		std::cout << ']' << std::endl;
	}
}

/**************************************************************************
 *   STEP 1: Find the smallest number of zeros in each row
 *           and subtract that minimum from its row
 **************************************************************************/
void CPSNWhere_Hungarian::step1(void) {
#ifdef PSN_HUNGARIAN_DEBUG_DISPLAY
	std::cout << "do step1" << std::endl;
#endif
	for (unsigned int rowIdx = 0; rowIdx < m_nSquareSize; rowIdx++) {
		// find min value
		float minValue = *std::min_element(m_matPCond[rowIdx].begin(), m_matPCond[rowIdx].end());
		// subtract with min value
		if (0.0f == minValue || std::numeric_limits<float>::infinity() == minValue ) { continue;	}
		for (unsigned int colIdx = 0; colIdx < m_nSquareSize; colIdx++) {
			m_matPCond[rowIdx][colIdx] -= minValue;
		}
	}
	m_nStepNum = 2;
}


/**************************************************************************  
 *   STEP 2: Find a zero in P_cond. If there are no starred zeros in its
 *           column or row start the zero. Repeat for each zero
 **************************************************************************/
void CPSNWhere_Hungarian::step2(
	std::vector<float> &rCov, 
	std::vector<float> &cCov,
	std::vector<std::vector<float>> &matM, 
	std::vector<std::vector<float>> &matPCond)
{
#ifdef PSN_HUNGARIAN_DEBUG_DISPLAY
	std::cout << "do step2" << std::endl;
#endif
	if (0 == matPCond.size()) { return; }
	unsigned int nSizeMatrix = (unsigned int)matPCond[0].size();

	// define variables	
	rCov.clear(); rCov.resize(nSizeMatrix, 0.0f);
	cCov.clear(); cCov.resize(nSizeMatrix, 0.0f);
	matM.clear(); matM.resize(nSizeMatrix, std::vector<float>(nSizeMatrix, 0.0f));

	for (unsigned int rowIdx = 0; rowIdx < nSizeMatrix; rowIdx++) {
		for (unsigned int colIdx = 0; colIdx < nSizeMatrix; colIdx++) {
			if (0.0f == matPCond[rowIdx][colIdx] && 0.0f == rCov[rowIdx] && 0.0f == cCov[colIdx]) {
				matM[rowIdx][colIdx] = 1.0f;
				rCov[rowIdx] = 1.0f;
				cCov[colIdx] = 1.0f;
			}
		}
	}

	// re-initialize the cover vectors
	rCov.clear(); rCov.resize(nSizeMatrix, 0.0f);
	cCov.clear(); cCov.resize(nSizeMatrix, 0.0f);

	m_nStepNum = 3;
}


/**************************************************************************
 *   STEP 3: Cover each column with a starred zero. If all the columns are
 *           covered then the matching is maximum
 **************************************************************************/
void CPSNWhere_Hungarian::step3(
	std::vector<float> &cCov, 
	std::vector<std::vector<float>> &matM)
{
#ifdef PSN_HUNGARIAN_DEBUG_DISPLAY
	std::cout << "do step3" << std::endl;
#endif
	if(matM.size() == 0) { assert("M matrix is empty"); }

	psnMatSum(matM, cCov, 1);
	if (matM[0].size() == std::accumulate(cCov.begin(), cCov.end(), 0.0f)) {
		m_nStepNum = 7;
	} else {
		m_nStepNum = 4;
	}
}


/**************************************************************************
 *   STEP 4: Find a noncovered zero and prime it.  If there is no starred
 *           zero in the row containing this primed zero, Go to Step 5.  
 *           Otherwise, cover this row and uncover the column containing 
 *           the starred zero. Continue in this manner until there are no 
 *           uncovered zeros left. Save the smallest uncovered value and 
 *           Go to Step 6.
 **************************************************************************/
void CPSNWhere_Hungarian::step4(
	std::vector<std::vector<float>> &matPCond, 
	std::vector<float> &rCov, 
	std::vector<float> &cCov, 
	std::vector<std::vector<float>> &matM, 
	std::vector<float> &zR, 
	std::vector<float> &zC)
{
#ifdef PSN_HUNGARIAN_DEBUG_DISPLAY
	std::cout << "do step4" << std::endl;
#endif
	unsigned int nSizeMatrix = (unsigned int)matPCond[0].size();
	bool zFlag = true;
	while (zFlag) {
		// find the first uncovered zero
		int row = -1, col = -1;
		bool bExitFlag = false;
		for (unsigned int rowIdx = 0; rowIdx < nSizeMatrix; rowIdx++) {
			for (unsigned int colIdx = 0; colIdx < nSizeMatrix; colIdx++) {
				if (0.0f == matPCond[rowIdx][colIdx] && 0.0f == rCov[rowIdx] && 0.0f == cCov[colIdx]) {
					row = (int)rowIdx;
					col = (int)colIdx;
					bExitFlag = true;
					break;
				}
			}
			if (bExitFlag){ break; }
		}

		if (-1 == row) {
			// if there are no uncovered zeros go to step 6
			m_nStepNum = 6;
			zFlag = false;
			zR.clear(); zR.resize(1, 0.0f);
			zC.clear(); zC.resize(1, 0.0f);
		} else {
			// prime the uncovered zero
			matM[row][col] = 2.0f;

			// if there is a starred zero in that row, cover the row and uncover the column containing the zero
			if (0 < std::count_if(matM[row].begin(), matM[row].end(), psnIsOne)) {
				rCov[row] = 1.0f;
				for (unsigned int colIdx = 0; colIdx < nSizeMatrix; colIdx++) {
					if (1.0f == matM[row][colIdx]) { cCov[colIdx] = 0.0f; }
				}
			} else {
				m_nStepNum = 5;
				zFlag = false;
				zR.clear(); zR.resize(1, (float)row);
				zC.clear(); zC.resize(1, (float)col);
			}
		}
	}
}


/**************************************************************************
 * STEP 5: Construct a series of alternating primed and starred zeros as
 *         follows.  Let Z0 represent the uncovered primed zero found in Step 4.
 *         Let Z1 denote the starred zero in the column of Z0 (if any). 
 *         Let Z2 denote the primed zero in the row of Z1 (there will always
 *         be one).  Continue until the series terminates at a primed zero
 *         that has no starred zero in its column.  Unstar each starred 
 *         zero of the series, star each primed zero of the series, erase 
 *         all primes and uncover every line in the matrix.  Return to Step 3.
 **************************************************************************/
void CPSNWhere_Hungarian::step5(
	std::vector<std::vector<float>> &matM,
	std::vector<float> &zR,
	std::vector<float> &zC,
	std::vector<float> &rCov,
	std::vector<float> &cCov)
{
#ifdef PSN_HUNGARIAN_DEBUG_DISPLAY
	std::cout << "do step5" << std::endl;
#endif
	unsigned int nMatrixSize = (unsigned int)matM[0].size();
	bool zFlag = true;

	unsigned int row = 0;
	while (zFlag) {
		// find the index number of the starred zero in the column
		int rIndex = -1;
		for (unsigned int rowIdx = 0; rowIdx < nMatrixSize; rowIdx++) {
			if (1.0f == matM[rowIdx][(unsigned int)zC[row]]) {
				rIndex = (int)rowIdx;
				break;
			}
		}

		if (rIndex >= 0) {
			// save the starred zero
			row++;			
			// save the row of the starred zero
			zR.push_back((float)rIndex);
			// the column of the starred zero is the same as the column of the primed zero
			zC.push_back(zC[row-1]);
		} else {
			zFlag = false;
		}

		// continue if there is a starred zero in the column of the primed zero
		if (zFlag) {
			// find the column of the primed zero in the last starred zeros row
			unsigned int cIndex = 0;
			for (unsigned int colIdx = 0; colIdx < nMatrixSize; colIdx++) {
				if (2.0f == matM[(unsigned int)zR[row]][colIdx]) {
					cIndex = colIdx;
					break;
				}
			}
			row++;
			zR.push_back(zR[row-1]);
			zC.push_back((float)cIndex);
		}
	}

	// UNSTAR all the starred zeros in the path and STAR all primed zeros
	for (unsigned int rowIdx = 0; rowIdx < zR.size(); rowIdx++) {
		if (1.0f == matM[(unsigned int)zR[rowIdx]][(unsigned int)zC[rowIdx]]) {
			matM[(unsigned int)zR[rowIdx]][(unsigned int)zC[rowIdx]] = 0.0f;
		} else {
			matM[(unsigned int)zR[rowIdx]][(unsigned int)zC[rowIdx]] = 1.0f;
		}
	}

	// clear the covers
	rCov.clear(); rCov.resize(nMatrixSize, 0.0f);
	cCov.clear(); cCov.resize(nMatrixSize, 0.0f);

	// remove all the primes
	for (unsigned int rowIdx = 0; rowIdx < nMatrixSize; rowIdx++) {
		for (unsigned int colIdx = 0; colIdx < nMatrixSize; colIdx++) {
			if (2 == matM[rowIdx][colIdx]) {
				matM[rowIdx][colIdx] = 0.0f;
			}
		}
	}

	m_nStepNum = 3;
}


/**************************************************************************
 * STEP 6: Add the minimum uncovered value to every element of each covered
 *         row, and subtract it from every element of each uncovered column.  
 *         Return to Step 4 without altering any stars, primes, or covered lines.
 **************************************************************************/
void CPSNWhere_Hungarian::step6(
	std::vector<std::vector<float>> &matPCond,
	std::vector<float> &rCov,
	std::vector<float> &cCov)
{
#ifdef PSN_HUNGARIAN_DEBUG_DISPLAY
	std::cout << "do step6" << std::endl;
#endif
	float fMinVal = std::numeric_limits<float>::infinity();
	for(unsigned int rowIdx = 0; rowIdx < m_nSquareSize; rowIdx++)
	{
		if(0 != rCov[rowIdx]){ continue; }
		for(unsigned int colIdx = 0; colIdx < m_nSquareSize; colIdx++)
		{
			if(0 != cCov[colIdx]){ continue; }
			if(matPCond[rowIdx][colIdx] < fMinVal)
			{
				fMinVal = matPCond[rowIdx][colIdx];
			}
		}
	}

	float rowBiasVal = 0.0f, colBiasVal = 0.0f;
	for(unsigned int rowIdx = 0; rowIdx < m_nSquareSize; rowIdx++)
	{
		rowBiasVal = 0.0f;
		if(1 == rCov[rowIdx]){ rowBiasVal = fMinVal; }
		for(unsigned int colIdx = 0; colIdx < m_nSquareSize; colIdx++)
		{
			colBiasVal = 0.0f;
			if(0 == cCov[colIdx]){ colBiasVal = -fMinVal; }
			matPCond[rowIdx][colIdx] += rowBiasVal + colBiasVal;
		}
	}

	m_nStepNum = 4;
}

unsigned int CPSNWhere_Hungarian::minLineCover(std::vector<std::vector<float>> &argMatrix)
{
#ifdef PSN_HUNGARIAN_DEBUG_DISPLAY
	std::cout << "do minLineCover" << std::endl;
#endif
	unsigned int nMatrixSize = (unsigned int)argMatrix[0].size();
	if(0 == nMatrixSize)
	{
		assert("minLineCover: empty argMatrix");
		return 0;
	}

	std::vector<std::vector<float>> matM;
	std::vector<float> rCov;
	std::vector<float> cCov;
	std::vector<float> zR;
	std::vector<float> zC;
	unsigned int cNum = nMatrixSize;

	// step 2
	this->step2(rCov, cCov, matM, argMatrix);

	// step 3
	this->step3(cCov, matM);

	// step 4
	this->step4(argMatrix, rCov, cCov, matM, zR, zC);

	// calculate the deficiency
	cNum -= (unsigned int)(std::accumulate(rCov.begin(), rCov.end(), 0.0f) + std::accumulate(cCov.begin(), cCov.end(), 0.0f));
	
	return cNum;
}

//()()
//('')HAANJU.YOO

