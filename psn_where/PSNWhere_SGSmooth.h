/******************************************************************************
 * Name : CPSNWhere_SGSmooth
 * Date : 2015.05.04
 * Author : HAANJU.YOO
 * Version : 0.9
 * Description :
 *	code is based on 
 *	- 'http://www.ks.uiuc.edu/Research/vmd/mailing_list/vmd-l/att-13397/sgsmooth.C'
 *	- MATLAB's 'smooth.m'
 *
 ******************************************************************************/
#pragma once
#include <stddef.h>
#include <queue>

class CPSNWhere_SGSmooth
{
public:
	CPSNWhere_SGSmooth(void);
	CPSNWhere_SGSmooth(int span, int degree, std::vector<double> *initialData = NULL);
	~CPSNWhere_SGSmooth(void);
	int Insert(double newData);
	int Insert(std::vector<double> &queueNewData);	
	double GetResult(int pos);
	std::vector<double> GetResult(int startPos, int endPos);
private:
	int Smoothing(void);
	bool RecalculateQ(int windowSize);
	std::vector<double> Filter(std::vector<double> &coefficients, std::deque<double> &data, int startPos = 0);

	std::deque<double> data_;
	std::deque<double> smoothedData_;
	int span_;
	int degree_;
	std::vector<double> Q_;
	std::vector<double> Qbegin_;
	std::vector<double> Qmid_;	
	std::vector<double> Qend_;
	int Qcols_;
	int Qrows_;
};

