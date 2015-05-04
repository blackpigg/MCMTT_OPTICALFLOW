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
	CPSNWhere_SGSmooth(int span, int degree);
	~CPSNWhere_SGSmooth(void);
	void Insert(std::deque<double> &queueNewData);
	double Insert(double newData);
	double GetResult(int index);
private:
	void Smoothing(void);
	void RecalculateQ(int numFrame);

	std::deque<double> data_;
	std::deque<double> smoothedData_;
	int span_;
	int degree_;
	std::vector<double> Q_;
	int Qcols_;
	int Qrows_;
};

