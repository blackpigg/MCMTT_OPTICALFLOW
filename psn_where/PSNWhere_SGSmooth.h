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
#include <queue>

struct Qset
{
	int rows;
	int cols;
	std::vector<double> Q;
	std::vector<double> Qbegin;
	std::vector<double> Qmid;
	std::vector<double> Qend;
};

class CPSNWhere_SGSmooth
{
public:
	CPSNWhere_SGSmooth(void);
	CPSNWhere_SGSmooth(int span, int degree, std::vector<double> *initialData = NULL);
	~CPSNWhere_SGSmooth(void);

	// setter
	int Insert(double newData);
	int Insert(std::vector<double> &queueNewData);
	void SetQ(Qset &Q) { Qset_ = Q; };

	// getter
	double GetResult(int pos);
	std::vector<double> GetResult(int startPos, int endPos);

	// others
	Qset* CalculateQ(int windowSize);
	
private:
	int Smoothing(void);	
	std::vector<double> Filter(std::vector<double> &coefficients, std::deque<double> &data, int startPos = 0);

	int span_;
	int degree_;
	Qset Qset_;
	std::deque<double> data_;
	std::deque<double> smoothedData_;	
	//std::vector<double> Q_;
	//std::vector<double> Qbegin_;
	//std::vector<double> Qmid_;	
	//std::vector<double> Qend_;
	//int Qcols_;
	//int Qrows_;
};

