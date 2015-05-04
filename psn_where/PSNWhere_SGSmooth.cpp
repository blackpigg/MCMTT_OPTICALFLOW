#include "PSNWhere_SGSmooth.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <cv.h>

//! default convergence
static const double TINY_FLOAT = 1.0e-300;
//! comfortable array of doubles
typedef std::vector<double> float_vect;
//! comfortable array of ints;
typedef std::vector<int>    int_vect;

CPSNWhere_SGSmooth::CPSNWhere_SGSmooth(int span, int degree)
{
	span_ = span;
	degree_ = degree;
}

CPSNWhere_SGSmooth::~CPSNWhere_SGSmooth(void)
{
	data_.clear();
	smoothedData_.clear();
}

void CPSNWhere_SGSmooth::Insert(std::deque<double> &queueNewData)
{
	data_.insert(data_.end(), queueNewData.begin(), queueNewData.end());
	smoothing();
}

double CPSNWhere_SGSmooth::Insert(double newData)
{
	data_.push_back(newData);	
	smoothing();
	return smoothedData_.back();
}

double CPSNWhere_SGSmooth::GetResult(int index)
{
	assert(index < smoothedData_.size());
	return smoothedData_[index];
}

void CPSNWhere_SGSmooth::Smoothing(void)
{
	int numData = (int)data_.size();
	int numFrame = std::min(span_, numData) / 2 * 2; // will subtract 1 if frame is even.
	if (numFrame <= degree_)
	{
		for (int idx = smoothedData_.size(); idx < data_.size(); idx++)
		{
			smoothedData_.push_back(data_[idx]);
		}
		return;
	}

	int halfFrameLength = (numFrame - 1) / 2;
	cv::Mat V = cv::Mat::ones(numFrame, degree_+1, CV_64F);
	for (int order = 1; order <= degree_; order++)
	{
		for (int timeIdx = -halfFrameLength; timeIdx <= halfFrameLength; timeIdx++)
		{
			V.at<double>(timeIdx, order) = std::pow((double)timeIdx, (double)order);
		}
	}
}

void CPSNWhere_SGSmooth::RecalculateQ(int numFrame)
{
	if (Qrows_ == numFrame) { return; }

	int halfFrameLength = (numFrame - 1) / 2;
	//cv::Mat V = cv::Mat::ones(numFrame, degree_+1, CV_64F);
	
	// direction => row
	std::vector<double> V(numFrame * (degree_+1));
	int pos = 0;

	for (int order = 1; order <= degree_; order++)
	{
		pos = order;
		for (int timeIdx = -halfFrameLength; timeIdx <= halfFrameLength; timeIdx++, pos+=degree_+1)
		{
			V[pos] = std::pow((double)timeIdx, (double)order);
		}
	}

	// find Q (we don't need R)
	Q_.clear();
	Qrows_ = numFrame;
	Qcols_ = degree_ + 1;
	Q_.resize(Qrows_* Qcols_);

	int pos = 0;
	for (int colIdx = 0; colIdx <= Qcols_; colIdx++)
	{
		pos = colIdx;
		std::vector<double> projectionsMagnitude(std::max(0, colIdx-1), 0.0);
		for (int rowIdx = 0; rowIdx < Qrows_; rowIdx++, pos+=Qcols_)
		{
			Q_[pos] = V[pos];
		}
		Q_.col(colIdx) = V.col(colIdx).clone();
		for (int preColIdx = 0; preColIdx < colIdx; preColIdx++)
		{
			Q_.col(colIdx) -= Q_.col(preColIdx).dot(V.col(colIdx)) * Q_.col(preColIdx);
		}
		double curNorm = 0.0;
		
	}
}

//()()
//('')HAANJU.YOO

