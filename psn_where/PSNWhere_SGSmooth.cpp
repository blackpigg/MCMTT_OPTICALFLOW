#include "PSNWhere_SGSmooth.h"
#include <stdio.h>
#include <stddef.h>
#include <math.h>
#include <assert.h>

#define SGS_DEFAULT_SPAN (9)
#define SGS_DEFAULT_DEGREE (1)

CPSNWhere_SGSmooth::CPSNWhere_SGSmooth(void)
	: span_(SGS_DEFAULT_SPAN)
	, degree_(SGS_DEFAULT_DEGREE)
{
	Qset_.cols = 0;
	Qset_.rows = 0;
}

CPSNWhere_SGSmooth::CPSNWhere_SGSmooth(int span, int degree, std::vector<double> *initialData)
	: span_(span)
	, degree_(degree)
{
	Qset_.cols = 0;
	Qset_.rows = 0;
	if (NULL == initialData) { return; }
	data_.insert(data_.begin(), initialData->begin(), initialData->end());
}

CPSNWhere_SGSmooth::~CPSNWhere_SGSmooth(void)
{
	data_.clear();
	smoothedData_.clear();
}

int CPSNWhere_SGSmooth::Insert(double newData)
{
	data_.push_back(newData);	
	return Smoothing();
}

int CPSNWhere_SGSmooth::Insert(std::vector<double> &queueNewData)
{
	data_.insert(data_.end(), queueNewData.begin(), queueNewData.end());
	return Smoothing();
}

double CPSNWhere_SGSmooth::GetResult(int pos)
{
	assert(pos < smoothedData_.size());
	return smoothedData_[pos];
}

std::vector<double> CPSNWhere_SGSmooth::GetResult(int startPos, int endPos)
{
	assert(endPos < smoothedData_.size() && startPos <= endPos);
	int numPoints = endPos - startPos + 1;
	std::vector<double> results;
	results.reserve(numPoints);
	for (int pos = startPos; pos <= endPos; pos++)
	{
		results.push_back(smoothedData_[pos]);
	}
	return results;
}

Qset* CPSNWhere_SGSmooth::CalculateQ(int windowSize)
{
	if (Qset_.rows == windowSize) { return false; }

	int halfWindowSize = (windowSize - 1) / 2;
	//cv::Mat V = cv::Mat::ones(numFrame, degree_+1, CV_64F);
	
	// Qmid
	Qset_.Qmid.clear();
	Qset_.Qmid.resize(windowSize, 1.0 / (double)windowSize);

	// direction => row
	int pos = 0;
	std::vector<double> V(windowSize * (degree_+1), 1.0);	
	for (int order = 1; order <= degree_; order++)
	{
		for (int timeIdx = -halfWindowSize, pos = order; timeIdx <= halfWindowSize; timeIdx++, pos += degree_ + 1)
		{
			V[pos] = std::pow((double)timeIdx, (double)order);
		}
	}

	// find Q (recuded QR factorization, we don't need R)
	Qset_.Q.clear();
	Qset_.rows = windowSize;
	Qset_.cols = degree_ + 1;
	Qset_.Q.resize(Qset_.rows* Qset_.cols, 0.0);

	int preColPos = 0;
	double columnNorm = 0.0;
	for (int colIdx = 0; colIdx < Qset_.cols; colIdx++)
	{
		std::vector<double> projectionsMagnitude(std::max(0, colIdx), 0.0);
		for (int rowIdx = 0, pos = colIdx; rowIdx < Qset_.rows; rowIdx++, pos += Qset_.cols)
		{
			Qset_.Q[pos] = V[pos];
			// for Q(preCol)' * V
			for (int preColIdx = 0, preColPos = pos - colIdx; preColIdx < colIdx; preColIdx++, preColPos++)
			{
				projectionsMagnitude[preColIdx] += Qset_.Q[preColPos] * V[pos];
			}
		}

		// orthogonalization
		columnNorm = 0.0;
		for (int rowIdx = 0, pos = colIdx; rowIdx < Qset_.rows; rowIdx++, pos+=Qset_.cols)
		{
			preColPos = pos - colIdx;
			for (int preColIdx = 0; preColIdx < colIdx; preColIdx++, preColPos++)
			{
				Qset_.Q[pos] -= projectionsMagnitude[preColIdx] * Qset_.Q[preColPos];
			}
			columnNorm += Qset_.Q[pos] * Qset_.Q[pos];
		}
		columnNorm = std::sqrt(columnNorm);

		// normalization
		for (int rowIdx = 0, pos = colIdx; rowIdx < Qset_.rows; rowIdx++, pos += Qset_.cols)
		{
			Qset_.Q[pos] /= columnNorm;
		}		
	}

	// Qbegin: q(1:hf,:)*q'
	Qset_.Qbegin.clear();
	Qset_.Qbegin.resize(halfWindowSize * Qset_.rows, 0.0);

	// Qend: q((hf+2):end,:)*q'
	Qset_.Qend.clear();
	Qset_.Qend.resize(halfWindowSize * Qset_.rows, 0.0);
	
	int posFrontHalf = 0;
	int posBackHalf = (halfWindowSize + 1) * Qset_.cols;	
	for (int rowIdx = 0, pos = 0; rowIdx < halfWindowSize; rowIdx++)
	{
		int posQ = 0;
		for (int colIdx = 0; colIdx < Qset_.rows; colIdx++, pos++)
		{
			for (int elementIdx = 0; elementIdx < Qset_.cols; elementIdx++, posQ++)
			{
				Qset_.Qbegin[pos] += Qset_.Q[posQ] * Qset_.Q[posFrontHalf + elementIdx];
				Qset_.Qend[pos] += Qset_.Q[posQ] * Qset_.Q[posBackHalf + elementIdx];
			}
		}
		posFrontHalf += Qset_.cols;
		posBackHalf += Qset_.cols;
	}

	return &Qset_;
}

int CPSNWhere_SGSmooth::Smoothing(void)
{
	int refreshPos = 0;
	int numData = (int)data_.size();
	int windowSize = std::min(span_, numData); 
	windowSize -= (windowSize + 1) % 2; // will subtract 1 if frame is even.
	if (windowSize <= degree_) // bypass
	{
		refreshPos = (int)smoothedData_.size();
		for (int idx = (int)smoothedData_.size(); idx < data_.size(); idx++)
		{
			smoothedData_.push_back(data_[idx]);
		}
		return refreshPos;
	}

	// smoothing
	int halfWindowSize = (windowSize - 1) / 2;
	int midStartPos = 0;
	bool bEntireUpdate = NULL == CalculateQ(windowSize)? false : true;
	std::vector<double> smoothedMid;
	if(bEntireUpdate)
	{
		// begin
		smoothedData_.clear();
		smoothedData_.resize(halfWindowSize, 0.0);
		for (int pos = 0, smoothDataPos = 0; smoothDataPos < halfWindowSize; smoothDataPos++)
		{
			for (int colIdx = 0; colIdx < windowSize; colIdx++, pos++)
			{
				smoothedData_[smoothDataPos] += Qset_.Qbegin[pos] * data_[colIdx];
			}
		}

		// middle
		smoothedMid = Filter(Qset_.Qmid, data_);
		midStartPos = windowSize - 1;
	}
	else
	{
		// middle
		refreshPos = (int)smoothedData_.size() - halfWindowSize;		
		smoothedMid = Filter(Qset_.Qmid, data_, (int)smoothedData_.size());
		smoothedData_.erase(smoothedData_.begin() + refreshPos, smoothedData_.end());
	}

	// middle
	smoothedData_.insert(smoothedData_.end(), smoothedMid.begin() + midStartPos, smoothedMid.end());

	// end
	for (int pos = 0, smoothDataPos = (int)data_.size() - halfWindowSize; smoothDataPos < data_.size(); smoothDataPos++)
	{
		double curSmoothedData = 0.0;
		for (int colIdx = (int)data_.size() - windowSize; colIdx < (int)data_.size(); colIdx++, pos++)
		{
			curSmoothedData += Qset_.Qend[pos] * data_[colIdx];
		}
		smoothedData_.push_back(curSmoothedData);
	}
	
	return refreshPos;
}

std::vector<double> CPSNWhere_SGSmooth::Filter(std::vector<double> &coefficients, std::deque<double> &data, int startPos)
{
	std::vector<double> results(data.size() - startPos, 0.0);
	for (int resultPos = 0; resultPos < data.size() - startPos; resultPos++)
	{
		int dataPos = startPos + resultPos;
		for (int coeffPos = 0; coeffPos < coefficients.size() && 0 <= dataPos; coeffPos++, dataPos--)
		{
			results[resultPos] += coefficients[coeffPos] * data[dataPos];
		}		
	}
	return results;
}

//()()
//('')HAANJU.YOO

