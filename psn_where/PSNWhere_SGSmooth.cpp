#include "PSNWhere_SGSmooth.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <cv.h>

#define SGS_DEFAULT_SPAN (9)
#define SGS_DEFAULT_DEGREE (2)

CPSNWhere_SGSmooth::CPSNWhere_SGSmooth(void)
	: span_(SGS_DEFAULT_SPAN)
	, degree_(SGS_DEFAULT_DEGREE)
	, Qcols_(0)
	, Qrows_(0)
{
}

CPSNWhere_SGSmooth::CPSNWhere_SGSmooth(int span, int degree, std::vector<double> *initialData)
	: span_(span)
	, degree_(degree)
	, Qcols_(0)
	, Qrows_(0)
{
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
	bool bEntireUpdate = RecalculateQ(windowSize);
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
				smoothedData_[smoothDataPos] += Qbegin_[pos] * data_[colIdx];
			}
		}

		// middle
		smoothedMid = Filter(Qmid_, data_);
		midStartPos = windowSize - 1;
	}
	else
	{
		// middle
		refreshPos = (int)smoothedData_.size() - halfWindowSize;		
		smoothedMid = Filter(Qmid_, data_, (int)smoothedData_.size());
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
			curSmoothedData += Qend_[pos] * data_[colIdx];
		}
		smoothedData_.push_back(curSmoothedData);
	}
	
	return refreshPos;
}

bool CPSNWhere_SGSmooth::RecalculateQ(int windowSize)
{
	if (Qrows_ == windowSize) { return false; }

	int halfWindowSize = (windowSize - 1) / 2;
	//cv::Mat V = cv::Mat::ones(numFrame, degree_+1, CV_64F);
	
	// Qmid
	Qmid_.clear();
	Qmid_.resize(windowSize, 1.0 / (double)windowSize);

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
	Q_.clear();
	Qrows_ = windowSize;
	Qcols_ = degree_ + 1;
	Q_.resize(Qrows_* Qcols_, 0.0);

	int preColPos = 0;
	double columnNorm = 0.0;
	for (int colIdx = 0; colIdx < Qcols_; colIdx++)
	{
		std::vector<double> projectionsMagnitude(std::max(0, colIdx), 0.0);
		for (int rowIdx = 0, pos = colIdx; rowIdx < Qrows_; rowIdx++, pos += Qcols_)
		{
			Q_[pos] = V[pos];
			// for Q(preCol)' * V
			for (int preColIdx = 0, preColPos = pos - colIdx; preColIdx < colIdx; preColIdx++, preColPos++)
			{
				projectionsMagnitude[preColIdx] += Q_[preColPos] * V[pos];
			}
		}

		// orthogonalization
		columnNorm = 0.0;
		for (int rowIdx = 0, pos = colIdx; rowIdx < Qrows_; rowIdx++, pos+=Qcols_)
		{
			preColPos = pos - colIdx;
			for (int preColIdx = 0; preColIdx < colIdx; preColIdx++, preColPos++)
			{
				Q_[pos] -= projectionsMagnitude[preColIdx] * Q_[preColPos];
			}
			columnNorm += Q_[pos] * Q_[pos];
		}
		columnNorm = std::sqrt(columnNorm);

		// normalization
		for (int rowIdx = 0, pos = colIdx; rowIdx < Qrows_; rowIdx++, pos += Qcols_)
		{
			Q_[pos] /= columnNorm;
		}		
	}

	// Qbegin: q(1:hf,:)*q'
	Qbegin_.clear();
	Qbegin_.resize(halfWindowSize * Qrows_, 0.0);

	// Qend: q((hf+2):end,:)*q'
	Qend_.clear();
	Qend_.resize(halfWindowSize * Qrows_, 0.0);
	
	int posFrontHalf = 0;
	int posBackHalf = (halfWindowSize + 1) * Qcols_;	
	for (int rowIdx = 0, pos = 0; rowIdx < halfWindowSize; rowIdx++)
	{
		int posQ = 0;
		for (int colIdx = 0; colIdx < Qrows_; colIdx++, pos++)
		{
			for (int elementIdx = 0; elementIdx < Qcols_; elementIdx++, posQ++)
			{
				Qbegin_[pos] += Q_[posQ] * Q_[posFrontHalf + elementIdx];
				Qend_[pos] += Q_[posQ] * Q_[posBackHalf + elementIdx];
			}
		}
		posFrontHalf += Qcols_;
		posBackHalf += Qcols_;
	}

	return true;
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

