#pragma once

#include "PSNWhere_Types.h"

/////////////////////////////////////////////////////////////////////////
// OPERATOR
/////////////////////////////////////////////////////////////////////////
namespace psn
{
// matrix operation
template<typename _Tp> _Tp MatTotalSum(cv::Mat &inputMat)
{
	_Tp resultSum = 0;
	for(int rowIdx = 0; rowIdx < inputMat.rows; rowIdx++)
	{
		for(int colIdx = 0; colIdx < inputMat.cols; colIdx++)
		{
			resultSum += inputMat.at<_Tp>(rowIdx, colIdx);
		}
	}
	return resultSum;
}
template<typename _Tp> bool MatLowerThan(cv::Mat &inputMat, _Tp compValue)
{
	//_Tp maxValue = std::max_element(inputMat.begin(), inputMat.end());
	//for(int rowIdx = 0; rowIdx < inputMat.rows; rowIdx++)
	//{
	//	for(int colIdx = 0; colIdx < inputMat.cols; colIdx++)
	//	{
	//		if(inputMat.at<_Tp>(rowIdx, colIdx) >= compValue)
	//		{
	//			return false;
	//		}
	//	}
	//}
	return (std::max_element(inputMat.begin(), inputMat.end()) < compValue) ? true : false;
}
template<typename _Tp> bool MatContainLowerThan(cv::Mat &inputMat, _Tp compValue)
{
	for(int rowIdx = 0; rowIdx < inputMat.rows; rowIdx++)
	{
		for(int colIdx = 0; colIdx < inputMat.cols; colIdx++)
		{
			if(inputMat.at<_Tp>(rowIdx, colIdx) < compValue)
			{
				return true;
			}
		}
	}
	return false;
}
template<typename _Tp> std::vector<_Tp> mat2vec_C1(cv::Mat &inputMat)
{
	std::vector<_Tp> vecResult;

	for(int rowIdx = 0; rowIdx < inputMat.rows; rowIdx++)
	{
		for(int colIdx = 0; colIdx < inputMat.cols; colIdx++)
		{
			vecResult.push_back(inputMat.at<_Tp>(rowIdx, colIdx));
		}
	}
	
	return vecResult;
}

void appendRow(cv::Mat &dstMat, cv::Mat row);
void appendCol(cv::Mat &dstMat, cv::Mat col);

// math things
std::vector<std::vector<unsigned int>> nchoosek(int n, int k);
double erf(double x);
double erfc(double x);
cv::Mat histogram(cv::Mat singleChannelImage, int numBin);
bool IsLineSegmentIntersect(PSN_Line &line1, PSN_Line &line2);
double Triangulation(PSN_Line &line1, PSN_Line &line2, PSN_Point3D &midPoint3D);

// display related
std::vector<cv::Scalar> GenerateColors(unsigned int numColor);
cv::Scalar hsv2rgb(double h, double s, double v);
cv::Scalar getColorByID(unsigned int nID, std::vector<cv::Scalar> *vecColors = NULL);
void DrawBoxWithID(cv::Mat &imageFrame, PSN_Rect box, unsigned int nID, int lineStyle = 1, int fontSize = 0, std::vector<cv::Scalar> *vecColors = NULL);
void Draw3DBoxWithID(cv::Mat &imageFrame, std::vector<PSN_Point2D> &pointArray, unsigned int nID, std::vector<cv::Scalar> *vecColors = NULL);
void DrawTriangleWithID(cv::Mat &imageFrame, PSN_Point2D &point, unsigned int nID, std::vector<cv::Scalar> *vecColors = NULL);
void DrawLine(cv::Mat &imageFrame, std::vector<PSN_Point2D> &pointArray, unsigned int nID, int lineThickness = 1, std::vector<cv::Scalar> *vecColors = NULL);
cv::Mat MakeMatTile(std::vector<cv::Mat> *imageArray, unsigned int numRows, unsigned int numCols);

// database related coordinate transformation
PSN_Point2D GetLocationOnTopView_PETS2009(PSN_Point3D location, bool bZoom = false);

// file interface related
std::vector<stDetection> ReadDetectionResultWithTxt(std::string strDatasetPath, unsigned int camIdx, unsigned int frameIdx);
std::vector<stTrack2DResult> Read2DTrackResultWithTxt(std::string strDatasetPath, unsigned int frameIdx);
stTrack2DResult Read2DTrackResultWithTxt(std::string strDataPath, unsigned int camID, unsigned int frameIdx);
bool CreateDirectoryForWindows(const std::string &dirName);
void printLog(const char *filename, std::string strLog);
std::string MakeTrackIDList(PSN_TrackSet *tracks);
}
