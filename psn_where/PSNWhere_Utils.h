#pragma once

#include <vector>
#include "PSNWhere_Types.h"

namespace psn
{
	// matrix operation
	template<typename _Tp> _Tp MatTotalSum(cv::Mat &inputMat);
	template<typename _Tp> bool MatLowerThan(cv::Mat &inputMat, _Tp compValue);
	template<typename _Tp> bool MatContainLowerThan(cv::Mat &inputMat, _Tp compValue);
	template<typename _Tp> std::vector<_Tp> mat2vec_C1(cv::Mat &inputMat);
	void appendRow(cv::Mat &dstMat, cv::Mat &row);
	void appendCol(cv::Mat &dstMat, cv::Mat &col);
	// math things
	void nchoosek(int n, int k, std::deque<std::vector<unsigned int>> &outputCombinations);
	double erf(double x);
	double erfc(double x);
	cv::Mat histogram(cv::Mat singleChannelImage, int numBin);
	double Triangulation(PSN_Line &line1, PSN_Line &line2, PSN_Point3D &midPoint3D);
	// visualization related
	cv::Mat MakeMatTile(std::vector<cv::Mat> *imageArray, unsigned int numRows, unsigned int numCols);	
	std::vector<cv::Scalar> GenerateColors(unsigned int numColor);
	cv::Scalar hsv2rgb(double h, double s, double v);
	cv::Scalar getColorByID(std::vector<cv::Scalar> &vecColors, unsigned int nID);
	void DrawBoxWithID(cv::Mat &imageFrame, PSN_Rect curRect, unsigned int nID, std::vector<cv::Scalar> &vecColors);
	void DrawBoxWithLargeID(cv::Mat &imageFrame, PSN_Rect curRect, unsigned int nID, std::vector<cv::Scalar> &vecColors, bool bDashed = false);
	void Draw3DBoxWithID(cv::Mat &imageFrame, std::vector<PSN_Point2D> &pointArray, unsigned int nID, std::vector<cv::Scalar> &vecColors);
	void DrawTriangleWithID(cv::Mat &imageFrame, PSN_Point2D &point, unsigned int nID, std::vector<cv::Scalar> &vecColors);
	void DrawLine(cv::Mat &imageFrame, std::vector<PSN_Point2D> &pointArray, unsigned int nID, std::vector<cv::Scalar> &vecColors, int lineThickness = 2);
	PSN_Point2D GetLocationOnTopView_PETS2009(PSN_Point3D &curPoint, bool bZoom);
	// file interface
	void printLog(const char *filename, const char *strLog);
	std::string MakeTrackIDList(PSN_TrackSet *tracks);
	std::vector<stDetection> ReadDetectionResultWithTxt(std::string strDatasetPath, unsigned int camIdx, unsigned int frameIdx);
	std::vector<stTrack2DResult> Read2DTrackResultWithTxt(std::string strDatasetPath, unsigned int frameIdx);
}