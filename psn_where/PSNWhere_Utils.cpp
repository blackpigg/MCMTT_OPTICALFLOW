#include "PSNWhere_Utils.h"

/////////////////////////////////////////////////////////////////////////
// PREDEFINE
/////////////////////////////////////////////////////////////////////////

// reference for erf function: http://math.stackexchange.com/questions/263216/error-function-erf-with-better-precision
static const double psn_tiny = 1e-300,
psn_half= 5.00000000000000000000e-01, /* 0x3FE00000, 0x00000000 */
psn_one = 1.00000000000000000000e+00, /* 0x3FF00000, 0x00000000 */
psn_two = 2.00000000000000000000e+00, /* 0x40000000, 0x00000000 */
/* c = (float)0.84506291151 */
psn_erx = 8.45062911510467529297e-01, /* 0x3FEB0AC1, 0x60000000 */
/*
* Coefficients for approximation to erf on [0,0.84375]
*/
psn_efx = 1.28379167095512586316e-01, /* 0x3FC06EBA, 0x8214DB69 */
psn_efx8= 1.02703333676410069053e+00, /* 0x3FF06EBA, 0x8214DB69 */
psn_pp0 = 1.28379167095512558561e-01, /* 0x3FC06EBA, 0x8214DB68 */
psn_pp1 = -3.25042107247001499370e-01, /* 0xBFD4CD7D, 0x691CB913 */
psn_pp2 = -2.84817495755985104766e-02, /* 0xBF9D2A51, 0xDBD7194F */
psn_pp3 = -5.77027029648944159157e-03, /* 0xBF77A291, 0x236668E4 */
psn_pp4 = -2.37630166566501626084e-05, /* 0xBEF8EAD6, 0x120016AC */
psn_qq1 = 3.97917223959155352819e-01, /* 0x3FD97779, 0xCDDADC09 */
psn_qq2 = 6.50222499887672944485e-02, /* 0x3FB0A54C, 0x5536CEBA */
psn_qq3 = 5.08130628187576562776e-03, /* 0x3F74D022, 0xC4D36B0F */
psn_qq4 = 1.32494738004321644526e-04, /* 0x3F215DC9, 0x221C1A10 */
psn_qq5 = -3.96022827877536812320e-06, /* 0xBED09C43, 0x42A26120 */
/*
* Coefficients for approximation to erf in [0.84375,1.25]
*/
psn_pa0 = -2.36211856075265944077e-03, /* 0xBF6359B8, 0xBEF77538 */
psn_pa1 = 4.14856118683748331666e-01, /* 0x3FDA8D00, 0xAD92B34D */
psn_pa2 = -3.72207876035701323847e-01, /* 0xBFD7D240, 0xFBB8C3F1 */
psn_pa3 = 3.18346619901161753674e-01, /* 0x3FD45FCA, 0x805120E4 */
psn_pa4 = -1.10894694282396677476e-01, /* 0xBFBC6398, 0x3D3E28EC */
psn_pa5 = 3.54783043256182359371e-02, /* 0x3FA22A36, 0x599795EB */
psn_pa6 = -2.16637559486879084300e-03, /* 0xBF61BF38, 0x0A96073F */
psn_qa1 = 1.06420880400844228286e-01, /* 0x3FBB3E66, 0x18EEE323 */
psn_qa2 = 5.40397917702171048937e-01, /* 0x3FE14AF0, 0x92EB6F33 */
psn_qa3 = 7.18286544141962662868e-02, /* 0x3FB2635C, 0xD99FE9A7 */
psn_qa4 = 1.26171219808761642112e-01, /* 0x3FC02660, 0xE763351F */
psn_qa5 = 1.36370839120290507362e-02, /* 0x3F8BEDC2, 0x6B51DD1C */
psn_qa6 = 1.19844998467991074170e-02, /* 0x3F888B54, 0x5735151D */
/*
* Coefficients for approximation to erfc in [1.25,1/0.35]
*/
psn_ra0 = -9.86494403484714822705e-03, /* 0xBF843412, 0x600D6435 */
psn_ra1 = -6.93858572707181764372e-01, /* 0xBFE63416, 0xE4BA7360 */
psn_ra2 = -1.05586262253232909814e+01, /* 0xC0251E04, 0x41B0E726 */
psn_ra3 = -6.23753324503260060396e+01, /* 0xC04F300A, 0xE4CBA38D */
psn_ra4 = -1.62396669462573470355e+02, /* 0xC0644CB1, 0x84282266 */
psn_ra5 = -1.84605092906711035994e+02, /* 0xC067135C, 0xEBCCABB2 */
psn_ra6 = -8.12874355063065934246e+01, /* 0xC0545265, 0x57E4D2F2 */
psn_ra7 = -9.81432934416914548592e+00, /* 0xC023A0EF, 0xC69AC25C */
psn_sa1 = 1.96512716674392571292e+01, /* 0x4033A6B9, 0xBD707687 */
psn_sa2 = 1.37657754143519042600e+02, /* 0x4061350C, 0x526AE721 */
psn_sa3 = 4.34565877475229228821e+02, /* 0x407B290D, 0xD58A1A71 */
psn_sa4 = 6.45387271733267880336e+02, /* 0x40842B19, 0x21EC2868 */
psn_sa5 = 4.29008140027567833386e+02, /* 0x407AD021, 0x57700314 */
psn_sa6 = 1.08635005541779435134e+02, /* 0x405B28A3, 0xEE48AE2C */
psn_sa7 = 6.57024977031928170135e+00, /* 0x401A47EF, 0x8E484A93 */
psn_sa8 = -6.04244152148580987438e-02, /* 0xBFAEEFF2, 0xEE749A62 */
/*
* Coefficients for approximation to erfc in [1/.35,28]
*/
psn_rb0 = -9.86494292470009928597e-03, /* 0xBF843412, 0x39E86F4A */
psn_rb1 = -7.99283237680523006574e-01, /* 0xBFE993BA, 0x70C285DE */
psn_rb2 = -1.77579549177547519889e+01, /* 0xC031C209, 0x555F995A */
psn_rb3 = -1.60636384855821916062e+02, /* 0xC064145D, 0x43C5ED98 */
psn_rb4 = -6.37566443368389627722e+02, /* 0xC083EC88, 0x1375F228 */
psn_rb5 = -1.02509513161107724954e+03, /* 0xC0900461, 0x6A2E5992 */
psn_rb6 = -4.83519191608651397019e+02, /* 0xC07E384E, 0x9BDC383F */
psn_sb1 = 3.03380607434824582924e+01, /* 0x403E568B, 0x261D5190 */
psn_sb2 = 3.25792512996573918826e+02, /* 0x40745CAE, 0x221B9F0A */
psn_sb3 = 1.53672958608443695994e+03, /* 0x409802EB, 0x189D5118 */
psn_sb4 = 3.19985821950859553908e+03, /* 0x40A8FFB7, 0x688C246A */
psn_sb5 = 2.55305040643316442583e+03, /* 0x40A3F219, 0xCEDF3BE6 */
psn_sb6 = 4.74528541206955367215e+02, /* 0x407DA874, 0xE79FE763 */
psn_sb7 = -2.24409524465858183362e+01; /* 0xC03670E2, 0x42712D62 */


/////////////////////////////////////////////////////////////////////////
// OPERATOR
/////////////////////////////////////////////////////////////////////////

/************************************************************************
 Method Name: appendRow
 Description: 
	- append additional row into cv::Mat
 Input Arguments:
	- dstMat: the matrix to which the append row
	- row: appending row
 Return Values:
	- void
************************************************************************/
void psn::appendRow(cv::Mat &dstMat, cv::Mat row)
{
	// append with empty matrix
	if (dstMat.empty())
	{
		dstMat = row.clone();
		return;
	}

	// normal appending
	if (dstMat.cols == row.cols)
	{
		cv::vconcat(dstMat, row, dstMat);
		return;
	}

	// expand dstMat
	if (dstMat.cols < row.cols)
	{
		cv::Mat newMat(dstMat.rows + 1, row.cols, dstMat.type());
		newMat(cv::Rect(0, 0, dstMat.rows, dstMat.cols)) = dstMat.clone();
		newMat.row(dstMat.rows) = row.clone();
		dstMat.release();
		dstMat = newMat.clone();
	}

	// expand row
	cv::Mat newRow(1, dstMat.cols, row.type());
	newRow(cv::Rect(0, 0, 1, row.cols)) = row.clone();
	cv::vconcat(dstMat, row, newRow);
}

/************************************************************************
 Method Name: appendCol
 Description: 
	- append additional column into cv::Mat
 Input Arguments:
	- dstMat: the matrix to which the append column
	- col: appending column
 Return Values:
	- void
************************************************************************/
void psn::appendCol(cv::Mat &dstMat, cv::Mat col)
{
	// append with empty matrix
	if (dstMat.empty())
	{
		dstMat = col.clone();
		return;
	}

	// normal appending
	if (dstMat.rows == col.rows)
	{
		cv::hconcat(dstMat, col, dstMat);
		return;
	}

	// expand dstMat
	if (dstMat.rows < col.rows)
	{
		cv::Mat newMat(col.rows, dstMat.cols + 1, dstMat.type());
		newMat(cv::Rect(0, 0, dstMat.rows, dstMat.cols)) = dstMat.clone();
		newMat.col(dstMat.cols) = col.clone();
		dstMat.release();
		dstMat = newMat.clone();
	}

	// expand row
	cv::Mat newRow(dstMat.rows, 1, col.type());
	newRow(cv::Rect(0, 0, col.rows, 1)) = col.clone();
	cv::hconcat(dstMat, col, newRow);
}

/************************************************************************
 Method Name: nchoosek
 Description: 
	- generate combinations consist of k elements from integer set {0, ..., n-1}
 Input Arguments:
	- n: choose from {0, ..., n-1}
	- k: the number of elements for output combinations 
 Return Values:
	- result combinations
************************************************************************/
std::vector<std::vector<unsigned int>> psn::nchoosek(int n, int k)
{
	std::vector<std::vector<unsigned int>> outputCombinations;
	if (n < k || n <= 0 ) { return outputCombinations; }

	std::vector<bool> v(n);
	std::vector<unsigned int> curCombination;

	std::fill(v.begin() + k, v.end(), true);
	do
	{
		curCombination.clear();
		curCombination.reserve(k);
		for (int idx = 0; idx < n; ++idx)
		{ 
			if (!v[idx]) { curCombination.push_back(unsigned int(idx));	} 
		}
		outputCombinations.push_back(curCombination);
	}
	while (std::next_permutation(v.begin(), v.end()));
	return outputCombinations;
}

/************************************************************************
 Method Name: erf
 Description: 
	- error function
 Input Arguments:
	- x: any real value
 Return Values:
	- error
************************************************************************/
double psn::erf(double x)
{
	bool bFastErf = false;

	if (bFastErf)
	{
		// FAST VERSION
		// constants
		double a1 =  0.254829592;
		double a2 = -0.284496736;
		double a3 =  1.421413741;
		double a4 = -1.453152027;
		double a5 =  1.061405429;
		double p  =  0.3275911;

		// Save the sign of x
		int sign = 1;
		if (x < 0) { sign = -1; }
		x = fabs(x);

		// A&S formula 7.1.26
		double t = 1.0/(1.0 + p*x);
		double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

		return sign*y;
	}

	// ACCURATE VERSION
	int n0, hx, ix, i;
	double R, S, P, Q, s, y, z, r;
	n0 = ((*(int*)&psn_one)>>29)^1;
	hx = *(n0 + (int*)&x);
	ix = hx & 0x7fffffff;
	if (ix >= 0x7ff00000) 
	{ 
		/* erf(nan)=nan */
		i = ((unsigned)hx >> 31) << 1;
		return (double)(1 - i) + psn_one / x; /* erf(+-inf)=+-1 */
	}

	if (ix < 0x3feb0000) 
	{ 
		/* |x|<0.84375 */
		if (ix < 0x3e300000) 
		{ 
			/* |x|<2**-28 */
			if (ix < 0x00800000) { return 0.125 * (8.0 * x + psn_efx8 * x); } /*avoid underflow */
			return x + psn_efx * x;
		}
		z = x*x;
		r = psn_pp0 + z * (psn_pp1 + z * (psn_pp2 + z * (psn_pp3 + z * psn_pp4)));
		s = psn_one + z * (psn_qq1 + z * (psn_qq2 + z * (psn_qq3 + z * (psn_qq4 + z * psn_qq5))));
		y = r/s;
		return x + x*y;
	}

	if (ix < 0x3ff40000)
	{ 
		/* 0.84375 <= |x| < 1.25 */
		s = fabs(x) - psn_one;
		P = psn_pa0+s*(psn_pa1+s*(psn_pa2+s*(psn_pa3+s*(psn_pa4+s*(psn_pa5+s*psn_pa6)))));
		Q = psn_one+s*(psn_qa1+s*(psn_qa2+s*(psn_qa3+s*(psn_qa4+s*(psn_qa5+s*psn_qa6)))));
		if (hx>=0) 
		{
			return psn_erx + P / Q; 
		}
		else 
		{
			return -psn_erx - P/Q;
		}
	}

	if (ix >= 0x40180000) 
	{ 
		/* inf>|x|>=6 */
		if(hx >= 0)
		{
			return psn_one - psn_tiny; 
		}
		else
		{
			return psn_tiny - psn_one;
		}
	}

	x = fabs(x);
	s = psn_one / (x * x);

	if (ix < 0x4006DB6E)
	{ 
		/* |x| < 1/0.35 */
		R = psn_ra0 + s * (psn_ra1 + s * (psn_ra2 + s * (psn_ra3 + s * (psn_ra4 + s * (psn_ra5 + s * (psn_ra6 + s * psn_ra7))))));
		S = psn_one + s * (psn_sa1 + s * (psn_sa2 + s * (psn_sa3 + s * (psn_sa4 + s * (psn_sa5 + s * (psn_sa6 + s * (psn_sa7 + s * psn_sa8)))))));
	} 
	else 
	{ 
		/* |x| >= 1/0.35 */
		R = psn_rb0 + s * (psn_rb1 + s * (psn_rb2 + s * (psn_rb3 + s * (psn_rb4 + s * (psn_rb5 + s * psn_rb6)))));
		S = psn_one + s * (psn_sb1 + s * (psn_sb2 + s * (psn_sb3 + s * (psn_sb4 + s * (psn_sb5 + s * (psn_sb6 + s * psn_sb7))))));
	}
	z = x;
	*(1 - n0 + (int*)&z) = 0;
	r = exp(-z * z - 0.5625) * exp((z - x) * (z + x) + R / S);
	if (hx >= 0)
	{
		return psn_one - r / x; 
	}
	else
	{
		return r / x - psn_one;
	}
}

/************************************************************************
 Method Name: erfc
 Description: 
	- complement error function
 Input Arguments:
	- any real value
 Return Values:
	- complement of error
************************************************************************/
double psn::erfc(double x)
{
	bool bFastErfc = false;

	// FAST VERSION
	if (bFastErfc) { return 1 - psn::erf(x); }

	// ACCURATE VERSION
	int n0, hx, ix;
	double R, S, P, Q, s, y, z, r;
	n0 = ((*(int*)&psn_one) >> 29)^1;
	hx = *(n0 + (int*)&x);
	ix = hx & 0x7fffffff;

	/* erfc(nan)=nan */
	/* erfc(+-inf)=0,2 */
	if (ix >= 0x7ff00000) {	return (double)(((unsigned)hx >> 31) << 1) + psn_one / x; }

	if (ix < 0x3feb0000) 
	{
		/* |x|<0.84375 */
		if (ix < 0x3c700000) { return psn_one - x; } /* |x|<2**-56 */
		z = x * x;
		r = psn_pp0 + z * (psn_pp1 + z * (psn_pp2 + z * (psn_pp3 + z * psn_pp4)));
		s = psn_one + z * (psn_qq1 + z * (psn_qq2 + z * (psn_qq3 + z * (psn_qq4 + z * psn_qq5))));
		y = r / s;
		if (hx < 0x3fd00000) 
		{ 
			/* x<1/4 */
			return psn_one - (x + x * y);
		} 
		else 
		{
			r = x * y;
			r += (x - psn_half);
			return psn_half - r;
		}
	}

	if (ix < 0x3ff40000) 
	{ 
		/* 0.84375 <= |x| < 1.25 */
		s = fabs(x) - psn_one;
		P = psn_pa0 + s * (psn_pa1 + s * (psn_pa2 + s * (psn_pa3 + s * (psn_pa4 + s * (psn_pa5 + s * psn_pa6)))));
		Q = psn_one + s * (psn_qa1 + s * (psn_qa2 + s * (psn_qa3 + s * (psn_qa4 + s * (psn_qa5 + s * psn_qa6)))));
		if (hx >= 0) 
		{
			z = psn_one - psn_erx;
			return z - P/Q;
		} 
		else 
		{
			z = psn_erx + P / Q; 
			return psn_one + z;
		}
	}
	
	if (ix < 0x403c0000) 
	{ 
		/* |x|<28 */
		x = fabs(x);
		s = psn_one / (x * x);
		if (ix < 0x4006DB6D) 
		{ 
			/* |x| < 1/.35 ~ 2.857143*/
			R = psn_ra0 + s * (psn_ra1 + s * (psn_ra2 + s * (psn_ra3 + s * (psn_ra4 + s * (psn_ra5 + s * (psn_ra6 + s * psn_ra7))))));
			S = psn_one + s * (psn_sa1 + s * (psn_sa2 + s * (psn_sa3 + s * (psn_sa4 + s * (psn_sa5 + s * (psn_sa6 + s * (psn_sa7 + s * psn_sa8)))))));
		} 
		else 
		{ 
			/* |x| >= 1/.35 ~ 2.857143 */
			if (hx<0&&ix>=0x40180000) {	return psn_two - psn_tiny; } /* x < -6 */
			R = psn_rb0 + s * (psn_rb1 + s * (psn_rb2 + s * (psn_rb3 + s * (psn_rb4 + s * (psn_rb5 + s * psn_rb6)))));
			S = psn_one + s * (psn_sb1 + s * (psn_sb2 + s * (psn_sb3 + s * (psn_sb4 + s * (psn_sb5 + s *(psn_sb6 + s * psn_sb7))))));
		}
		z = x;
		*(1 - n0 + (int*)&z) = 0;
		r = exp(-z * z - 0.5625) * exp((z - x) * (z + x) + R / S);
		if (hx > 0)
		{
			return r / x; 
		}
		else 
		{
			return psn_two - r / x;
		}
	} 
	else 
	{
		if (hx > 0)
		{
			return psn_tiny * psn_tiny; 
		}
		else 
		{
			return psn_two - psn_tiny;
		}
	}
}

/************************************************************************
 Method Name: histogram
 Description: 
	- generate RGB color histogram with input patch
 Input Arguments:
	- singleChannelImage: input image patch
	- numBin: the number of bins
 Return Values:
	- color histrogram contained in cv::Mat
************************************************************************/
cv::Mat psn::histogram(cv::Mat singleChannelImage, int numBin)
{
	assert(1 == singleChannelImage.channels() && numBin > 0);
	cv::Mat vecHistogram = cv::Mat::zeros(numBin, 1, CV_64FC1);
	double binSize = 256.0 / numBin;
	
	uchar *pData = singleChannelImage.data;
	int curHistogramBinIdx = 0;
	for (int idx = 0; idx < singleChannelImage.rows * singleChannelImage.cols; idx++, pData++)
	{
		curHistogramBinIdx = (int)std::floor((double)(*pData) / binSize);
		vecHistogram.at<double>(curHistogramBinIdx, 0)++;
	}

	return vecHistogram;
}

/************************************************************************
 Method Name: IsLineSegmentIntersect
 Description: 
	- Check line-line intersetction
 Input Arguments:
	- line1: the first line
	- line2: the second line
 Return Values:
	- wheather the lines are intersect or not
************************************************************************/
bool psn::IsLineSegmentIntersect(PSN_Line &line1, PSN_Line &line2)
{
	// from: "Tricks of the Windows Game Programming Gurus"
	double s1x, s1y, s2x, s2y;
	s1x = line1.second.x - line1.first.x;
	s1y = line1.second.y - line1.first.y;
	s2x = line2.second.x - line2.first.x;
	s2y = line2.second.y - line2.first.y;

	double s, t;
	s = (-s1y * (line1.first.x - line2.first.x) + s1x * (line1.first.y - line2.first.y)) / (-s2x * s1y + s1x * s2y);
	t = ( s2x * (line1.first.y - line2.first.y) - s2y * (line1.first.x - line2.first.x)) / (-s2x * s1y + s1x * s2y);

	if (0 <= s && s <= 1 && 0 <= t && t <= 1) { return true; }
	return false;
}

/************************************************************************
 Method Name: Triangulation
 Description: 
	- find the point which has the smallest total distance to each line
 Input Arguments:
	- line1: the first line
	- line2: the second line
 Return Values:
	- the distance between lines
************************************************************************/
double psn::Triangulation(PSN_Line &line1, PSN_Line &line2, PSN_Point3D &midPoint3D)
{
	PSN_Point3D line1Direct = line1.first - line1.second;
	PSN_Point3D line2Direct = line2.first - line2.second;
	PSN_Point3D lineOffset = line2.second - line1.second;

	cv::Mat matA(2, 2, CV_32FC1);
	matA.at<float>(0, 0) = (float)line1Direct.dot(line1Direct);
	matA.at<float>(0, 1) = (float)line1Direct.dot(-line2Direct);
	matA.at<float>(1, 0) = (float)line2Direct.dot(line1Direct);
	matA.at<float>(1, 1) = (float)line2Direct.dot(-line2Direct);
	
	cv::Mat vecB(2, 1, CV_32FC1);
	vecB.at<float>(0, 0) = (float)line1Direct.dot(lineOffset);
	vecB.at<float>(1, 0) = (float)line2Direct.dot(lineOffset);

	cv::Mat vecT = matA.inv() * vecB;

	line1Direct *= (double)vecT.at<float>(0, 0);
	line2Direct *= (double)vecT.at<float>(1, 0);
	PSN_Point3D closePoint1 = line1.second + line1Direct;
	PSN_Point3D closePoint2 = line2.second + line2Direct;

	midPoint3D = (closePoint1 + closePoint2) / 2;

	return (closePoint1 - closePoint2).norm_L2();
}

/************************************************************************
 Method Name: GenerateColors
 Description: 
	- generate distinguishable color set
 Input Arguments:
	- numColor: the number of colors needed
 Return Values:
	- vector of RGB color coordinates
************************************************************************/
std::vector<cv::Scalar> psn::GenerateColors(unsigned int numColor)
{
	// refer: http://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically/
	//# use golden ratio
	//golden_ratio_conjugate = 0.618033988749895
	//h = rand # use random start value
	//gen_html {
	//  h += golden_ratio_conjugate
	//  h %= 1
	//  hsv_to_rgb(h, 0.5, 0.95)
	//}

	double golden_ratio_conjugate = 0.618033988749895;
	//double hVal = (double)std::rand()/(INT_MAX);
	double hVal = 0.0;
	std::vector<cv::Scalar> resultColors;
	resultColors.reserve(numColor);
	for (unsigned int colorIdx = 0; colorIdx < numColor; colorIdx++)
	{
		hVal += golden_ratio_conjugate;
		hVal = std::fmod(hVal, 1.0);
		resultColors.push_back(psn::hsv2rgb(hVal, 0.5, 0.95));
	}
	return resultColors;
}

/************************************************************************
 Method Name: hsv2rgb
 Description: 
	- HSV -> RGB
 Input Arguments:
	- h: hue
	- s: saturation
	- v: value
 Return Values:
	- RGB color coordinates
************************************************************************/
cv::Scalar psn::hsv2rgb(double h, double s, double v)
{
	// refer: http://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically/
	//# HSV values in [0..1[
	//# returns [r, g, b] values from 0 to 255
	//def hsv_to_rgb(h, s, v)
	//  h_i = (h*6).to_i
	//  f = h*6 - h_i
	//  p = v * (1 - s)
	//  q = v * (1 - f*s)
	//  t = v * (1 - (1 - f) * s)
	//  r, g, b = v, t, p if h_i==0
	//  r, g, b = q, v, p if h_i==1
	//  r, g, b = p, v, t if h_i==2
	//  r, g, b = p, q, v if h_i==3
	//  r, g, b = t, p, v if h_i==4
	//  r, g, b = v, p, q if h_i==5
	//  [(r*256).to_i, (g*256).to_i, (b*256).to_i]
	//end

	int h_i = (int)(h * 6);
	double f = h * 6 - (double)h_i;
	double p = v * (1 - s);
	double q = v * (1 - f * s);
	double t = v * (1 - (1 - f) * s);
	double r, g, b;
	switch (h_i)
	{
	case 0: r = v; g = t; b = p; break;
	case 1: r = q; g = v; b = p; break;
	case 2: r = p; g = v; b = t; break;
	case 3: r = p; g = q; b = v; break;
	case 4: r = t; g = p; b = v; break;
	case 5: r = v; g = p; b = q; break;
	default: 
		break;
	}
	
	return cv::Scalar((int)(r * 255), (int)(g * 255), (int)(b * 255));
}

/************************************************************************
 Method Name: getColorByID
 Description: 
	- get distinguishable color by its ID. ID can be larger than the size
	of distinguishable colors
 Input Arguments:
	- vecColors: distinguishable color space
	- nID: its ID
 Return Values:
	- RGB color coordinates
************************************************************************/
cv::Scalar psn::getColorByID(unsigned int nID, std::vector<cv::Scalar> *vecColors)
{
	if (NULL == vecColors) { return cv::Scalar(255, 255, 255); }
	unsigned int colorIdx = nID % vecColors->size();
	return (*vecColors)[colorIdx];
}

/************************************************************************
 Method Name: DrawBoxWithID
 Description: 
	- draw box and 
 Input Arguments:
	- imageFrame: frame for drawing
	- curRect: box for drawing
	- nID: box's ID
	- lineStyle: 0: thin / 1: thick
	- fontSize: 0: small / 1:big
	- vecColors: a distinguishable color space. when it is set by NULL, draw
	items with white color
 Return Values:
	- none
************************************************************************/
void psn::DrawBoxWithID(cv::Mat &imageFrame, PSN_Rect box, unsigned int nID, int lineStyle, int fontSize, std::vector<cv::Scalar> *vecColors)
{
	// get label length
	unsigned int labelLength = nID > 0 ? 0 : 1;
	unsigned int tempLabel = nID;
	while (tempLabel > 0)
	{
		tempLabel /= 10;
		labelLength++;
	}

	cv::Scalar curColor = psn::getColorByID(nID, vecColors);
	if (0 == fontSize)
	{
		cv::rectangle(imageFrame, box.cv(), curColor, 1);
		cv::rectangle(imageFrame, cv::Rect((int)box.x, (int)box.y - 10, 7 * labelLength, 14), curColor, CV_FILLED);
		cv::putText(imageFrame, std::to_string(nID), cv::Point((int)box.x, (int)box.y-1), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 0));
	}
	else
	{
		cv::rectangle(imageFrame, box.cv(), curColor, 1+lineStyle);	
		cv::putText(imageFrame, std::to_string(nID), cv::Point((int)box.x, (int)box.y+40), cv::FONT_HERSHEY_SIMPLEX, 0.5, curColor);
	}
}

/************************************************************************
 Method Name: Draw3DBoxWithID
 Description: 
	- draw 3D box for the target on the frame with its ID
 Input Arguments:
	- imageFrame: the frame for drawing
	- pointArray: pre-projected corners of 3D box
	- nID: box's ID
	- vecColors: a distinguishable color space. when it is set by NULL, draw
	items with white color
 Return Values:
	- none
************************************************************************/
void psn::Draw3DBoxWithID(cv::Mat &imageFrame, std::vector<PSN_Point2D> &pointArray, unsigned int nID, std::vector<cv::Scalar> *vecColors)
{
	// point 1 to 4 : roof box (clock-wise)
	// point 5 to 8 : bottom box (clock-wise, point 1 must be at a vertically upside of point 5)

	if (pointArray.size() < 8) { return; }
	cv::Scalar curColor = psn::getColorByID(nID, vecColors);

	// draw roof and bottom box
	std::vector<cv::Point> points;
	unsigned int drawOrder[10] = {0, 1, 2, 3, 0, 4, 5, 6, 7, 4};
	for (unsigned int pointIdx = 0; pointIdx < 10; pointIdx++)
	{
		points.push_back(pointArray[drawOrder[pointIdx]].cv());
	}

	const cv::Point *pts = (const cv::Point*)cv::Mat(points).data;
	int npts = cv::Mat(points).rows;
	cv::polylines(imageFrame, &pts, &npts, 1, false, curColor);

	// draw rest part of rectangle
	for (unsigned int pointIdx = 1; pointIdx < 4; pointIdx++)
	{
		cv::line(imageFrame, pointArray[pointIdx].cv(), pointArray[pointIdx+4].cv(), curColor);
	}

	// get label length
	unsigned int labelLength = nID > 0 ? 0 : 1;
	unsigned int tempLabel = nID;
	while (tempLabel > 0)
	{
		tempLabel /= 10;
		labelLength++;
	}

	// draw label
	cv::rectangle(imageFrame, cv::Rect((int)pointArray[0].x, (int)pointArray[0].y - 10, 7 * labelLength, 14), curColor, CV_FILLED);
	cv::putText(imageFrame, std::to_string(nID), cv::Point((int)pointArray[0].x, (int)pointArray[0].y-1), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255));

}

/************************************************************************
 Method Name: DrawTriangleWithID
 Description: 
	- For topview, draw triangle indicates the location of target
 Input Arguments:
	- imageFrame: the frame for drawing
	- point: location of the center of triangle
	- nID: its ID
	- vecColors: a distinguishable color space. when it is set by NULL, draw
	items with white color
 Return Values:
	- none
************************************************************************/
void psn::DrawTriangleWithID(cv::Mat &imageFrame, PSN_Point2D &point, unsigned int nID, std::vector<cv::Scalar> *vecColors)
{
	double size = 5;
	cv::Scalar curColor = psn::getColorByID(nID, vecColors);
	// draw five points
	double cosd30xSize = size * 0.866025403784439;
	double sind30xSize = size * 0.5;
	std::vector<cv::Point> points;
	PSN_Point2D topPoint(point.x, point.y - size);
	PSN_Point2D leftPoint(point.x - cosd30xSize, point.y + sind30xSize);
	PSN_Point2D rightPoint(point.x + cosd30xSize, point.y + sind30xSize);
	points.push_back(topPoint.cv());
	points.push_back(leftPoint.cv());
	points.push_back(rightPoint.cv());
	points.push_back(topPoint.cv());
	points.push_back(leftPoint.cv()); // for handling thick line

	const cv::Point *pts = (const cv::Point*)cv::Mat(points).data;
	int npts = cv::Mat(points).rows;
	cv::polylines(imageFrame, &pts, &npts, 1, false, curColor);
	cv::putText(imageFrame, std::to_string(nID), cv::Point((int)point.x, (int)point.y-2), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255));
}

/************************************************************************
 Method Name: DrawLine
 Description: 
	- draw polylines on the frame
 Input Arguments:
	- imageFrame: the frame for drawing
	- pointArray: vetices of line
	- nID: its ID (for color)
	- lineThickness: 0: thin / 1: thick
	- vecColors: a distinguishable color space. when it is set by NULL, draw
	items with white color
 Return Values:
	- none
************************************************************************/
void psn::DrawLine(cv::Mat &imageFrame, std::vector<PSN_Point2D> &pointArray, unsigned int nID, int lineThickness, std::vector<cv::Scalar> *vecColors)
{
	cv::Scalar curColor = psn::getColorByID(nID, vecColors);
	for (int pointIdx = 0; pointIdx < (int)pointArray.size() - 1; pointIdx++)
	{
		cv::line(imageFrame, pointArray[pointIdx].cv(), pointArray[pointIdx+1].cv(), curColor, 1+lineThickness);
	}
}

/************************************************************************
 Method Name: MakeMatTile
 Description: 
	- Make one result image for multi-camera input for visualization
 Input Arguments:
	- imageArray: 
	- numRows: number of rows in tile image
	- numCols: number of colums in tile image
 Return Values:
	- cv::Mat: result image
************************************************************************/
cv::Mat psn::MakeMatTile(std::vector<cv::Mat> *imageArray, unsigned int numRows, unsigned int numCols)
{
	// column first arrangement
	unsigned int numImage = (unsigned int)imageArray->size();	
	unsigned int acturalNumCols = numImage < numCols ? numImage : numCols;
	unsigned int acturalNumRows = (0 == numImage % numCols) ? numImage / numCols : numImage / numCols + 1;

	// find maximum size image
	cv::Size maxSize = (*imageArray)[0].size();
	for (unsigned int imageIdx = 0; imageIdx < numImage; imageIdx++)
	{
		if((*imageArray)[imageIdx].rows > maxSize.height){ maxSize.height = (*imageArray)[imageIdx].rows; }
		if((*imageArray)[imageIdx].cols > maxSize.width){ maxSize.width = (*imageArray)[imageIdx].cols; }
	}

	// make augmenting matrix
	std::vector<cv::Mat> augMats;
	cv::Mat baseMat;
	baseMat = 1 == (*imageArray)[0].channels()? cv::Mat::zeros(maxSize, CV_8UC1): cv::Mat::zeros(maxSize, CV_8UC3); 	
	unsigned int numAugMats = acturalNumRows * acturalNumCols;
	for (unsigned int imageIdx = 0; imageIdx < numAugMats; imageIdx++)
	{
		cv::Mat augMat = baseMat.clone();
		if (imageIdx < numImage)
		{
			(*imageArray)[imageIdx].copyTo(augMat(cv::Rect(0, 0, (*imageArray)[imageIdx].cols, (*imageArray)[imageIdx].rows)));
		}
		augMats.push_back(augMat);
	}

	// matrix concatenation
	cv::Mat hConcatMat;
	cv::Mat resultMat;
	for (unsigned int rowIdx = 0; rowIdx < acturalNumRows; rowIdx++)
	{
		unsigned int startIdx = rowIdx * acturalNumCols;		
		for (unsigned int colIdx = 0; colIdx < acturalNumCols; colIdx++)
		{
			if (0 == colIdx)
			{
				hConcatMat = augMats[startIdx].clone();
				continue;
			}
			cv::hconcat(hConcatMat, augMats[startIdx + colIdx], hConcatMat);
		}
		if (0 == rowIdx)
		{
			resultMat = hConcatMat.clone();
			continue;
		}
		cv::vconcat(resultMat, hConcatMat, resultMat);
	}

	return resultMat;
}

/************************************************************************
 Method Name: GetLocationOnTopView_PETS2009
 Description: 
	- Find the image coordinates correspond to the 3D world coordinates in
	PETS2009
 Input Arguments:
	- location: world coordinates
	- bZoom: true: coordinates for zoomed top view / false: default
 Return Values:
	- image coorindates for topview
************************************************************************/
PSN_Point2D psn::GetLocationOnTopView_PETS2009(PSN_Point3D location, bool bZoom)
{
	// more detail, refer MATLAB code, LocationOnTopView
	cv::Mat AffineMat(2, 3, CV_64FC1);
	AffineMat.at<double>(0, 0) = 0.003717382189026;
	AffineMat.at<double>(0, 1) = -0.011189305998417;
	AffineMat.at<double>(0, 2) = 322.0;
	AffineMat.at<double>(1, 0) = -0.011189305998417;
	AffineMat.at<double>(1, 1) = -0.003717382189026;
	AffineMat.at<double>(1, 2) = 278;

	cv::Mat homogeniousPoint(3, 1, CV_64FC1);
	homogeniousPoint.at<double>(0, 0) = location.x;
	homogeniousPoint.at<double>(1, 0) = location.y;
	homogeniousPoint.at<double>(2, 0) = 1.0;

	cv::Mat resultVec = AffineMat * homogeniousPoint;
	if (bZoom)
	{
		//// zoom
		//resultVec.at<double>(0, 0) = 1.5 * (resultVec.at<double>(0, 0) - 150.0);
		//resultVec.at<double>(1, 0) = 1.5 * (resultVec.at<double>(1, 0) - 150.0);

		// zoom2
		resultVec.at<double>(0, 0) = 1.5 * (1.5 * (resultVec.at<double>(0, 0) - 150.0) - 100.0);
		resultVec.at<double>(1, 0) = 1.5 * (1.5 * (resultVec.at<double>(1, 0) - 150.0) - 100.0);
	}

	return PSN_Point2D(resultVec.at<double>(0, 0), resultVec.at<double>(1, 0));
}

/************************************************************************
 Method Name: IsDirectoryExists
 Description: 
	- Check wheter the directory exists or not
 Input Arguments:
	- dirName: directory path
 Return Values:
	- true: exists / false: non-exists
************************************************************************/
bool psn::CreateDirectoryForWindows(const std::string &dirName)
{
	std::wstring wideStrDirName = L"";
	wideStrDirName.assign(dirName.begin(), dirName.end());
	if (CreateDirectory(wideStrDirName.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError()) { return true; }
	return false;
}

/************************************************************************
 Method Name: printLog
 Description: 
	- print out log file
 Input Arguments:
	- filename: file path
	- strLog: log string
 Return Values:
	- none
************************************************************************/
void psn::printLog(const char *filename, std::string strLog)
{
	try
	{
		FILE *fp;
		fopen_s(&fp, filename, "a");
		fprintf(fp, strLog.c_str());
		fclose(fp);
	}
	catch (DWORD dwError)
	{
		printf("[ERROR] cannot open logging file! error code %d\n", dwError);
		return;
	}
}

/************************************************************************
 Method Name: MakeTrackIDList
 Description: 
	- 
 Input Arguments:
	- 
	- 
 Return Values:
	- none
************************************************************************/
std::string psn::MakeTrackIDList(PSN_TrackSet *tracks)
{
	std::string strResult("{");
	for (PSN_TrackSet::iterator trackIter = tracks->begin();
		trackIter != tracks->end();
		trackIter++)
	{
		strResult = strResult + std::to_string((*trackIter)->id);
		if (trackIter < tracks->end() - 1) { strResult += ","; }
	}
	strResult += "}";

	return strResult;
}

///************************************************************************
// Method Name: GrabFrame
// Description: 
//	- frame grabbing
// Input Arguments:
//	- strDatasetPath: path of the dataset
//	- camID: camera ID
//	- frameIdx: frame index
// Return Values:
//	- grabbed frame in cv::Mat container
//************************************************************************/
//cv::Mat psn::GrabFrame(std::string strDatasetPath, unsigned int camID, unsigned int frameIdx)
//{
//	char inputFilePath[300];
//	if (PSN_INPUT_TYPE)	
//	{
//		sprintf_s(inputFilePath, sizeof(inputFilePath), "%s\\View_%03d\\frame_%04d.jpg", strDatasetPath.c_str(), camID, frameIdx);						
//	} 
//	else 
//	{
//		sprintf_s(inputFilePath, sizeof(inputFilePath), "%s\\%d_%d.jpg", strDatasetPath.c_str(), camID, frameIdx);						
//	}	
//	return cv::imread(inputFilePath, cv::IMREAD_COLOR);
//}

/************************************************************************
 Method Name: ReadDetectionResultWithTxt
 Description: 
	- 
 Input Arguments:
	- 
 Return Values:
	- Track3D*: 
************************************************************************/
std::vector<stDetection> psn::ReadDetectionResultWithTxt(std::string strDatasetPath, unsigned int camIdx, unsigned int frameIdx) {
	std::vector<stDetection> vec_result;
	char textfile_path[128];
	int num_detection = 0;
	float x, y, w, h, temp;

	FILE *fid;
	try {
		switch (PSN_DETECTION_TYPE)	{
		case 0:	// Head
			sprintf_s(textfile_path, sizeof(textfile_path), "%s\\detectionResult\\cam%d\\%04d.txt", strDatasetPath.c_str(), camIdx, frameIdx);
			fopen_s(&fid, textfile_path, "r");
			if (NULL == fid)	{ return vec_result; }
			// read # of detections
			fscanf_s(fid, "%d\n", &num_detection);
			vec_result.reserve(num_detection);
			// read box infos
			for (int detect_idx = 0; detect_idx < num_detection; detect_idx++) {
				fscanf_s(fid, "%f,%f,%f,%f,%f,%f\n", &temp, &temp, &w, &h, &x, &y);
				stDetection cur_detection;
				cur_detection.box = PSN_Rect((double)x, (double)y, (double)w, (double)h);
				vec_result.push_back(cur_detection);
			}
			fclose(fid);
			break;
		case 1:	// Full-body
			// file open
			switch (PSN_INPUT_TYPE) {
			case 0: // ETRI testbed
				sprintf_s(textfile_path, sizeof(textfile_path), "%s\\detectionResult\\cam%d\\%04d.txt", strDatasetPath.c_str(), camIdx, frameIdx);
				break;
			case 1: // PETS.S2.L1					
				sprintf_s(textfile_path, sizeof(textfile_path), "%s\\View_%03d\\detectionResult\\frame_%04d.txt", strDatasetPath.c_str(), camIdx, frameIdx);		
				break;
			default:
				break;
			}
			fopen_s(&fid, textfile_path, "r");
			if (NULL == fid) { return vec_result; }

			switch (PSN_INPUT_TYPE) {
			case 0: // ETRI testbed
				// read # of detections
				fscanf_s(fid, "%d\n", &num_detection);
				vec_result.reserve(num_detection);

				// read box infos
				for (int detect_idx = 0; detect_idx < num_detection; detect_idx++) {
					fscanf_s(fid, "%f %f %f %f %f %f\n", &temp, &temp, &w, &h, &x, &y);
					stDetection cur_detection;
					cur_detection.box = PSN_Rect((double)x, (double)y, (double)w, (double)h);	
					//curDetection.vecPartBoxes.reserve(8);
					vec_result.push_back(cur_detection);
				}
				break;
			case 1: // PETS.S2.L1
				// read # of detections
				fscanf_s(fid, "numBoxes:%d\n", &num_detection);
				vec_result.reserve(num_detection);

				// read box infos
				for (int detect_idx = 0; detect_idx < num_detection; detect_idx++) {
					fscanf_s(fid, "{\n\tROOT:{%f,%f,%f,%f}\n", &x, &y, &w, &h);
					stDetection cur_detection;
					cur_detection.box = PSN_Rect((double)x, (double)y, (double)w, (double)h);

					// read part info
					cur_detection.vecPartBoxes.reserve(8);
					for (unsigned int partIdx = 0; partIdx < NUM_DETECTION_PART; partIdx++)	{
						char strPartName[20];
						sprintf_s(strPartName, "\t%s:", DETCTION_PART_NAME[partIdx].c_str());
						fscanf_s(fid, strPartName);
						fscanf_s(fid, "{%f,%f,%f,%f}\n", &x, &y, &w, &h);
						PSN_Rect partBox((double)x, (double)y, (double)w, (double)h);
						cur_detection.vecPartBoxes.push_back(partBox);
					}
					fscanf_s(fid, "}\n");
					vec_result.push_back(cur_detection);
				}
				break;
			default:
				break;
			}
			fclose(fid);
			break;
		default:
			break;
		}
	} catch(DWORD dwError) {
		printf("[ERROR] file open error with detection result reading: %d\n", dwError);
	}
	return vec_result;
}

/************************************************************************
 Method Name: Read2DTrackResultWithTxt
 Description: 
	- 
 Input Arguments:
	- 
 Return Values:
	- 
************************************************************************/
std::vector<stTrack2DResult> psn::Read2DTrackResultWithTxt(std::string strDatasetPath, unsigned int frameIdx)
{
	std::vector<stTrack2DResult> resultSet;
	unsigned int inCamIdx = 0, inFrameIdx = 0;
	unsigned int numModel = 0;
	char textFilePath[128];

	for (unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		stTrack2DResult curResult;
		curResult.camID = camIdx;
		curResult.frameIdx = frameIdx;
		
		FILE *fid;

		sprintf_s(textFilePath, sizeof(textFilePath), "%s\\trackletInput\\trackResult\\track2D_result_cam%d_frame%04d.txt", strDatasetPath.c_str(), CAM_ID[camIdx], frameIdx);
		fopen_s(&fid, textFilePath, "r");
		fscanf_s(fid, "camIdx:%d\nframeIdx:%d\nnumModel:%d\n", &inCamIdx, &inFrameIdx, &numModel);
		for(unsigned int modelIdx = 0; modelIdx < numModel; modelIdx++)
		{
			stObject2DInfo curObject;
			int x, y, w, h;
		
			fscanf_s(fid, "{id:%d,box:{%d,%d,%d,%d}}\n", &curObject.id, &x, &y, &w, &h);
			curObject.id--;
			curObject.box.x = (double)x;
			curObject.box.y = (double)y;
			curObject.box.w = (double)w;
			curObject.box.h = (double)h;

			curResult.object2DInfos.push_back(curObject);
		}

		resultSet.push_back(curResult);
		fclose(fid);
	}

	return resultSet;
}

/************************************************************************
 Method Name: Read2DTrackResultWithTxt
 Description: 
	- 
 Input Arguments:
	- 
 Return Values:
	- 
************************************************************************/
stTrack2DResult psn::Read2DTrackResultWithTxt(std::string strDataPath, unsigned int camID, unsigned int frameIdx)
{
	char strFilename[128];
	sprintf_s(strFilename, "track2D_result_cam%d_frame%04d.txt", (int)camID, (int)frameIdx);
	std::string strFilePathAndName = strDataPath + std::string(strFilename);

	stTrack2DResult result;
	result.camID = camID;
	result.frameIdx = frameIdx;

	int readInt1 = 0, readInt2 = 0;
	float readFloat = 0.0f;
	float x = 0.0f, y = 0.0f, w = 0.0f, h = 0.0f;

	FILE *fp;
	try
	{
		fopen_s(&fp, strFilePathAndName.c_str(), "r");

		// frame infos
		fscanf_s(fp, "camIdx:%d\nframeIdx:%d\n", &readInt1, &readInt2);

		// object infos
		int numObj = 0;
		fscanf_s(fp, "numObjectInfos:%d{\n", &numObj);		
		result.object2DInfos.reserve(numObj);

		for (size_t objIdx = 0; objIdx < numObj; objIdx++)
		{
			stObject2DInfo curObject;			

			fscanf_s(fp, "\t{\n");
			////////////////
			fscanf_s(fp, "\t\tid:%d\n", &readInt1);						curObject.id = (unsigned int)readInt1;
			fscanf_s(fp, "\t\tbox:(%f,%f,%f,%f)\n", &x, &y, &w, &h);	curObject.box = PSN_Rect((double)x, (double)y, (double)w, (double)h);
			fscanf_s(fp, "\t\thead:(%f,%f,%f,%f)\n", &x, &y, &w, &h);	curObject.head = PSN_Rect((double)x, (double)y, (double)w, (double)h);			
			fscanf_s(fp, "\t\tscore:%f\n", &readFloat);					curObject.score = (double)readFloat;

			fscanf_s(fp, "\t\tfeaturePointsPrev:%d,{", &readInt1);		curObject.featurePointsPrev.reserve(readInt1);
			for (size_t fIdx = 0; fIdx < readInt1; fIdx++)
			{
				fscanf_s(fp, "(%f,%f)", &x, &y);
				if (readInt1 > fIdx + 1) { fscanf_s(fp, ","); }
				curObject.featurePointsPrev.push_back(cv::Point2f(x, y));
			}
			fscanf_s(fp, "}\n");
			fscanf_s(fp, "\t\tfeaturePointsCurr:%d,{", &readInt1);		curObject.featurePointsCurr.reserve(readInt1);
			for (size_t fIdx = 0; fIdx < readInt1; fIdx++)
			{				
				fscanf_s(fp, "(%f,%f)", &x, &y);
				if (readInt1 > fIdx + 1) { fscanf_s(fp, ","); }
				curObject.featurePointsCurr.push_back(cv::Point2f(x, y));
			}
			fscanf_s(fp, "}\n");
			////////////////
			fscanf_s(fp, "\t}\n");

			result.object2DInfos.push_back(curObject);
		}
		fscanf_s(fp, "}\n");

		// detection rects	
		fscanf_s(fp, "detectionRects:%d,{", &readInt1);
		result.vecDetectionRects.reserve(readInt1);
		for(size_t rectIdx = 0; rectIdx < readInt1; rectIdx++)
		{
			fscanf_s(fp, "(%f,%f,%f,%f)", &x, &y, &w, &h);
			result.vecDetectionRects.push_back(PSN_Rect((double)x, (double)y, (double)w, (double)h));
			if (readInt1 > rectIdx + 1) { fscanf_s(fp, ","); }
		}
		fscanf_s(fp, "}\n");

		// tracker rects
		fscanf_s(fp, "trackerRects:%d,{", &readInt1);
		result.vecTrackerRects.reserve(readInt1);
		for(size_t rectIdx = 0; rectIdx < readInt1; rectIdx++)
		{
			fscanf_s(fp, "(%f,%f,%f,%f)", &x, &y, &w, &h);
			result.vecTrackerRects.push_back(PSN_Rect((double)x, (double)y, (double)w, (double)h));
			if (readInt1 > rectIdx + 1) { fscanf_s(fp, ","); }
		}
		fscanf_s(fp, "}\n");

		fclose(fp);
	}
	catch(DWORD dwError)
	{
		printf("[ERROR](FilePrintResult) cannot open file! error code %d\n", dwError);
		return result;
	}

	return result;
}

//()()
//('')HAANJU.YOO

