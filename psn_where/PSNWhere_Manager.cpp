#include "PSNWhere_Manager.h"

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
void psn::appendRow(cv::Mat &dstMat, cv::Mat &row)
{
	// append with empty matrix
	if(dstMat.empty())
	{
		dstMat = row.clone();
		return;
	}

	// normal appending
	if(dstMat.cols == row.cols)
	{
		cv::vconcat(dstMat, row, dstMat);
		return;
	}

	// expand dstMat
	if(dstMat.cols < row.cols)
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

void psn::appendCol(cv::Mat &dstMat, cv::Mat &col)
{
	// append with empty matrix
	if(dstMat.empty())
	{
		dstMat = col.clone();
		return;
	}

	// normal appending
	if(dstMat.rows == col.rows)
	{
		cv::hconcat(dstMat, col, dstMat);
		return;
	}

	// expand dstMat
	if(dstMat.rows < col.rows)
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

void psn::nchoosek(int n, int k, std::deque<std::vector<unsigned int>> &outputCombinations)
{
	outputCombinations.clear();
	if(n < k || n <= 0 ){ return; }

	// generate combinations consist of k elements from integer set {0, ..., n-1}
	std::vector<bool> v(n);
	std::vector<unsigned int> curCombination;

	std::fill(v.begin() + k, v.end(), true);	

	do
	{
		curCombination.clear();
		curCombination.reserve(k);
		for(int idx = 0; idx < n; ++idx)
		{
			if(!v[idx])
			{
				curCombination.push_back(unsigned int(idx));
			}
		}
		outputCombinations.push_back(curCombination);		
	}
	while(std::next_permutation(v.begin(), v.end()));
}

double psn::erf(double x)
{
	bool bFastErf = false;

	if(bFastErf)
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
		if (x < 0)
			sign = -1;
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
	if(ix >= 0x7ff00000) 
	{ 
		/* erf(nan)=nan */
		i = ((unsigned)hx >> 31) << 1;
		return (double)(1 - i) + psn_one / x; /* erf(+-inf)=+-1 */
	}

	if(ix < 0x3feb0000) 
	{ 
		/* |x|<0.84375 */
		if(ix < 0x3e300000) 
		{ 
			/* |x|<2**-28 */
			if (ix < 0x00800000)
			{
				return 0.125 * (8.0 * x + psn_efx8 * x); /*avoid underflow */
			}
			return x + psn_efx * x;
		}
		z = x*x;
		r = psn_pp0 + z * (psn_pp1 + z * (psn_pp2 + z * (psn_pp3 + z * psn_pp4)));
		s = psn_one + z * (psn_qq1 + z * (psn_qq2 + z * (psn_qq3 + z * (psn_qq4 + z * psn_qq5))));
		y = r/s;
		return x + x*y;
	}

	if(ix < 0x3ff40000)
	{ 
		/* 0.84375 <= |x| < 1.25 */
		s = fabs(x) - psn_one;
		P = psn_pa0+s*(psn_pa1+s*(psn_pa2+s*(psn_pa3+s*(psn_pa4+s*(psn_pa5+s*psn_pa6)))));
		Q = psn_one+s*(psn_qa1+s*(psn_qa2+s*(psn_qa3+s*(psn_qa4+s*(psn_qa5+s*psn_qa6)))));
		if(hx>=0) 
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

	if(ix < 0x4006DB6E)
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
	if(hx >= 0)
	{
		return psn_one - r / x; 
	}
	else
	{
		return r / x - psn_one;
	}
}

double psn::erfc(double x)
{
	bool bFastErfc = false;

	if(bFastErfc)
	{
		// FAST VERSION
		return 1 - psn::erf(x);
	}

	// ACCURATE VERSION
	int n0, hx, ix;
	double R, S, P, Q, s, y, z, r;
	n0 = ((*(int*)&psn_one) >> 29)^1;
	hx = *(n0 + (int*)&x);
	ix = hx & 0x7fffffff;
	if(ix >= 0x7ff00000) 
	{ 
		/* erfc(nan)=nan */
		/* erfc(+-inf)=0,2 */
		return (double)(((unsigned)hx >> 31) << 1) + psn_one / x;
	}

	if(ix < 0x3feb0000) 
	{ /* |x|<0.84375 */
		if(ix < 0x3c700000)
		{
			/* |x|<2**-56 */
			return psn_one - x;
		}
		z = x * x;
		r = psn_pp0 + z * (psn_pp1 + z * (psn_pp2 + z * (psn_pp3 + z * psn_pp4)));
		s = psn_one + z * (psn_qq1 + z * (psn_qq2 + z * (psn_qq3 + z * (psn_qq4 + z * psn_qq5))));
		y = r / s;
		if(hx < 0x3fd00000) 
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

	if(ix < 0x3ff40000) 
	{ 
		/* 0.84375 <= |x| < 1.25 */
		s = fabs(x) - psn_one;
		P = psn_pa0 + s * (psn_pa1 + s * (psn_pa2 + s * (psn_pa3 + s * (psn_pa4 + s * (psn_pa5 + s * psn_pa6)))));
		Q = psn_one + s * (psn_qa1 + s * (psn_qa2 + s * (psn_qa3 + s * (psn_qa4 + s * (psn_qa5 + s * psn_qa6)))));
		if(hx >= 0) 
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
		if(ix < 0x4006DB6D) 
		{ 
			/* |x| < 1/.35 ~ 2.857143*/
			R = psn_ra0 + s * (psn_ra1 + s * (psn_ra2 + s * (psn_ra3 + s * (psn_ra4 + s * (psn_ra5 + s * (psn_ra6 + s * psn_ra7))))));
			S = psn_one + s * (psn_sa1 + s * (psn_sa2 + s * (psn_sa3 + s * (psn_sa4 + s * (psn_sa5 + s * (psn_sa6 + s * (psn_sa7 + s * psn_sa8)))))));
		} 
		else 
		{ 
			/* |x| >= 1/.35 ~ 2.857143 */
			if(hx<0&&ix>=0x40180000)
			{
				return psn_two - psn_tiny;/* x < -6 */
			}
			R = psn_rb0 + s * (psn_rb1 + s * (psn_rb2 + s * (psn_rb3 + s * (psn_rb4 + s * (psn_rb5 + s * psn_rb6)))));
			S = psn_one + s * (psn_sb1 + s * (psn_sb2 + s * (psn_sb3 + s * (psn_sb4 + s * (psn_sb5 + s *(psn_sb6 + s * psn_sb7))))));
		}
		z = x;
		*(1 - n0 + (int*)&z) = 0;
		r = exp(-z * z - 0.5625) * exp((z - x) * (z + x) + R / S);
		if(hx > 0)
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
		if(hx > 0)
		{
			return psn_tiny * psn_tiny; 
		}
		else 
		{
			return psn_two - psn_tiny;
		}
	}
}

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
	for(unsigned int colorIdx = 0; colorIdx < numColor; colorIdx++)
	{
		hVal += golden_ratio_conjugate;
		hVal = std::fmod(hVal, 1.0);
		resultColors.push_back(psn::hsv2rgb(hVal, 0.5, 0.95));
	}
	return resultColors;
}

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
	switch(h_i)
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

cv::Scalar psn::getColorByID(std::vector<cv::Scalar> &vecColors, unsigned int nID)
{
	unsigned int colorIdx = nID % vecColors.size();
	return vecColors[colorIdx];
}

void psn::DrawBoxWithID(cv::Mat &imageFrame, PSN_Rect curRect, unsigned int nID, std::vector<cv::Scalar> &vecColors)
{
	// get label length
	unsigned int labelLength = nID > 0 ? 0 : 1;
	unsigned int tempLabel = nID;
	while(tempLabel > 0)
	{
		tempLabel /= 10;
		labelLength++;
	}

	// get color
	cv::Scalar curColor = psn::getColorByID(vecColors, nID);

	cv::rectangle(imageFrame, curRect.cv(), curColor, 1);
	cv::rectangle(imageFrame, cv::Rect((int)curRect.x, (int)curRect.y - 10, 7 * labelLength, 14), curColor, CV_FILLED);
	//cv::putText(imageFrame, std::to_string(nID), cv::Point((int)curRect.x, (int)curRect.y-1), cv::FONT_HERSHEY_TRIPLEX, 0.5, psn::getColorByID(vecColors, nID));
	cv::putText(imageFrame, std::to_string(nID), cv::Point((int)curRect.x, (int)curRect.y-1), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255));
}

void psn::DrawBoxWithLargeID(cv::Mat &imageFrame, PSN_Rect curRect, unsigned int nID, std::vector<cv::Scalar> &vecColors, bool bDashed)
{
	// get label length
	unsigned int labelLength = nID > 0 ? 0 : 1;
	unsigned int tempLabel = nID;
	while(tempLabel > 0)
	{
		tempLabel /= 10;
		labelLength++;
	}

	// get color
	cv::Scalar curColor = psn::getColorByID(vecColors, nID);

	if(bDashed)
	{
		cv::rectangle(imageFrame, curRect.cv(), curColor, 1);
	}
	else
	{
		cv::rectangle(imageFrame, curRect.cv(), curColor, 2);
	}
	//cv::rectangle(imageFrame, cv::Rect((int)curRect.x, (int)curRect.y - 10, 7 * labelLength, 14), curColor, CV_FILLED);
	//cv::putText(imageFrame, std::to_string(nID), cv::Point((int)curRect.x, (int)curRect.y-1), cv::FONT_HERSHEY_TRIPLEX, 0.5, psn::getColorByID(vecColors, nID));
	cv::putText(imageFrame, std::to_string(nID), cv::Point((int)curRect.x, (int)curRect.y+40), cv::FONT_HERSHEY_SIMPLEX, 1.0, curColor);
}

void psn::Draw3DBoxWithID(cv::Mat &imageFrame, std::vector<PSN_Point2D> &pointArray, unsigned int nID, std::vector<cv::Scalar> &vecColors)
{
	// point 1 to 4 : roof box (clock-wise)
	// point 5 to 8 : bottom box (clock-wise, point 1 must be at a vertically upside of point 5)

	if(pointArray.size() < 8){ return; }

	// get color
	cv::Scalar curColor = psn::getColorByID(vecColors, nID);

	// draw roof and bottom box
	std::vector<cv::Point> points;
	unsigned int drawOrder[10] = {0, 1, 2, 3, 0, 4, 5, 6, 7, 4};
	for(unsigned int pointIdx = 0; pointIdx < 10; pointIdx++)
	{
		points.push_back(pointArray[drawOrder[pointIdx]].cv());
	}

	const cv::Point *pts = (const cv::Point*)cv::Mat(points).data;
	int npts = cv::Mat(points).rows;
	cv::polylines(imageFrame, &pts, &npts, 1, false, curColor);

	// draw rest part of rectangle
	for(unsigned int pointIdx = 1; pointIdx < 4; pointIdx++)
	{
		cv::line(imageFrame, pointArray[pointIdx].cv(), pointArray[pointIdx+4].cv(), curColor);
	}

	// get label length
	unsigned int labelLength = nID > 0 ? 0 : 1;
	unsigned int tempLabel = nID;
	while(tempLabel > 0)
	{
		tempLabel /= 10;
		labelLength++;
	}

	// draw label
	cv::rectangle(imageFrame, cv::Rect((int)pointArray[0].x, (int)pointArray[0].y - 10, 7 * labelLength, 14), curColor, CV_FILLED);
	cv::putText(imageFrame, std::to_string(nID), cv::Point((int)pointArray[0].x, (int)pointArray[0].y-1), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255));

}

void psn::DrawTriangleWithID(cv::Mat &imageFrame, PSN_Point2D &point, unsigned int nID, std::vector<cv::Scalar> &vecColors)
{
	double size = 5;

	// get color
	cv::Scalar curColor = psn::getColorByID(vecColors, nID);

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

void psn::DrawLine(cv::Mat &imageFrame, std::vector<PSN_Point2D> &pointArray, unsigned int nID, std::vector<cv::Scalar> &vecColors, int lineThickness)
{
	// get color
	cv::Scalar curColor = psn::getColorByID(vecColors, nID);

	for(size_t pointIdx = 0; pointIdx < pointArray.size() - 1; pointIdx++)
	{
		cv::line(imageFrame, pointArray[pointIdx].cv(), pointArray[pointIdx+1].cv(), curColor, lineThickness);
	}
}

PSN_Point2D psn::GetLocationOnTopView_PETS2009(PSN_Point3D &curPoint, bool bZoom)
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
	homogeniousPoint.at<double>(0, 0) = curPoint.x;
	homogeniousPoint.at<double>(1, 0) = curPoint.y;
	homogeniousPoint.at<double>(2, 0) = 1.0;

	cv::Mat resultVec = AffineMat * homogeniousPoint;

	if (bZoom)
	{
		// zoom
		resultVec.at<double>(0, 0) = 1.5 * (resultVec.at<double>(0, 0) - 150.0);
		resultVec.at<double>(1, 0) = 1.5 * (resultVec.at<double>(1, 0) - 150.0);

		//// zoom2
		//resultVec.at<double>(0, 0) = 1.5 * (1.5 * (resultVec.at<double>(0, 0) - 150.0) - 100.0);
		//resultVec.at<double>(1, 0) = 1.5 * (1.5 * (resultVec.at<double>(1, 0) - 150.0) - 100.0);
	}

	return PSN_Point2D(resultVec.at<double>(0, 0), resultVec.at<double>(1, 0));
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
void psn::printLog(const char *filename, const char *strLog)
{
	char strPrint[128];
	sprintf_s(strPrint, "%s\n", strLog);
		
	try
	{
		FILE *fp;
		fopen_s(&fp, filename, "a");
		fprintf(fp, strPrint);
		fclose(fp);
	}
	catch(DWORD dwError)
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

/////////////////////////////////////////////////////////////////////////
// CTrackletCombination MEMBER FUNCTIONS
/////////////////////////////////////////////////////////////////////////

CTrackletCombination::CTrackletCombination(void)
{
	memset(this->tracklets, NULL, sizeof(stTracklet2D*) * NUM_CAM);
	numTracklets = 0;
}

CTrackletCombination& CTrackletCombination::operator=(const CTrackletCombination &a)
{ 
	memcpy(this->tracklets, a.tracklets, sizeof(stTracklet2D*) * NUM_CAM);
	this->numTracklets = a.numTracklets;
	return *this;
}

bool CTrackletCombination::operator==(const CTrackletCombination &a)
{ 
	if(this->numTracklets != a.numTracklets)
	{
		return false;
	}
	for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		if(this->tracklets[camIdx] != a.tracklets[camIdx])
		{
			return false;
		}
	}
	return true;
}

void CTrackletCombination::set(unsigned int camIdx, stTracklet2D *tracklet)
{ 
	if(this->tracklets[camIdx] == tracklet)
	{
		return;
	}
	if(NULL == this->tracklets[camIdx])
	{
		numTracklets++;
	}
	else if(NULL == tracklet)
	{
		numTracklets--;
	}
	this->tracklets[camIdx] = tracklet;
}

stTracklet2D* CTrackletCombination::get(unsigned int camIdx)
{ 
	return this->tracklets[camIdx]; 
}

void CTrackletCombination::print()
{
	printf("[");
	for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		if(NULL != tracklets[camIdx]){ printf("%d", tracklets[camIdx]->id); }
		else{ printf("x"); }

		if(camIdx < NUM_CAM - 1){ printf(","); }
		else{ printf("]\n"); }
	}
}

bool CTrackletCombination::checkCoupling(CTrackletCombination &a)
{
	for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		if(this->tracklets[camIdx] == a.tracklets[camIdx] && NULL != this->tracklets[camIdx])
		{
			return true;
		}
	}
	return false;
}

unsigned int CTrackletCombination::size()
{
	return this->numTracklets;
}


/////////////////////////////////////////////////////////////////////////
// Track3D MEMBER FUNCTIONS
/////////////////////////////////////////////////////////////////////////
Track3D::Track3D()
	: id(0)
	, bActive(true)
	, bValid(true)
	, tree(NULL)
	, parentTrack(NULL)
	, timeStart(0)
	, timeEnd(0)
	, timeGeneration(0)
	, duration(0)
	, costTotal(0.0)
	, costReconstruction (0.0)
	, costLink(0.0)
	, costEnter(0.0)
	, costExit(0.0)
	, loglikelihood(0.0)
	, GTProb(0.0)
	, BranchGTProb(0.0)
	, bWasBestSolution(true)
	, bCurrentBestSolution(false)
	, bNewTrack(true)
{
}

Track3D::~Track3D()
{

}

void Track3D::Initialize(unsigned int id, Track3D *parentTrack, unsigned int timeGeneration, CTrackletCombination &trackletCombination)
{
	this->id = id;
	this->curTracklet2Ds = trackletCombination;
	if(NULL == parentTrack)
	{
		this->timeStart = timeGeneration;
		this->timeEnd = timeGeneration;
		this->duration = 1;
		return;
	}
	this->tree = parentTrack->tree;
	this->parentTrack = parentTrack;
	this->timeStart = parentTrack->timeStart;
	this->timeEnd = timeGeneration;
	this->timeGeneration = timeGeneration;
	this->duration = this->timeEnd - this->timeStart + 1;
	this->costEnter = parentTrack->costEnter;
	this->loglikelihood = parentTrack->loglikelihood;
}

void Track3D::RemoveFromTree()
{
	// find valid parent (for children's adoption)
	Track3D *newParentTrack = this->parentTrack;
	while(NULL != newParentTrack && !newParentTrack->bValid)
	{
		newParentTrack = newParentTrack->parentTrack;
	}

	// remove from children's parent pointer and adopt to new parent
	for(std::deque<Track3D*>::iterator childTrackIter = this->childrenTrack.begin();
		childTrackIter != this->childrenTrack.end();
		childTrackIter++)
	{
		(*childTrackIter)->parentTrack = newParentTrack;
		if(NULL != (*childTrackIter)->parentTrack)
		{
			(*childTrackIter)->parentTrack->childrenTrack.push_back(*childTrackIter);
		}
	}

	// remove from parent's children list
	if(NULL != this->parentTrack)
	{
		for(std::deque<Track3D*>::iterator childTrackIter = this->parentTrack->childrenTrack.begin();
			childTrackIter != this->parentTrack->childrenTrack.end();
			childTrackIter++)
		{
			if((*childTrackIter)->id != this->id)
			{ continue; }

			this->parentTrack->childrenTrack.erase(childTrackIter);
			break;
		}
	}
}

std::deque<Track3D*> Track3D::GatherValidChildrenTracks(Track3D* validParentTrack, std::deque<Track3D*> &targetChildrenTracks)
{
	PSN_TrackSet newChildrenTracks;
	for(PSN_TrackSet::iterator childTrackIter = targetChildrenTracks.begin();
		childTrackIter != targetChildrenTracks.end();
		childTrackIter++)
	{
		if((*childTrackIter)->bValid)
		{
			(*childTrackIter)->parentTrack = validParentTrack;
			newChildrenTracks.push_back(*childTrackIter);
			continue;
		}
		PSN_TrackSet foundChildrenTracks = GatherValidChildrenTracks(validParentTrack, (*childTrackIter)->childrenTrack);
		newChildrenTracks.insert(newChildrenTracks.end(), foundChildrenTracks.begin(), foundChildrenTracks.end());
	}

	return newChildrenTracks;
}

//#define KALMAN_PROCESSNOISE_SIG (1.0E-5)
//#define KALMAN_MEASUREMENTNOISE_SIG (1.0E-5)
//#define KALMAN_POSTERROR_COV (0.1)
//#define KALMAN_CONFIDENCE_LEVEN (9)
//void Track3D::SetKalmanFilter(PSN_Point3D &initialPoint)
//{
//	this->KF.init(6, 3, 0);
//	this->KFMeasurement = cv::Mat(3, 1, CV_32FC1);
//
//	cv::setIdentity(this->KF.transitionMatrix); // [1,0,0,1,0,0; 0,1,0,0,1,0; 0,0,1,0,0,1; 0,0,0,1,0,0, ...]
//	this->KF.transitionMatrix.at<float>(0, 3) = 1.0f;
//	this->KF.transitionMatrix.at<float>(1, 4) = 1.0f;
//	this->KF.transitionMatrix.at<float>(2, 5) = 1.0f;
//
//	cv::setIdentity(this->KF.measurementMatrix);
//	cv::setIdentity(this->KF.processNoiseCov, cv::Scalar::all(KALMAN_PROCESSNOISE_SIG));
//	cv::setIdentity(this->KF.measurementNoiseCov, cv::Scalar::all(KALMAN_MEASUREMENTNOISE_SIG));
//	cv::setIdentity(this->KF.errorCovPost, cv::Scalar::all(KALMAN_POSTERROR_COV));
//
//	this->KF.statePost.at<float>(0, 0) = (float)initialPoint.x;
//	this->KF.statePost.at<float>(1, 0) = (float)initialPoint.y;
//	this->KF.statePost.at<float>(2, 0) = (float)initialPoint.z;
//	this->KF.statePost.at<float>(3, 0) = 0.0f;
//	this->KF.statePost.at<float>(4, 0) = 0.0f;
//	this->KF.statePost.at<float>(5, 0) = 0.0f;
//	cv::Mat curKFPrediction = this->KF.predict();
//}


/////////////////////////////////////////////////////////////////////////
// TrackTree MEMBER FUNCTIONS
/////////////////////////////////////////////////////////////////////////
TrackTree::TrackTree()
	: id(0)
	, timeGeneration(0)
	, bValid(true)
	, numMeasurements(0)
//	, maxGTProb(0.0)
{
}

TrackTree::~TrackTree()
{
}

void TrackTree::Initialize(unsigned int id, Track3D *seedTrack, unsigned int timeGeneration, std::list<TrackTree> &treeList)
{
	this->id = id;
	this->timeGeneration = timeGeneration;
	this->tracks.push_back(seedTrack);
	
	treeList.push_back(*this);
	seedTrack->tree = &treeList.back();
	
	for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		if(NULL == seedTrack->curTracklet2Ds.get(camIdx))
		{
			continue;
		}
		stTracklet2DInfo newTrackletInfo;
		newTrackletInfo.tracklet2D = seedTrack->curTracklet2Ds.get(camIdx);
		newTrackletInfo.queueRelatedTracks.push_back(seedTrack);
		seedTrack->tree->tracklet2Ds[camIdx].push_back(newTrackletInfo);
		seedTrack->tree->numMeasurements++;
	}
	//this->m_queueActiveTrees.push_back(seedTrack->tree);
}

/************************************************************************
 Method Name: ResetGlobalTrackProbInTree
 Description: 
	- 
 Input Arguments:
	- 
 Return Values:
	- none
************************************************************************/
void TrackTree::ResetGlobalTrackProbInTree()
{
	for(std::deque<Track3D*>::iterator trackIter = this->tracks.begin();
		trackIter != this->tracks.end();
		trackIter++)
	{
		(*trackIter)->GTProb = 0.0;
	}
}

Track3D* TrackTree::FindPruningPoint(unsigned int timeWindowStart, Track3D *rootOfBranch)
{
	if(NULL == rootOfBranch)
	{
		if(0 == this->tracks.size())
		{ return NULL; }
		rootOfBranch = this->tracks[0];
	}

	if(rootOfBranch->timeGeneration >= timeWindowStart)
	{ return NULL; }

	// if more than one child placed inside of processing window, then the current node is the pruning point
	Track3D *curPruningPoint = NULL;
	for(PSN_TrackSet::iterator trackIter = rootOfBranch->childrenTrack.begin();
		trackIter != rootOfBranch->childrenTrack.end();
		trackIter++)
	{
		curPruningPoint = FindPruningPoint(timeWindowStart, *trackIter);
		if(NULL == curPruningPoint)
		{ break; }
	}

	if(NULL == curPruningPoint)
	{ return rootOfBranch; }

	return curPruningPoint;
}

bool TrackTree::CheckBranchContainsBestSolution(Track3D *rootOfBranch)
{
	if(rootOfBranch->bCurrentBestSolution)
	{ return true; }

	for(PSN_TrackSet::iterator trackIter = rootOfBranch->childrenTrack.begin();
		trackIter != rootOfBranch->childrenTrack.end();
		trackIter++)
	{
		if(CheckBranchContainsBestSolution(*trackIter))
		{ return true; }
	}
	return false;
}

/************************************************************************
 Method Name: SetValidityFlagInTrackBranch
 Description: 
	- Recursively clear validity flags on descendants
 Input Arguments:
	- queueChildrenTracks: queue of children tracks
 Return Values:
	- none
************************************************************************/
void TrackTree::SetValidityFlagInTrackBranch(Track3D* rootOfBranch, bool bValid)
{
	rootOfBranch->bValid = bValid;
	for(std::deque<Track3D*>::iterator trackIter = rootOfBranch->childrenTrack.begin();
		trackIter != rootOfBranch->childrenTrack.end();
		trackIter++)
	{
		SetValidityFlagInTrackBranch(*trackIter, bValid);
	}
}


/************************************************************************
 Method Name: GetTracksInBranch
 Description: 
	- Recursively gather pointers of tracks in descendants
 Input Arguments:
	-
	-
 Return Values:
	- 
************************************************************************/
void TrackTree::GetTracksInBranch(Track3D* rootOfBranch, std::deque<Track3D*> &queueOutput)
{
	queueOutput.push_back(rootOfBranch);
	for(std::deque<Track3D*>::iterator trackIter = rootOfBranch->childrenTrack.begin();
		trackIter != rootOfBranch->childrenTrack.end();
		trackIter++)
	{
		GetTracksInBranch(*trackIter, queueOutput);
	}
}


/************************************************************************
 Method Name: MakeTreeNodesWithChildren
 Description: 
	- Recursively generate track node information
 Input Arguments:
	- queueChildrenTracks: queue of children tracks
	- parentNodeIdx: parent node index
	- outQueueNodes: output node index queue
 Return Values:
	- none
************************************************************************/
void TrackTree::MakeTreeNodesWithChildren(std::deque<Track3D*> queueChildrenTracks, const unsigned int parentNodeIdx, std::deque<unsigned int> &outQueueNodes)
{
	for(std::deque<Track3D*>::iterator trackIter = queueChildrenTracks.begin();
		trackIter != queueChildrenTracks.end();
		trackIter++)
	{
		outQueueNodes.push_back(parentNodeIdx);
		MakeTreeNodesWithChildren((*trackIter)->childrenTrack, (unsigned int)outQueueNodes.size(), outQueueNodes);
	}
}


/************************************************************************
 Method Name: GTProbOfBrach
 Description: 
	- Recursively sum global track probabilities of track branch
 Input Arguments:
	-
 Return Values:
	- double: sum of global track probability
************************************************************************/
double TrackTree::GTProbOfBrach(Track3D *rootOfBranch)
{
	double GTProb = rootOfBranch->GTProb;
	for(std::deque<Track3D*>::iterator trackIter = rootOfBranch->childrenTrack.begin();
		trackIter != rootOfBranch->childrenTrack.end();
		trackIter++)
	{
		GTProb += GTProbOfBrach(*trackIter);
	}
	return GTProb;
}


/************************************************************************
 Method Name: MaxGTProbOfBrach
 Description: 
	- find the maximum global track probability in the branch and set
	that value into 'BranchGTProb' property
 Input Arguments:
	- rootOfBranch: a pointer of the seed track of the branch
 Return Values:
	- double: maximum value of global track probability in the branch
************************************************************************/
double TrackTree::MaxGTProbOfBrach(Track3D *rootOfBranch)
{
	double MaxGTProb = rootOfBranch->GTProb;
	for(std::deque<Track3D*>::iterator trackIter = rootOfBranch->childrenTrack.begin();
		trackIter != rootOfBranch->childrenTrack.end();
		trackIter++)
	{
		double curGTProb = MaxGTProbOfBrach(*trackIter);
		if(MaxGTProb < curGTProb)
		{
			MaxGTProb = curGTProb;
		}
	}
	rootOfBranch->BranchGTProb = MaxGTProb;
	return MaxGTProb;
}


/************************************************************************
 Method Name: InvalidateBranchWithMinGTProb
 Description: 
	- 
 Input Arguments:
	-
 Return Values:
	- none
************************************************************************/
void TrackTree::InvalidateBranchWithMinGTProb(Track3D *rootOfBranch, double minGTProb)
{
	if(rootOfBranch->GTProb < minGTProb)
	{
		rootOfBranch->bValid = false;
	}

	for(std::deque<Track3D*>::iterator trackIter = rootOfBranch->childrenTrack.begin();
		trackIter != rootOfBranch->childrenTrack.end();
		trackIter++)
	{
		InvalidateBranchWithMinGTProb(*trackIter, minGTProb);
	}
}


/************************************************************************
 Method Name: FindMaxGTProbBranch
 Description: 
	- find 
 Input Arguments:
	- queueChildrenTracks: queue of children tracks
 Return Values:
	- none
************************************************************************/
Track3D* TrackTree::FindMaxGTProbBranch(Track3D* branchSeedTrack, size_t timeIndex)
{
	if(branchSeedTrack->timeGeneration >= timeIndex){ return NULL; }

	Track3D* maxGTProbChild = NULL;
	for(std::deque<Track3D*>::iterator trackIter = branchSeedTrack->childrenTrack.begin();
		trackIter != branchSeedTrack->childrenTrack.end();
		trackIter++)
	{
		Track3D* curGTProbChild = FindMaxGTProbBranch((*trackIter), timeIndex);
		if(NULL == curGTProbChild){ continue; }
		if(NULL == maxGTProbChild || curGTProbChild > maxGTProbChild)
		{
			maxGTProbChild = curGTProbChild;
		}		
	}	
	return NULL == maxGTProbChild? branchSeedTrack : maxGTProbChild;
}

/************************************************************************
 Method Name: FindOldestTrackInBranch
 Description: 
	- 
 Input Arguments:
	- 
 Return Values:
	- Track3D*: 
************************************************************************/
Track3D* TrackTree::FindOldestTrackInBranch(Track3D *trackInBranch, int nMostPreviousFrameIdx)
{
	Track3D *oldestTrack = trackInBranch;
	while (true) 
	{
		if (NULL == oldestTrack->parentTrack) { break; }
		if (nMostPreviousFrameIdx >= (int)oldestTrack->parentTrack->timeGeneration) { break; }
		oldestTrack = oldestTrack->parentTrack;
	}
	return oldestTrack;
}

/************************************************************************
 Method Name: CheckConnectivityOfTrees
 Description: 
	- Check whether two trees share a common measurement or not
 Input Arguments:
	- tree1: 
	- tree2: 
 Return Values:
	- bool: true for connected trees
************************************************************************/
bool TrackTree::CheckConnectivityOfTrees(TrackTree *tree1, TrackTree *tree2)
{
	for(unsigned int camIdx = 0; camIdx < NUM_CAM; camIdx++)
	{
		if(0 == tree1->tracklet2Ds[camIdx].size() || 0 == tree2->tracklet2Ds[camIdx].size())
		{
			continue;
		}
		for(std::deque<stTracklet2DInfo>::iterator info1Iter = tree1->tracklet2Ds[camIdx].begin();
			info1Iter != tree1->tracklet2Ds[camIdx].end();
			info1Iter++)
		{
			for(std::deque<stTracklet2DInfo>::iterator info2Iter = tree2->tracklet2Ds[camIdx].begin();
				info2Iter != tree2->tracklet2Ds[camIdx].end();
				info2Iter++)
			{
				if((*info1Iter).tracklet2D == (*info2Iter).tracklet2D)
				{
					return true;
				}
			}
		}
	}
	return false;
}

/////////////////////////////////////////////////////////////////////////
// CPSNWhere_Manager
/////////////////////////////////////////////////////////////////////////

/************************************************************************
 Method Name: CPSNWhere_Manager
 Description: 
	- 클래스 생성자
 Input Arguments:
	-
	-
 Return Values:
	- class instance
************************************************************************/
CPSNWhere_Manager::CPSNWhere_Manager(void)
{
}


/************************************************************************
 Method Name: ~CPSNWhere_Manager
 Description: 
	- 클래스 종료자
 Input Arguments:
	-
	-
 Return Values:
	- none
************************************************************************/
CPSNWhere_Manager::~CPSNWhere_Manager(void)
{
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
void CPSNWhere_Manager::printLog(const char *filename, const char *strLog)
{
	char strPrint[128];
	sprintf_s(strPrint, "%s\n", strLog);
		
	try
	{
		FILE *fp;
		fopen_s(&fp, filename, "a");
		fprintf(fp, strPrint);
		fclose(fp);
	}
	catch(DWORD dwError)
	{
		printf("[ERROR] cannot open logging file! error code %d\n", dwError);
		return;
	}
}


/************************************************************************
 Method Name: Triangulation
 Description: 
	- 
 Input Arguments:
	- 
 Return Values:
	- 
************************************************************************/
double CPSNWhere_Manager::Triangulation(PSN_Line &line1, PSN_Line &line2, PSN_Point3D &midPoint3D)
{
#ifdef PSN_DEBUG_MODE_
	//printf("[CPSNWhere_Associator3D](DistanceBackProjectionLine) start\n");
#endif
	
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

#ifdef PSN_DEBUG_MODE_
	//printf("[CPSNWhere_Associator3D](DistanceBackProjectionLine) return:%f\n", fDistance);
#endif
	return (closePoint1 - closePoint2).norm_L2();
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
cv::Mat CPSNWhere_Manager::MakeMatTile(std::vector<cv::Mat> *imageArray, unsigned int numRows, unsigned int numCols)
{

#ifdef PSN_DEBUG_MODE_
	//printf("[CPSNWhere_Manager](MakeMatTile) start\n");
#endif
	// column first arrangement
	unsigned int numImage = (unsigned int)imageArray->size();	
	unsigned int acturalNumCols = numImage < numCols ? numImage : numCols;
	unsigned int acturalNumRows = (0 == numImage % numCols) ? numImage / numCols : numImage / numCols + 1;

#ifdef PSN_DEBUG_MODE_
	//printf("[CPSNWhere_Manager](MakeMatTile) numImage:%d, numRows:%d, numCols:%d\n", numImage, acturalNumRows, acturalNumCols);
#endif

	// find maximum size image
	cv::Size maxSize = (*imageArray)[0].size();
	for(unsigned int imageIdx = 0; imageIdx < numImage; imageIdx++)
	{
		if((*imageArray)[imageIdx].rows > maxSize.height){ maxSize.height = (*imageArray)[imageIdx].rows; }
		if((*imageArray)[imageIdx].cols > maxSize.width){ maxSize.width = (*imageArray)[imageIdx].cols; }
	}
#ifdef PSN_DEBUG_MODE_
	//printf("[CPSNWhere_Manager](MakeMatTile) maxWidth:%d, maxHeight:%d\n", maxSize.width, maxSize.height);
#endif

	// make augmenting matrix
	std::vector<cv::Mat> augMats;
	cv::Mat baseMat;
	if(1 == (*imageArray)[0].channels()){	baseMat = cv::Mat::zeros(maxSize, CV_8UC1); }
	else{ baseMat = cv::Mat::zeros(maxSize, CV_8UC3); }	
	//for(unsigned int imageIdx = 0; imageIdx < numImage; imageIdx++)
	//{
	//	cv::Mat augMat = baseMat.clone();
	//	(*imageArray)[imageIdx].copyTo(augMat(cv::Rect(0, 0, (*imageArray)[imageIdx].cols, (*imageArray)[imageIdx].rows)));
	//	augMats.push_back(augMat);
	//}
	unsigned int numAugMats = acturalNumRows * acturalNumCols;
	for(unsigned int imageIdx = 0; imageIdx < numAugMats; imageIdx++)
	{
		cv::Mat augMat = baseMat.clone();
		if(imageIdx < numImage)
		{
			(*imageArray)[imageIdx].copyTo(augMat(cv::Rect(0, 0, (*imageArray)[imageIdx].cols, (*imageArray)[imageIdx].rows)));
		}
		augMats.push_back(augMat);
	}

	// matrix concatenation
	cv::Mat hConcatMat;
	cv::Mat resultMat;
	for(unsigned int rowIdx = 0; rowIdx < acturalNumRows; rowIdx++)
	{
		unsigned int startIdx = rowIdx * acturalNumCols;		
		for(unsigned int colIdx = 0; colIdx < acturalNumCols; colIdx++)
		{
			if(0 == colIdx)
			{
				hConcatMat = augMats[startIdx].clone();
				continue;
			}
			cv::hconcat(hConcatMat, augMats[startIdx + colIdx], hConcatMat);
		}

		if(0 == rowIdx)
		{
			resultMat = hConcatMat.clone();
			continue;
		}
		cv::vconcat(resultMat, hConcatMat, resultMat);
	}

#ifdef PSN_DEBUG_MODE_
	//printf("[CPSNWhere_Manager](MakeMatTile) end\n");
#endif

	return resultMat;
}

std::vector<stDetection> CPSNWhere_Manager::ReadDetectionResultWithTxt(std::string strDatasetPath, unsigned int camIdx, unsigned int frameIdx) {
	std::vector<stDetection> vec_result;
	char textfile_path[128];
	int num_detection = 0;
	float x, y, w, h, temp;

	FILE *fid;
	try {
		switch(PSN_DETECTION_TYPE)	{
		case 0:	// Head
			sprintf_s(textfile_path, sizeof(textfile_path), "%s\\detectionResult\\cam%d\\%04d.txt", strDatasetPath.c_str(), camIdx, frameIdx);
			fopen_s(&fid, textfile_path, "r");
			if(NULL == fid)	{
#ifdef PSN_DEBUG_MODE_
				printf("[WARNING]...[CPSNWhere_Manager](ReadDetectionResultWithTxt) file reading is failed\n");
#endif
				return vec_result;
			}
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
			if(NULL == fid)
			{
#ifdef PSN_DEBUG_MODE_
				printf("[WARNING]...[CPSNWhere_Manager](ReadDetectionResultWithTxt) file reading is failed\n");
#endif
				return vec_result;
			}

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

std::vector<stTrack2DResult> CPSNWhere_Manager::Read2DTrackResultWithTxt(std::string strDatasetPath, unsigned int frameIdx)
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
