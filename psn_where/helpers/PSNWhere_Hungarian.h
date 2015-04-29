/******************************************************************************
 * Name : PSNWhere_Hungarian
 * Date : 2014.02.07
 * Author : HAANJU.YOO
 * Version : 1.0
 * Description :
 *	- Class for Hungarian matching mathod
 *	- C++ porting result of Alex Melin's MATLAB code
 *	- Implemented with only c standard library
 *
 ******************************************************************************
 *                                            ....
 *                                           W$$$$$u
 *                                           $$$$F**+           .oW$$$eu
 *                                           ..ueeeWeeo..      e$$$$$$$$$
 *                                       .eW$$$$$$$$$$$$$$$b- d$$$$$$$$$$W
 *                           ,,,,,,,uee$$$$$$$$$$$$$$$$$$$$$ H$$$$$$$$$$$~
 *                        :eoC$$$$$$$$$$$C""?$$$$$$$$$$$$$$$ T$$$$$$$$$$"
 *                         $$$*$$$$$$$$$$$$$e "$$$$$$$$$$$$$$i$$$$$$$$F"
 *                         ?f"!?$$$$$$$$$$$$$$ud$$$$$$$$$$$$$$$$$$$$*Co
 *                         $   o$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
 *                 !!!!m.*eeeW$$$$$$$$$$$f?$$$$$$$$$$$$$$$$$$$$$$$$$$$$$U
 *                 !!!!!! !$$$$$$$$$$$$$$  T$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
 *                  *!!*.o$$$$$$$$$$$$$$$e,d$$$$$$$$$$$$$$$$$$$$$$$$$$$$$:
 *                 "eee$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$C
 *                b ?$$$$$$$$$$$$$$**$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$!
 *                Tb "$$$$$$$$$$$$$$*uL"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
 *                 $$o."?$$$$$$$$F" u$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
 *                  $$$$en ```    .e$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
 *                   $$$B*  =*"?.e$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$F
 *                    $$$W"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
 *                     "$$$o#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
 *                    R: ?$$$W$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" :!i.
 *                     !!n.?$???""``.......,``````"""""""""""``   ...+!!!
 *                      !* ,+::!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*`
 *                      "!?!!!!!!!!!!!!!!!!!!~ !!!!!!!!!!!!!!!!!!!~`
 *                      +!!!!!!!!!!!!!!!!!!!! !!!!!!!!!!!!!!!!!!?!`
 *                    .!!!!!!!!!!!!!!!!!!!!!' !!!!!!!!!!!!!!!, !!!!
 *                   :!!!!!!!!!!!!!!!!!!!!!!' !!!!!!!!!!!!!!!!! `!!:
 *                .+!!!!!!!!!!!!!!!!!!!!!~~!! !!!!!!!!!!!!!!!!!! !!!.
 *               :!!!!!!!!!!!!!!!!!!!!!!!!!.`:!!!!!!!!!!!!!!!!!:: `!!+
 *               "~!!!!!!!!!!!!!!!!!!!!!!!!!!.~!!!!!!!!!!!!!!!!!!!!.`!!:
 *                   ~~!!!!!!!!!!!!!!!!!!!!!!! ;!!!!~` ..eeeeeeo.`+!.!!!!.
 *                 :..    `+~!!!!!!!!!!!!!!!!! :!;`.e$$$$$$$$$$$$$u .
 *                 $$$$$$beeeu..  `````~+~~~~~" ` !$$$$$$$$$$$$$$$$ $b
 *                 $$$$$$$$$$$$$$$$$$$$$UU$U$$$$$ ~$$$$$$$$$$$$$$$$ $$o
 *                !$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$. $$$$$$$$$$$$$$$~ $$$u
 *                !$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$! $$$$$$$$$$$$$$$ 8$$$$.
 *                !$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$X $$$$$$$$$$$$$$`u$$$$$W
 *                !$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$! $$$$$$$$$$$$$".$$$$$$$:
 *                 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$  $$$$$$$$$$$$F.$$$$$$$$$
 *                 ?$$$$$$$$$$$$$$$$$$$$$$$$$$$$f $$$$$$$$$$$$' $$$$$$$$$$.
 *                  $$$$$$$$$$$$$$$$$$$$$$$$$$$$ $$$$$$$$$$$$$  $$$$$$$$$$!
 *                  "$$$$$$$$$$$$$$$$$$$$$$$$$$$ ?$$$$$$$$$$$$  $$$$$$$$$$!
 *                   "$$$$$$$$$$$$$$$$$$$$$$$$Fib ?$$$$$$$$$$$b ?$$$$$$$$$
 *                     "$$$$$$$$$$$$$$$$$$$$"o$$$b."$$$$$$$$$$$  $$$$$$$$'
 *                    e. ?$$$$$$$$$$$$$$$$$ d$$$$$$o."?$$$$$$$$H $$$$$$$'
 *                   $$$W.`?$$$$$$$$$$$$$$$ $$$$$$$$$e. "??$$$f .$$$$$$'
 *                  d$$$$$$o "?$$$$$$$$$$$$ $$$$$$$$$$$$$eeeeee$$$$$$$"
 *                  $$$$$$$$$bu "?$$$$$$$$$ 3$$$$$$$$$$$$$$$$$$$$*$$"
 *                 d$$$$$$$$$$$$$e. "?$$$$$:`$$$$$$$$$$$$$$$$$$$$8
 *         e$$e.   $$$$$$$$$$$$$$$$$$+  "??f "$$$$$$$$$$$$$$$$$$$$c
 *        $$$$$$$o $$$$$$$$$$$$$$$F"          `$$$$$$$$$$$$$$$$$$$$b.
 *       M$$$$$$$$U$$$$$$$$$$$$$F"              ?$$$$$$$$$$$$$$$$$$$$$u
 *       ?$$$$$$$$$$$$$$$$$$$$F                   "?$$$$$$$$$$$$$$$$$$$$u
 *        "$$$$$$$$$$$$$$$$$$"                       ?$$$$$$$$$$$$$$$$$$$$o
 *          "?$$$$$$$$$$$$$F                            "?$$$$$$$$$$$$$$$$$$
 *             "??$$$$$$$F                                 ""?3$$$$$$$$$$$$F
 *                                                       .e$$$$$$$$$$$$$$$$'
 *                                                      u$$$$$$$$$$$$$$$$$
 *                                                     `$$$$$$$$$$$$$$$$"
 *                                                      "$$$$$$$$$$$$F"
 *                                                        ""?????""
 *
 ******************************************************************************/

#pragma once

#include <vector>

typedef struct _stMatchInfo
{
	std::vector<unsigned int> rows;
	std::vector<unsigned int> cols;
	std::vector<float> matchCosts;
	std::vector<std::vector<float>> costMatrix;
	std::vector<std::vector<float>> matchMatrix;
	float totalCost;
} stMatchInfo;

class CPSNWhere_Hungarian
{
public:
	CPSNWhere_Hungarian(void);
	~CPSNWhere_Hungarian(void);

	void Initialize(std::vector<float> costArray, unsigned int numRows, unsigned int numCols);
	void Initialize(std::vector<std::vector<float>> costMatrix);
	void Initialize(float* costArray, unsigned int numRows, unsigned int numCols);
	void Initialize(float** cost2DArray, unsigned int numRows, unsigned int numCols);
	stMatchInfo* Match();
	void Finalize(void);
	void PrintCost(void);
	void PrintMatch(void);

	static inline void psnMatSum(std::vector<std::vector<float>> &argInputMat, std::vector<float> &vecDest, unsigned int nDirection);

private:
	void step1(void);
	void step2(
		std::vector<float> &rCov, 
		std::vector<float> &cCov, 
		std::vector<std::vector<float>> &matM, 
		std::vector<std::vector<float>> &matPCond);
	void step3(
		std::vector<float> &cCov, 
		std::vector<std::vector<float>> &matM);
	void step4(
		std::vector<std::vector<float>> &matPCond, 
		std::vector<float> &rCov, 
		std::vector<float> &cCov, 
		std::vector<std::vector<float>> &matM, 
		std::vector<float> &zR, 
		std::vector<float> &zC);
	void step5(
		std::vector<std::vector<float>> &matM,
		std::vector<float> &zR,
		std::vector<float> &zC,
		std::vector<float> &rCov,
		std::vector<float> &cCov);
	void step6(
		std::vector<std::vector<float>> &matPCond,
		std::vector<float> &rCov,
		std::vector<float> &cCov);

	unsigned int minLineCover(std::vector<std::vector<float>> &argMatrix);
	void CostMatrixPreprocessing(void);
	void MatchResultPostProcessing(void);

	bool m_bInit;
	stMatchInfo m_matchInfo;
	std::vector<std::vector<float>> m_costMatrix;	
	std::vector<std::vector<float>> m_matMatching;
	std::vector<std::vector<float>> m_matPCond;
	std::vector<std::vector<bool>> m_matIsFinite;
	//cv::Mat m_costMatrix;
	//cv::Mat m_matPCond;
	unsigned int m_numRows;
	unsigned int m_numCols;
	unsigned int m_nSquareSize;
	unsigned int m_nStepNum;
	float m_fInfiniteReplacementCost;
};