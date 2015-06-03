/**************************************************************************
* Name : Multi-camera Multi-target Tracking in PSN
* Author : HAANJU.YOO
* Initial Date : 2013.08.29
* Last Update : 2013.08.29
* Version : 0.9
***************************************************************************
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
*        $$$$$$$o $$$$$$$$$$$$$$$F"          `$$$$$$$$$$$$$$$$$$$$b.0
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
***************************************************************************
*
* Using DDK dataset
*
**************************************************************************/
#include "stdafx.h"
#include <iostream>
#include <omp.h>
#include "PSNWhere.h"

int _tmain(int argc, _TCHAR* argv[])
{
	CPSNWhere psnWhere;
	cv::Mat inputFrame[NUM_CAM];
	char inputFilePath[300];

	// argument handling
#ifdef _UNICODE
	std::wstring wstrDatasetPath(argv[1]);

	std::string strDatasetPath(wstrDatasetPath.begin(), wstrDatasetPath.end());
#else
	std::string strDatasetPath = argv[1];
#endif

	// input frame range
	//int frameIdxStart = 0;
	//int frameIdxEnd = 300;
	
	// P6S1
	//int frameIdxStart = 100;
	//int frameIdxEnd = 430;

	// PETS2009
	int frameIdxStart = 0;
	int frameIdxEnd = 794;	

	/////////////////////////////////////////////////////////////////
	// INITIALIZATION
	/////////////////////////////////////////////////////////////////
	psnWhere = CPSNWhere();
	psnWhere.Initialize(strDatasetPath);

	/////////////////////////////////////////////////////////////////
	// MAIN LOOP
	/////////////////////////////////////////////////////////////////
	for(int frameIdx = frameIdxStart; frameIdx <= frameIdxEnd; frameIdx++)
	{
		//---------------------------------------------------
		// FRAME GRABBING
		//---------------------------------------------------
#pragma omp parallel for
		for(int camIdx = 0; camIdx < NUM_CAM; camIdx++)
		{			
			unsigned int curCamID = CAM_ID[camIdx];
			if (PSN_INPUT_TYPE)
			{
				sprintf_s(inputFilePath, sizeof(inputFilePath), "%s\\View_%03d\\frame_%04d.jpg", strDatasetPath.c_str(), curCamID, frameIdx);						
			}
			else
			{
				sprintf_s(inputFilePath, sizeof(inputFilePath), "%s\\%d_%d.jpg", strDatasetPath.c_str(), curCamID, frameIdx);						
			}
			inputFrame[camIdx] = cv::imread(inputFilePath, cv::IMREAD_COLOR);
			if(!inputFrame[camIdx].data)
			{
				std::cout << "Can't open the input frame" << std::endl;
				inputFrame[camIdx].release();
				return -1;
			}
		}
		
		//---------------------------------------------------
		// TRACKING
		//---------------------------------------------------
		psnWhere.TrackPeople(inputFrame, frameIdx);

		//---------------------------------------------------
		// MEMORY CLEARING
		//---------------------------------------------------
		for(int camIdx = 0; camIdx < NUM_CAM; camIdx++)
		{
			inputFrame[camIdx].release();
		}
	}


	/////////////////////////////////////////////////////////////////
	// TERMINATION
	/////////////////////////////////////////////////////////////////
	psnWhere.Finalize();

	return 0;
}

//()()
//('')HAANJU.YOO
