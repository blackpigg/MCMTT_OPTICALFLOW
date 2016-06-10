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
#include "PSNWhere.h"
#include "helpers\ParameterParser.h"

bool ReadParams(_TCHAR** parameterPath, 
	stParamsMain         &outputParamsMain, 
	stParamsDetection    &outputParamsDetection, 
	stParamsTracker2D    &outputParamsTracker2D,
	stParamsAssociator3D &outputParamsAssociator3D);

int _tmain(int argc, _TCHAR* argv[])
{
	char    strInputFilePath[128];
	cv::Mat inputFrame[NUM_CAM];

	stParamsMain         params;
	stParamsDetection    paramsDetection;
	ParamsTracker2D    paramsTracker2D;
	ParamsAssociator3D paramsAssociator3D;

	if (!ReadParams(argv, params, paramsDetection, paramsTracker2D, paramsAssociator3D))
	{ 
		printf("[ERROR] cannot load parameters!!\n");
		return 0;
	}	
	
	for (int configIdx = 0; configIdx < params.sizeOfKs.size(); configIdx++)
	{		
		paramsAssociator3D.nKBestSize_ = params.sizeOfKs[configIdx];
		for (int expIdx = 0; expIdx < params.numExperiments; expIdx++)
		{
			// show experiment label
			cv::Mat expLabel = cv::Mat::zeros(40, 200, CV_8UC1);
			cv::putText(expLabel, 
				        "K:" + std::to_string(paramsAssociator3D.nKBestSize_) + "/" + std::to_string(expIdx),
						cv::Point(0, 30), 
						cv::FONT_HERSHEY_SIMPLEX, 
						1.0, 
						cv::Scalar(255, 255, 255));
			cv::imshow("experiment label", expLabel);
			cv::waitKey(1);

			/////////////////////////////////////////////////////////////////
			// INITIALIZATION
			/////////////////////////////////////////////////////////////////	
			CPSNWhere psnWhere = CPSNWhere();
			psnWhere.Initialize(params.strDatasetPath, &paramsAssociator3D);			
	
			/////////////////////////////////////////////////////////////////
			// MAIN LOOP
			/////////////////////////////////////////////////////////////////
			for (int frameIdx = params.startFrameIdx; frameIdx <= params.endFrameIdx; frameIdx++) 
			{
				//---------------------------------------------------
				// FRAME GRABBING
				//---------------------------------------------------		
				for (int camIdx = 0; camIdx < NUM_CAM; camIdx++) 
				{
					unsigned int curCamID = CAM_ID[camIdx];				
					if (PSN_INPUT_TYPE)
					{
						sprintf_s(strInputFilePath, sizeof(strInputFilePath), "%s\\View_%03d\\frame_%04d.jpg", params.strDatasetPath.c_str(), curCamID, frameIdx);						
					}
					else 
					{
						sprintf_s(strInputFilePath, sizeof(strInputFilePath), "%s\\%d_%d.jpg", params.strDatasetPath.c_str(), curCamID, frameIdx);
					}
					inputFrame[camIdx] = cv::imread(strInputFilePath, cv::IMREAD_COLOR);
					if (!inputFrame[camIdx].data) 
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
				for (int camIdx = 0; camIdx < NUM_CAM; camIdx++) { inputFrame[camIdx].release(); }
			}
	
			/////////////////////////////////////////////////////////////////
			// TERMINATION
			/////////////////////////////////////////////////////////////////
			psnWhere.Finalize();
		}
	}

	return 0;
}

bool ReadParams(_TCHAR** parameterPath, 
	stParamsMain         &outputParamsMain, 
	stParamsDetection    &outputParamsDetection, 
	stParamsTracker2D    &outputParamsTracker2D,
	stParamsAssociator3D &outputParamsAssociator3D)
{
	#ifdef _UNICODE
	std::wstring wstrDatasetPath(parameterPath[1]);
	std::string strDatasetPath(wstrDatasetPath.begin(), wstrDatasetPath.end());
#else
	std::string strDatasetPath = parameterPath[1];
#endif
	CParameterParser parser;
	PARAM_SET params;
	if (!parser.ReadParams(strDatasetPath.c_str(), params))
	{		
		return false;
	}

	// default values
	outputParamsAssociator3D.nProcWindowSize = -1;
	outputParamsAssociator3D.nKBestSize = -1;
	outputParamsAssociator3D.nMaxTrackInOptimization = -1;
	outputParamsAssociator3D.nMaxTrackInUnconfirmedTrackTree = -1;
	outputParamsAssociator3D.nNumFrameForConfirmation = 3;

	for (int paramIdx = 0; paramIdx < params.size(); paramIdx++)
	{		
		if (0 == params[paramIdx].first.compare("DATASET_PATH"))
		{ outputParamsMain.strDatasetPath = params[paramIdx].second; }

		else if (0 == params[paramIdx].first.compare("NUM_EXPERIMENTS"))
		{ outputParamsMain.numExperiments = std::stoi(params[paramIdx].second); }

		else if (0 == params[paramIdx].first.compare("START_FRAME_IDX"))
		{ outputParamsMain.startFrameIdx = std::stoi(params[paramIdx].second); }

		else if (0 == params[paramIdx].first.compare("END_FRAME_IDX"))
		{ outputParamsMain.endFrameIdx = std::stoi(params[paramIdx].second); }

		else if (0 == params[paramIdx].first.compare("SIZE_OF_KS"))
		{
			parser.ParseArray(params[paramIdx].second, outputParamsMain.sizeOfKs);
		}

		else if (0 == params[paramIdx].first.compare("NUM_FRAMES_FOR_CONFIRMATION"))
		{ outputParamsAssociator3D.nNumFrameForConfirmation = std::stoi(params[paramIdx].second); }
	}
	return true;
}

//()()
//('')HAANJU.YOO


