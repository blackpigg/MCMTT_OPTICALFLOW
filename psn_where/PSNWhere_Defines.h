#pragma once

/////////////////////////////////////////////////////////////////////////
// FLAGS
/////////////////////////////////////////////////////////////////////////

// recording
//#define DO_RECORD
//#define SHOW_TOPVIEW

// snapshot
//#define LOAD_SNAPSHOT_
//#define SAVE_SNAPSHOT_

// console loggin
#define PSN_DEBUG_MODE_
#define PSN_MONITOR_MODE_
#define PSN_PRINT_LOG_

/////////////////////////////////////////////////////////////////////////
// DATASET
/////////////////////////////////////////////////////////////////////////

// dataset base path
typedef enum { 
	DATASET_ETRI = 0,
	DATASET_PETS2009_S2L1,
	DATASET_PETS2009_S2L2,
	DATASET_PETS2009_S2L3,
	DATASET_NUM
} DATASET_TYPE;

const std::string DATASET_NAME[DATASET_NUM] = {
	"ETRI_LABS1",
	"PETS2009_S2L1",
	"PETS2009_S2L2",
	"PETS2009_S2L3"
};

const std::string DATASET_PATH[DATASET_NUM] = {
	"D:/Workspace/Dataset/PILSNU/ETRI_Lab.S1",			// ETRI
	"D:/Workspace/Dataset/PETS2009/S2/L1/Time_12-34",	// PETS2009 S2.L1
	"D:/Workspace/Dataset/PETS2009/S2/L2/Time_14-55",	// PETS2009 S2.L2
	"D:/Workspace/Dataset/PETS2009/S2/L3/Time_14-41"	// PETS2009 S2.L3
};

const int DATASET_START_FRAME_IDX[DATASET_NUM] = {0, 0, 3, 0};
//const int DATASET_END_FRAME_IDX[DATASET_NUM] = {332, 794, 432, 239};
const int DATASET_END_FRAME_IDX[DATASET_NUM] = {30, 794, 432, 239};
const int DATASET_NUM_CAM[DATASET_NUM] = {4, 3, 3, 3};
const int DATASET_CAM_ID[DATASET_NUM][4] = { // '-1' for dummy index
	{1, 2, 3, 4},
	{1, 5, 7, -1},
	{1, 2, 3, -1},
	{1, 2, 4, -1}
};

// other input paths
#define CALIBRATION_PATH ("/calibrationInfos/")
#define DETECTION_PATH ("/detectionResult/")
#define GROUNDTRUTH_PATH ("/groundTruth/cropped.txt")

/////////////////////////////////////////////////////////////////////////
// OUTPUT
/////////////////////////////////////////////////////////////////////////

// output path
#define RESULT_SAVE_PATH ("D:/Workspace/ExperimentalResult/THESIS/")


/////////////////////////////////////////////////////////////////////////
// INPUT PRESETS
/////////////////////////////////////////////////////////////////////////
#define PSN_INPUT_TYPE (2)	// 0:ETRI / 1:PETS2009 / 2:PILSNU
#define PSN_DETECTION_TYPE (1)	// 0:Head / 1:Full-body

#if 0 == PSN_INPUT_TYPE
	// ETRI Testbed setting
	define NUM_CAM (4)
	const unsigned int CAM_ID[NUM_CAM] = {0, 1, 2, 3};	
#else
	//// single camera
	//#define NUM_CAM (1)
	//const unsigned int CAM_ID[NUM_CAM] = {1};

	//// PETS.S2.L1 setting
	//#define NUM_CAM (3)
	//const unsigned int CAM_ID[NUM_CAM] = {1, 5, 7};

	//// PETS.S2.L2 setting
	//#define NUM_CAM (3)
	//const unsigned int CAM_ID[NUM_CAM] = {1, 2, 3};

	//// PETS.S2.L3 setting
	//#define NUM_CAM (3)
	//const unsigned int CAM_ID[NUM_CAM] = {1, 2, 4};

	// PILSNU ETRI_Lab.S1 setting
	#define NUM_CAM (4)
	const unsigned int CAM_ID[NUM_CAM] = {1, 2, 3, 4};
#endif

#define NUM_DETECTION_PART (8)
const std::string DETCTION_PART_NAME[NUM_DETECTION_PART] = {
	"HEAD", "F1", "S1", "GR", "S2", "A1", "A2", "F2"
};

/////////////////////////////////////////////////////////////////////////
// PREDEFINED VALUES
/////////////////////////////////////////////////////////////////////////
#define PSN_P_INF_SI (INT_MAX)
#define PSN_N_INF_SI (INT_MIN)
#define PSN_P_INF_F (FLT_MAX)
#define PSN_N_INF_F (FLT_MIN)
#define PSN_PI (3.1415926535897);

/////////////////////////////////////////////////////////////////////////
// VISUALIZATION SETTTING
/////////////////////////////////////////////////////////////////////////
#define DISP_TRAJECTORY3D_LENGTH (40)
#define DISPLAY_ID_MODE (1) // 0: raw track id, 1: id for visualization

/////////////////////////////////////////////////////////////////////////
// EVALUATION SETTING
/////////////////////////////////////////////////////////////////////////

const double CROP_ZONE[DATASET_NUM][5] = {
//  {X_MIN,    X_MAX,  Y_MIN,    Y_MAX,  MARGIN}
	{-3000.0,  4000.0, -5000.0,  2000.0, 500.0},
	{-14069.6, 4981.3, -14274.0, 1733.5, 1000.0},
	{-14069.6, 4981.3, -14274.0, 1733.5, 1000.0},
	{-14069.6, 4981.3, -14274.0, 1733.5, 1000.0},
};

