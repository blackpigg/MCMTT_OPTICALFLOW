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
// PATH
/////////////////////////////////////////////////////////////////////////
// input
//#define DATASET_PATH ("D:\\Workspace\\Dataset\\ETRI\\P5\\")
#define CALIBRATION_PATH ("/calibrationInfos/")
#define DETECTION_PATH ("/detectionResult/")
#define TRACKLET_PATH ("/trackletInput/")
// output
#define RESULT_SAVE_PATH ("D:/Workspace/ExperimentalResult/PETS2009/")
#define SNAPSHOT_PATH ("logs/snapshot/")
#define TRACK_SAVE_PATH ("logs/tracks/")

/////////////////////////////////////////////////////////////////////////
// INPUT PRESETS
/////////////////////////////////////////////////////////////////////////
#define PSN_INPUT_TYPE (1)	// 0:ETRI / 1:PETS2009
#define PSN_DETECTION_TYPE (1)	// 0:Head / 1:Full-body

#if 0 == PSN_INPUT_TYPE
	// ETRI Testbed setting
	define NUM_CAM (4)
	const unsigned int CAM_ID[NUM_CAM] = {0, 1, 2, 3};	
#else
	//// PETS.S2.L1 setting
	//#define NUM_CAM (3)
	//const unsigned int CAM_ID[NUM_CAM] = {1, 5, 7};

	// PETS.S2.L2 setting
	//#define NUM_CAM (3)
	//const unsigned int CAM_ID[NUM_CAM] = {1, 2, 3};

	//// PETS.S2.L3 setting
	//#define NUM_CAM (3)
	//const unsigned int CAM_ID[NUM_CAM] = {1, 2, 4};

	// PETS.S2.L3 setting
	#define NUM_CAM (1)
	const unsigned int CAM_ID[NUM_CAM] = {1};
#endif

#define NUM_DETECTION_PART (8)
const std::string DETCTION_PART_NAME[NUM_DETECTION_PART] = {"HEAD", "F1", "S1", "GR", "S2", "A1", "A2", "F2"};

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
#define CROP_ZONE_X_MIN (-14069.6)
#define CROP_ZONE_X_MAX (4981.3)
#define CROP_ZONE_Y_MIN (-14274.0)
#define CROP_ZONE_Y_MAX (1733.5)
#define CROP_ZONE_MARGIN (1000.0)
