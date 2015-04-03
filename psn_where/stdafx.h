// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>
#include <afx.h>

#ifndef _UNICODE
    #ifdef _DEBUG
        #pragma comment(lib, "nafxcwd.lib")
    #else
        #pragma comment(lib, "nafxcw.lib")
    #endif
#else
    #ifdef _DEBUG
        #pragma comment(lib, "uafxcwd.lib")
    #else
        #pragma comment(lib, "uafxcw.lib")
    #endif
#endif

