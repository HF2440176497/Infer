
#pragma once

#include "utils/common_include.h"

#if defined(_WIN32)
#	define U_OS_WINDOWS
#else
#   define U_OS_LINUX
#endif


namespace iLog {

    std::string format(const char* fmt, ...);


}