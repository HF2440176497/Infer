
#pragma once

#include "utils/common_include.h"

#if defined(_WIN32)
#	define U_OS_WINDOWS
#else
#   define U_OS_LINUX
#endif


namespace iLog {

    std::string format(const char* fmt, ...);
    bool mkdirs(const std::string& path);
    bool exists(const std::string& path);
    bool save_file(const std::string& file, const void* data, size_t length, bool mk_dirs = true);
}
