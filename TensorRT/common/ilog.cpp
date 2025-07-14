
#include "ilog.h"



namespace iLog {

    std::string format(const char* fmt, ...) {
        va_list vl;
        va_start(vl, fmt);
        char buffer[2048];
        vsnprintf(buffer, sizeof(buffer), fmt, vl);
        return buffer;
    }

}