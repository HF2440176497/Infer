

#include <filesystem>
#include <system_error>
#include <fstream>
#include <system_error>

#include "ilog.h"


namespace fs = std::filesystem;

namespace iLog {

    std::string format(const char* fmt, ...) {
        va_list vl;
        va_start(vl, fmt);
        char buffer[2048];
        vsnprintf(buffer, sizeof(buffer), fmt, vl);
        return buffer;
    }

    bool mkdirs(const std::string& path) {
        std::error_code ec;
        bool created = fs::create_directory(path, ec);
        return !ec && created;
    }

    bool exists(const std::string& path) {
        std::error_code ec;
        return fs::exists(path, ec) && !ec;
    }

    bool save_file(const std::string& file, const void* data, size_t length, bool mk_dirs) {
        if (mk_dirs) {
            std::error_code ec;
            fs::path dir_path = fs::path(file).parent_path();
            if (!dir_path.empty() && !fs::exists(dir_path, ec)) {
                bool status = fs::create_directories(dir_path, ec);
                if (!status) {
                    return false;
                }
            }
        }
        std::ofstream out(file, std::ios::binary);
        if (!out.is_open()) {
            return false;
        }
        if (data && length > 0) {
            out.write(static_cast<const char*>(data), length);
            if (!out.good()) {  // 检查写入是否成功
                out.close();
                return false;
            }
        }
        out.close();
        return true;
    }
}