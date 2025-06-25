
#pragma once

#include "common_include.h"


#define Assert(op)					 \
	do{                              \
		bool cond = !(!(op));        \
		if(!cond){                   \
			INFOF("Assert failed, " #op);  \
		}                                  \
	}while(false)


namespace CUDATools {

	bool check_device_id(int device_id);
    int current_device_id();
    std::string device_capability(int device_id);
    std::string device_name(int device_id);
    std::string device_description();

};