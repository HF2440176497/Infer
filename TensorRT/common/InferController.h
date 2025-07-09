
#pragma once

#include "common_include.h"
#include "monopoly_allocator.h"

template <class Input, class Output, class StartParam = std::tuple<std::string, int>>
class InferController {
public:
    struct Job {
        Input                                               input;
        Output                                              output;
        MonopolyAllocator<TRT::Tensor>::MonopolyDataPointer mono_tensor;
        std::shared_ptr<std::promise<Output>>               pro;
    };

    virtual ~InferController() { stop(); }

    void stop() {
        run_ = false;
        cond_.notify_all();

        {
            std::unique_lock<std::mutex> l(jobs_lock_);
            while (!jobs_.empty()) {
                auto& item = jobs_.front();
                if (item.pro) item.pro->set_value(Output());
                jobs_.pop();
            }
        };

        if (worker_) {
            worker_->join();
            worker_.reset();
        }
    }

    bool startup(const StartParam& param){
        run_ = true;
        std::promise<bool> pro;
        start_param_ = param;
        worker_      = std::make_shared<std::thread>(&InferController::worker, this, std::ref(pro));
        return pro.get_future().get();
    }

    virtual bool get_jobs_and_wait(std::vector<Job>& fetch_jobs, int max_size) {
        std::unique_lock<std::mutex> l(jobs_lock_);
        cond_.wait(l, [&]() { return !run_ || !jobs_.empty(); });

        if (!run_) return false;

        fetch_jobs.clear();

        // 为空或达到最大获取长度时返回
        for (int i = 0; i < max_size && !jobs_.empty(); ++i) {
            fetch_jobs.emplace_back(std::move(jobs_.front()));
            jobs_.pop();
        }
        return true;
    }

    virtual bool get_job_and_wait(Job& fetch_job) {
        std::unique_lock<std::mutex> l(jobs_lock_);
        cond_.wait(l, [&]() { return !run_ || !jobs_.empty(); });

        if (!run_) return false;

        fetch_job = std::move(jobs_.front());
        jobs_.pop();
        return true;
    }

protected:
    virtual void worker(std::promise<bool>& result) = 0;
    virtual bool preprocess(Job& job, const Input& input) = 0;

protected:
    StartParam start_param_;
    std::atomic<bool> run_;
    std::mutex jobs_lock_;
    std::queue<Job> jobs_;
    std::shared_ptr<std::thread> worker_;
    std::condition_variable cond_;
    std::shared_ptr<MonopolyAllocator<TRT::Tensor>> tensor_allocator_;

};  // class InferController