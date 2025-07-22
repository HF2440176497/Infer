
#pragma once

#include "common_include.h"
#include "monopoly_allocator.h"
#include "utils/processor.cuh"

template <class Input, class Output, class StartParam>
class InferController {
public:
    struct Job {
        Input                                               input;
        Output                                              output;
        std::shared_ptr<utils::AffineTrans>                 trans;
        MonopolyAllocator<TRT::Tensor>::MonopolyDataPointer mono_tensor;  // 保存预处理完成的结果
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

    bool startup(const StartParam& param) {
        run_ = true;
        std::promise<bool> pro;
        start_param_ = param;  // start_param_ 类型是模板参数，在派生类中才会使用
        return worker_serial();
        // worker_ = std::make_shared<std::thread>(&InferController::worker, this, std::ref(pro));
        // return pro.get_future().get();
    }

    virtual std::vector<std::shared_future<Output>> commits(const std::vector<Input>& inputs) {
        int              batch_size = std::min((int)inputs.size(), this->tensor_allocator_->capacity());
        std::vector<Job> jobs(inputs.size());
        std::vector<std::shared_future<Output>> results(inputs.size());  // 异步等待结果
        adjust_mem();

        int nepoch = (inputs.size() + batch_size - 1) / batch_size;
        for (int epoch = 0; epoch < nepoch; ++epoch) {
            int begin = epoch * batch_size;
            int end = std::min((int)inputs.size(), begin + batch_size);  // 不要超出范围

            for (int i = begin; i < end; ++i) {
                Job& job = jobs[i];
                job.pro = std::make_shared<std::promise<Output>>();
                if (!preprocess(job, inputs[i])) {
                    INFO("preprocess error happened");
                    job.pro->set_value(Output());  // 预处理失败时，设置结果但是为空
                }
                results[i] = job.pro->get_future();
            }
            // preproces complete
            {
                std::unique_lock<std::mutex> l(jobs_lock_);
                for (int i = begin; i < end; ++i) {
                    jobs_.emplace(std::move(jobs[i]));
                };
            }
            cond_.notify_one();
        }  // for (int epoch = 0; epoch < nepoch; ++epoch)
        return results;
    }

    virtual std::vector<std::shared_future<Output>> commits_serial(const std::vector<Input>& inputs) {
        int              batch_size = std::min((int)inputs.size(), this->tensor_allocator_->capacity());  // 进行预处理的 batch
        std::vector<Job> jobs(inputs.size());
        std::vector<std::shared_future<Output>> results(inputs.size());  // 异步等待结果
        adjust_mem();

        int nepoch = (inputs.size() + batch_size - 1) / batch_size;

        INFO("WARNING: nepoch [%d]", nepoch);

        for (int epoch = 0; epoch < nepoch; ++epoch) {
            int begin = epoch * batch_size;
            int end = std::min((int)inputs.size(), begin + batch_size);  // 不要超出范围

            for (int i = begin; i < end; ++i) {
                Job& job = jobs[i];
                job.pro = std::make_shared<std::promise<Output>>();
                if (!preprocess(job, inputs[i])) {
                    INFO("preprocess error happened");
                    job.pro->set_value(Output());
                }
                results[i] = job.pro->get_future();
            }
        
            // TODO: 同步流程，不进行并发保护
            for (int i = begin; i < end; ++i) {
                jobs_.emplace(std::move(jobs[i]));
            }
            this->inference_handle();
        }  // for (int epoch = 0; epoch < nepoch; ++epoch)
        return results;
    }
    // TODO: 串行试验阶段

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
    virtual void adjust_mem() = 0;
    virtual bool preprocess(Job& job, const Input& input) = 0;
    virtual void worker(std::promise<bool>& result) = 0;
    virtual bool worker_serial() = 0;
    virtual void inference_handle() = 0;
    

protected:
    StartParam start_param_;
    std::atomic<bool> run_;
    std::mutex jobs_lock_;
    std::queue<Job> jobs_;
    std::shared_ptr<std::thread> worker_;
    std::condition_variable cond_;
    std::shared_ptr<MonopolyAllocator<TRT::Tensor>> tensor_allocator_;

};  // class InferController
