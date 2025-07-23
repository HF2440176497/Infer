
#pragma once

#include "common_include.h"
#include "monopoly_allocator.h"
#include "utils/processor.cuh"
#include "common/job_queue.hpp"


template <class Input, class Output, class StartParam>
class InferController {
public:
    using IutputType = Input;
    using OutputType = Output;
    using StartParamType = StartParam;
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
        jobs_queue_->stop();  // 先解除可能存在的队列阻塞
        if (worker_) {
            worker_->join();
            worker_.reset();
        }
        jobs_queue_ = nullptr;  // 待线程停止后可置空
    }

    bool startup(const StartParam& param) {
        run_ = true;
        std::promise<bool> pro;
        start_param_ = param;  // start_param_ 类型是模板参数，在派生类中才会使用
        worker_ = std::make_shared<std::thread>(&InferController::worker, this, std::ref(pro));
        return pro.get_future().get();
    }

    virtual std::vector<std::shared_future<Output>> commits(const std::vector<Input>& inputs) {
        int              batch_size = std::min((int)inputs.size(), this->tensor_allocator_->capacity());
        std::vector<Job> jobs(inputs.size());
        std::vector<std::shared_future<Output>> results(inputs.size());  // 异步等待结果

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
            auto begin_it = jobs.begin() + begin;
            auto end_it = jobs.begin() + end;
            jobs_queue_->batch_emplace(begin_it, end_it);
        }  // for (int epoch = 0; epoch < nepoch; ++epoch)
        return results;
    }

protected:
    virtual bool preprocess(Job& job, const Input& input) = 0;
    virtual void worker(std::promise<bool>& result) = 0;

protected:
    StartParam start_param_;
    std::atomic<bool> run_;
    std::shared_ptr<std::thread> worker_;
    std::shared_ptr<MonopolyAllocator<TRT::Tensor>> tensor_allocator_;
    std::unique_ptr<JobQueue<Job>> jobs_queue_;
};  // class InferController
