
#pragma once

#include "common_include.h"
#include "monopoly_allocator.h"

template <class Input, class Output, class StartParam>
class InferController {
public:
    struct Job {
        Input                                               input;
        Output                                              output;
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
            }
            // preproces complete
            {
                std::unique_lock<std::mutex> l(jobs_lock_);
                for (int i = begin; i < end; ++i) {
                    jobs_.emplace(std::move(jobs[i]));
                };
            }
            cond_.notify_one();
            for (int i = begin; i < end; ++i) {
                Job& job = jobs[i];
                results[i] = job.pro->get_future();  // 此时的 job 是 jobs_ 中的元素
            }
        }  // for (int epoch = 0; epoch < nepoch; ++epoch)
        return results;
    }

    virtual std::vector<std::shared_future<Output>> commits_serial(const std::vector<Input>& inputs) {
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
                }
                job.pro->set_value(Output());  // TODO: 验证预处理阶段，只需要空结果
                // INFO("Current preprocess index [%d] complete", i);
            }
            
            // TODO: 串行方案下，job 不需要转移到 jobs_，因此在当前作用域内继续使用 job

            // 开始推理
            for (int i = begin; i < end; ++i) {
                Job& job = jobs[i];
                job.mono_tensor->release();  // TODO: 异步形式 应当在 worker 进行
                results[i] = job.pro->get_future();
            }
        }  // for (int epoch = 0; epoch < nepoch; ++epoch)
        return results;
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
    virtual bool preprocess(Job& job, const Input& input) = 0;
    virtual void worker(std::promise<bool>& result) = 0;
    virtual bool worker_serial() = 0;
    virtual void adjust_mem() = 0;

protected:
    StartParam start_param_;
    std::atomic<bool> run_;
    std::mutex jobs_lock_;
    std::queue<Job> jobs_;
    std::shared_ptr<std::thread> worker_;
    std::condition_variable cond_;
    std::shared_ptr<MonopolyAllocator<TRT::Tensor>> tensor_allocator_;

};  // class InferController