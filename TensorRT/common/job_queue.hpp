
#pragma once

#include <queue>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <stdexcept>
#include <iostream>

template <typename Job>
class JobQueue {
public:
    using SizeCallback = std::function<void(size_t)>;
    using ClearCallback = std::function<void(Job&)>;

    explicit JobQueue(size_t max_size = 0, size_t warn_size = 0, SizeCallback warn_callback = nullptr, ClearCallback clear_callback = nullptr)
        : max_size_(max_size), warn_size_(warn_size), warn_callback_(warn_callback), clear_callback_(clear_callback), run_(true) {
        if (max_size_ < 0 || warn_size_ < 0) {
            throw std::invalid_argument("size cannot be minus");
        }
        if (warn_size_ > max_size_ && max_size_ > 0) {
            throw std::invalid_argument("warn_size cannot exceed max_size");
        }
    }
    JobQueue(const JobQueue&) = delete;
    JobQueue& operator=(const JobQueue&) = delete;
    JobQueue(JobQueue&&) = default;
    JobQueue& operator=(JobQueue&&) = default;

public:
    bool get_jobs_and_wait(std::vector<Job>& fetch_jobs, size_t max_size) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this] { return !run_ || !queue_.empty(); });
        
        if (!run_) return false;
        
        fetch_jobs.clear();
        fetch_jobs.reserve(std::min(max_size, queue_.size()));
        
        while (!queue_.empty() && fetch_jobs.size() < max_size) {
            fetch_jobs.emplace_back(std::move(queue_.front()));
            queue_.pop();
        }
        if (max_size_ > 0) {  // 如果设置有最大空间限制
            lock.unlock();
            cond_space_.notify_one();
        } else {
            lock.unlock();
        }
        return true;
    }

    /**
     * @brief 添加单个任务 阻塞等待
     * @param job 任务对象
     */
    void emplace(Job&& job) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (max_size_ > 0) {
            cond_space_.wait(lock, [this] {
                return !run_ || queue_.size() < max_size_;
            });
            if (!run_) return;
        }
        queue_.push(std::move(job));
        check_queue_length();
        lock.unlock();
        cond_.notify_one();
    }

    template <typename InputIt>
    void batch_emplace(InputIt begin, InputIt end) {
        if (begin == end) return;
        
        std::unique_lock<std::mutex> lock(mutex_);
        const size_t add_count = std::distance(begin, end);
        
        // 等待队列有足够空间
        if (max_size_ > 0) {
            cond_space_.wait(lock, [this, add_count] {
                return !run_ || (queue_.size() + add_count <= max_size_);
            });
            if (!run_) return;
        }
        for (auto it = begin; it != end; ++it) {
            queue_.push(std::move(*it));
        }
        check_queue_length();
        lock.unlock();
        cond_.notify_one();  // 通知消费者
    }

    /**
     * @brief 清空队列并处理未完成的任务
     */
    void clear() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!queue_.empty()) {
            auto& item = queue_.front();
            if (clear_callback_) {
                clear_callback_(item);
            }
            queue_.pop();
        }
        
        if (max_size_ > 0) {
            lock.unlock();
            cond_space_.notify_all();  // 释放所有等待的生产者
        }
    }

    /**
     * @brief 停止队列操作（唤醒所有等待线程）
     */
    void stop() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            run_ = false;
        }
        cond_.notify_all();
        cond_space_.notify_all();
    }

    /**
     * @brief 设置清除回调函数
     */
    void set_clear_callback(ClearCallback callback) {
        std::unique_lock<std::mutex> lock(mutex_);
        clear_callback_ = callback;
    }

    /**
     * @brief 设置警告回调函数
     */
    void set_warn_callback(SizeCallback callback) {
        std::unique_lock<std::mutex> lock(mutex_);
        warn_callback_ = callback;
    }
private:
    void check_queue_length() {
        if (warn_callback_ && warn_size_ > 0 && 
            queue_.size() >= warn_size_ && 
            !warn_triggered_) 
        {
            warn_triggered_ = true;
            warn_callback_(queue_.size());
        } else if (queue_.size() < warn_size_) {  // 回落到阈值以下，允许再次触发
            warn_triggered_ = false;
        }
    }

private:
    mutable std::mutex      mutex_;
    std::condition_variable cond_;        // 通知消费者
    std::condition_variable cond_space_;  // 通知生产者
    std::queue<Job>         queue_;
    bool                    run_;
    size_t                  max_size_;        // 0 = unlimited
    size_t                  warn_size_;       // 0 = no warning
    bool                    warn_triggered_ = false;
    SizeCallback            warn_callback_;   // 长度预警回调
    ClearCallback           clear_callback_;  // 清除任务回调
};