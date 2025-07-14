//
// Created by behrooz on 24/06/22.
//

#ifndef PARTH_RINGBUFFER_H
#define PARTH_RINGBUFFER_H

#include <assert.h>
#include <iostream>
#include <vector>

///---------------------------------------------------------------------------------------\n
/// RingStorage - This class is a circular buffer\n
/// NOTE: DO NOT USE THE ITERATOR WITH std::fill and these stuff it is only good
/// for for iterator\n Use cases:\n for(auto iter = RS.begin(); iter !=
/// RS.end(); iter++){for-body}\n for(auto i = RS.getStartIdx(); i !=
/// RS.getEndIdx(); i = RS.getNextIdx()){for-body}\n
///---------------------------------------------------------------------------------------\n
template <class T>
class RingStorage {
public:
    RingStorage(int max_size = 1)
    {
        assert(max_size > 0);
        if (max_size < 1) {
            throw std::runtime_error(
                "The size of the storage should be more than 0\n");
        }
        head_ = 0;
        tail_ = 0;
        max_size_ = max_size + 1;
        storage_.resize(max_size_);
    }

    struct Iterator {
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using pointer = T*;
        using reference = T&;

        Iterator(pointer ptr, pointer start, pointer end, bool isfull)
        {
            this->ptr_ = ptr;
            this->start_ptr_ = start;
            this->end_ptr_ = end;
            this->one_round_happened_ = !isfull;
        }

        // Copy constructor
        Iterator(const Iterator& iter)
        {
            this->ptr_ = iter.ptr_;
            this->start_ptr_ = iter.start_ptr_;
            this->end_ptr_ = iter.end_ptr_;
            this->one_round_happened_ = !iter.isfull;
        }

        reference operator*() const { return *ptr_; }
        pointer operator->() { return ptr_; }
        Iterator& operator++()
        {
            if (ptr_ == end_ptr_) {
                ptr_ = start_ptr_;
            }
            else {
                ptr_++;
            }
        }
        friend bool operator==(const Iterator& a, const Iterator& b)
        {
            if (a.m_ptr == b.m_ptr) {
                if (a.one_round_happened) {
                    return true;
                }
                else {
                    a.one_round_happened = true;
                    return false;
                }
            }
            return false;
        };
        friend bool operator!=(const Iterator& a, const Iterator& b)
        {
            if (a == b) {
                return false;
            }
            return true;
        };

    private:
        pointer ptr_;
        pointer start_ptr_;
        pointer end_ptr_;
        bool one_round_happened_ = false;
    };

    Iterator begin()
    {
        assert(!isEmpty());
        return Iterator(&storage_[tail_], storage_.front(), storage_.back(),
            isFull());
    }

    Iterator end()
    {
        assert(!isEmpty());
        return Iterator(&storage_[head_], storage_.front(), storage_.back(),
            isFull());
    }

    // Properties extraction functions
    bool isEmpty() const { return (head_ == tail_); }
    bool isFull() const { return (((head_ + 1) % max_size_) == tail_); }
    int size()
    {
        if (!isFull()) {
            if (head_ >= tail_) {
                return head_ - tail_;
            }
            else {
                return max_size_ + head_ - tail_;
            }
        }
        else {
            return max_size_ - 1;
        }
    }
    size_t capacity() { return max_size_ - 1; }

    // working with the buffer functions
    void reset(size_t max_size)
    {
        assert(max_size > 0);
        if (max_size < 1) {
            throw std::runtime_error(
                "The size of the storage should be more than 0\n");
        }
        head_ = 0;
        tail_ = 0;
        max_size_ = max_size + 1;
        storage_.resize(max_size + 1);
    }

    void advancePtr()
    {
        head_ = (head_ + 1) % (max_size_);
        if (head_ == tail_) {
            tail_ = (tail_ + 1) % (max_size_);
        }
    }

    void retreatPtr() { tail_ = (tail_ + 1) % (max_size_); }

    void addElementRef(T& elem)
    {
        storage_[head_] = elem;
        advancePtr();
    }

    void addElement(T elem)
    {
        storage_[head_] = elem;
        advancePtr();
    }

    bool removeOldest(T& elem)
    {
        if (isEmpty()) {
            throw std::runtime_error("removeOldest - The storage is empty\n");
        }
        elem = storage_[tail_];
        retreatPtr();
        return true;
    }

    int getOldestElementIdx() const
    {
        if (isEmpty()) {
            throw std::runtime_error("getNewestElementIdx - The storage is empty\n");
        }
        return tail_;
    }

    int getNewestElementIdx() const
    {
        if (isEmpty()) {
            throw std::runtime_error("getNewestElementIdx - The storage is empty\n");
        }
        if (head_ != 0) {
            return head_ - 1;
        }
        else {
            return max_size_ - 1;
        }
    }

    ///\description Return the newest element by reference
    T& getNewest() { return getElement(getNewestElementIdx()); }

    ///\description Return the oldest element by reference
    T& getOldest() { return getElement(getOldestElementIdx()); }

    int getStartIdx() const { return tail_; }

    int getEndIdx() const { return head_; }

    int getNextIdx(int i) const
    {
        if (i + 1 == max_size_) {
            return 0;
        }
        else {
            return i + 1;
        }
    }

    int getPrevIdx(int i) const
    {
        if (isEmpty()) {
            throw std::runtime_error("getPrevIdx - The storage is empty\n");
        }
        if (i != 0) {
            return i - 1;
        }
        else {
            return max_size_ - 1;
        }
    }

    T& getElement(size_t idx)
    {
        if (isEmpty()) {
            throw std::runtime_error("getElement - The storage is empty\n");
        }
        return storage_[idx];
    }

private:
    std::vector<T> storage_; ///< @brief Storage
    int max_size_; ///< @brief Maximum number of records in the storage
    int head_; /// <@brief The index of the oldest element in the storage
    int tail_; /// <@brief The index of the newest element in the storage
};

#endif // IPC_RINGBUFFER_H
