//
// Created by behrooz on 2021-03-21.
//

#ifndef LBC_LIB_PAPI_UTILS_H
#define LBC_LIB_PAPI_UTILS_H


#include <papi.h>
#include "csv_utils.h"
#include <string>
#include <vector>
#include <Utils.h>
#include <sparse_utilities.h>
#include <test_utils.h>
#include <utils.h>
#define ERROR_RETURN(retval) { fprintf(stderr, "Error %d %s:line %d: \n", retval,__FILE__,__LINE__);  exit(retval); }
using namespace sym_lib;
namespace profiling_utils{
    unsigned long omp_get_thread_num_wrapper(){
        return (unsigned long)omp_get_thread_num();
    }

    #define num_tests_ 10

    struct PAPIEVENT{
        int code;
        std::string name;
        std::vector<long long> counters_values;
        long long  median_value;
    };

    class PAPISerialProfiler
    {
    private:
        int event_set{};
        int num_events;
        std::vector<PAPIEVENT> events;
        std::vector<long long> current_counter_value;
    public:
        PAPISerialProfiler(std::vector<int> events_code, std::vector<std::string>& events_name_csv_header, bool init_header){
            num_events = events_code.size();
            this->events.resize(num_events);
            for(int e = 0; e < num_events; e++){
                this->events[e].code = events_code[e];
                char EventCodeStr[PAPI_MAX_STR_LEN];
                PAPI_event_code_to_name(this->events[e].code, EventCodeStr);
                std::string ret(EventCodeStr);
                this->events[e].name = ret;
                //Init the name in the CSV header
                if(init_header){
                    events_name_csv_header.push_back(this->events[e].name);
                }
                current_counter_value.resize(1);
                this->events[e].median_value = 0;
            }
        }

        //Return the number of events that we need to count
        int getNumEvents(){return num_events;}
        //Return the list of all events
        std::vector<PAPIEVENT>& getEvents(){ return events; }

        //Start measuring current event value
        void startProfiling(int code){
            event_set = PAPI_NULL;
            int retval;
            if((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT) {
                ERROR_RETURN(retval)
                return;
            }
            if((retval = PAPI_create_eventset(&event_set)) != PAPI_OK) {
                ERROR_RETURN(retval)
                return;
            }
            if((retval = PAPI_add_event(event_set, code)) != PAPI_OK) {
                ERROR_RETURN(retval)
                return;
            }

            if((retval = PAPI_reset(event_set)) != PAPI_OK){
                ERROR_RETURN(retval)
            }

            if((retval = PAPI_start(event_set)) != PAPI_OK){
                ERROR_RETURN(retval)
            }

        }

        void endProfiling(int code){
            int retval=0;

            if((retval = PAPI_stop(event_set, current_counter_value.data())) != PAPI_OK){
                ERROR_RETURN(retval)
            }

            for(auto& e: events){
                if(e.counters_values.size() > num_tests_){
                    std::cout << "Please clear the papi counters before starting this experiment - or make sure that the number of tests are equal"  << std::endl;
                }
                if(e.code == code){
                    e.counters_values.push_back(current_counter_value.front());
                }
            }


            if ( (retval=PAPI_remove_event(event_set, code))!=PAPI_OK){
                ERROR_RETURN(retval)
            }

            PAPI_shutdown();
        }

        void computeMedian(){
            for(auto& e: events) {
                std::sort(e.counters_values.begin(), e.counters_values.end());
                if (e.counters_values.size() == 1) {
                    e.median_value = e.counters_values.front();
                } else if (e.counters_values.size() > 1) {
                    e.median_value = e.counters_values[(e.counters_values.size() / 2)];
                } else {
                    e.median_value = 0;
                }
            }
        }
        //Add the median off values to the csv header
        void addDataToCSV(CSVManager& csv_file, bool add_counter_value){
            //get Median
            if(events.empty()){
                std::cout << "There are no events to record" << std::endl;
                return;
            }
            for(auto& e: events){
                std::sort(e.counters_values.begin(), e.counters_values.end());
                if(e.counters_values.size() == 1){
                    e.median_value = e.counters_values.front();
                } else if(e.counters_values.size() > 1){
                    e.median_value = e.counters_values[(e.counters_values.size() / 2)];
                } else {
                    e.median_value = 0;
                }
                auto headers = csv_file.getHeaderNames();
                if(std::find(headers.begin(), headers.end(), e.name) == headers.end()){
                    std::cout<< "The event\t" << e.name << "\t is not in the headers" << std::endl;
                    continue;
                }
                if(add_counter_value){
                    csv_file.addElementToRecord(e.median_value, e.name);
                } else {
                    csv_file.addElementToRecord(-1, e.name);
                }
            }
        }

        void resetPAPICounters(){
            for(auto& e: events){
                e.median_value = 0;
                e.counters_values.clear();
            }
        }

        ~PAPISerialProfiler()=default;
    };

    /*
    * @brief This class is used for parallel profiling per thread
    * Usage example
    * prepareThreadProfiling()
    * for(auto& code: code_set){
    *     omp parallel{
    *         int event_set = PAPI_NULL;
    *         startProfilingThread(code, event_set)
    *         omp pragma for{
    *             some code
    *         }
    *         endProfilingThread(code, event_set, omp_get_thread_num())
    *     }
    * finishThreadProfiling() -> omit this function for now
    *
    *
    * later
    * addDataToCSVThread(csv_file)
    * resetThreadPAPICounters()
    *
    * }
    */
    class PAPIParallelProfiler
    {
    private:
        int num_events;
        std::vector<std::vector<PAPIEVENT>> threads_events;
        std::vector<long long> current_counter_value;
        std::vector<int> event_codes;
        int num_threads;
    public:
        /*
         * @brief this constructor can be called in the testing area (before building the kernel object)
         * and it should be passed to the FusionDemoCustom class
         * @param events_code a vector of the PAPI counters
         * @param num_threads is the number of threads that the code wants to run with
         * @param events_name_csv_header is the header that will be used in a csv object.
         * @param init_header if it is false, it does not initialize the header
         *
         * class usage example
         *
         */
        PAPIParallelProfiler(std::vector<int> events_code, int num_threads, std::vector<std::string>& events_name_csv_header, bool init_header=true){
            num_events = events_code.size();
            this->num_threads = num_threads;
            this->threads_events.resize(num_threads);
            this->current_counter_value.resize(num_threads, 0);
            this->event_codes = events_code;
            for(auto& events: this->threads_events){
                events.resize(num_events);
            }

            int cnt = 0;
            for(auto& events: this->threads_events){
                cnt++;
                for(int e = 0; e < num_events; e++){
                    events[e].code = events_code[e];
                    char EventCodeStr[PAPI_MAX_STR_LEN];
                    PAPI_event_code_to_name(events[e].code, EventCodeStr);
                    std::string ret(EventCodeStr);
                    events[e].name = ret;
                    //Init the name in the CSV header
                    events[e].median_value = 0;
                    if(init_header){
                        std::string thread_event_name;
                        thread_event_name = "Thread" + std::to_string(cnt) + "_" + events[e].name;
                        events_name_csv_header.push_back(thread_event_name);
                    }
                }
            }
        }

        /*
         * @brief Return the number of events that we need to count
         */
        int getNumEvents() const{return num_events;}
        /*
         * @brief Return the list of all records for specific thread
         */
        std::vector<PAPIEVENT>& getEventRecord(int thread_id){ return threads_events[thread_id]; }
        /*
         * @brief return the initialized codes
         */
        std::vector<int>& getCodes(){return event_codes;}

        /*
         * @brief this function should be initialized outside Parallel region
         * and before any other call to PAPI. It can be placed in the end of the build_set function
         * if you want to shutdown the PAPI using FinishThreadProfiling, it should be called again for next
         * profiling
         */
        static void prepareThreadProfiling(){
            //Init PAPI Threads
            if( PAPI_is_initialized() == PAPI_NOT_INITED ) {
                std::cout << "*** Initializing the PAPI Library ***" << std::endl;
                // Initialize PAPI library for each thread.
                int retval = PAPI_library_init( PAPI_VER_CURRENT );
                if ( retval != PAPI_VER_CURRENT ) {
                    ERROR_RETURN(retval)
                }

                retval = PAPI_thread_init(omp_get_thread_num_wrapper);
                if ( retval != PAPI_OK ) {
                    if ( retval == PAPI_ECMP ) {
                        ERROR_RETURN(retval)
                    }
                    else {
                        ERROR_RETURN(retval)
                    }
                }
            }
        }
        /*
         * @brief shutdown the PAPI counters
         */
        static void finishThreadProfiling(){
            PAPI_shutdown();
        }


        ///\Description



        /*
        * @brief Start measuring current code value
        * @param code is the event code
         * event_set is for managing the event
        */
        void startProfilingThread(int code, int &event_set){
            event_set = PAPI_NULL;
            int retval;
            PAPI_register_thread();

            if((retval = PAPI_create_eventset(&event_set)) != PAPI_OK) {
                ERROR_RETURN(retval)
                return;
            }
            if((retval = PAPI_add_event(event_set, code)) != PAPI_OK) {
                std::cout << "*** for code 25 it seems that you didn't reduce the paranoid flags ***" << std::endl;
                ERROR_RETURN(retval)
                return;
            }

            if((retval = PAPI_reset(event_set)) != PAPI_OK){
                ERROR_RETURN(retval)
            }

            if((retval = PAPI_start(event_set)) != PAPI_OK){
                ERROR_RETURN(retval)
            }

        }

        /*
         * @brief add the record to the thread counter recorders and reset the counters
         */
        void endProfilingThread(int code, int &event_set, int thread_id){
            int retval=0;
            current_counter_value[thread_id] = 0;
            if((retval = PAPI_stop(event_set, &current_counter_value[thread_id])) != PAPI_OK){
                ERROR_RETURN(retval)
            }

            for(auto& e: threads_events[thread_id]){
                if(e.counters_values.size() > num_tests_){
                    std::cout << "Please clear the papi counters before starting this experiment" << std::endl;
                }
                if(e.code == code){
                    e.counters_values.push_back(current_counter_value[thread_id]);
                }
            }

            if ( (retval=PAPI_remove_event(event_set, code))!=PAPI_OK){
                ERROR_RETURN(retval)
            }

            if ( (retval=PAPI_destroy_eventset( &event_set ))!=PAPI_OK){
                ERROR_RETURN(retval)
            }

            PAPI_unregister_thread();
        }

        /*
         * @brief add the record to the thread counter recorders and reset the counters
         * @param test_number: The current test id, it is one based so the first test is equal to 1
         */
        void endProfilingThread(int code, int test_number, int &event_set, int thread_id){
            int retval=0;
            current_counter_value[thread_id] = 0;
            if((retval = PAPI_stop(event_set, &current_counter_value[thread_id])) != PAPI_OK){
                ERROR_RETURN(retval)
            }

            for(auto& e: threads_events[thread_id]){
                if(e.counters_values.size() > num_tests_){
                    std::cout << "Please clear the papi counters before starting this experiment -"
                                 " Or the test_number is insane: " << e.counters_values.size() << std::endl;
                }
                if(e.code == code){
                    if(e.counters_values.size() < test_number){
                        e.counters_values.push_back(current_counter_value[thread_id]);
                    } else if(e.counters_values.size() == test_number){
                        e.counters_values[test_number - 1] += (current_counter_value[thread_id]);
                    } else {
                        std::cerr <<"Your test_number is wrong probably" << std::endl;
                    }
                }
            }

            if ( (retval=PAPI_remove_event(event_set, code))!=PAPI_OK){
                ERROR_RETURN(retval)
            }

            if ( (retval=PAPI_destroy_eventset( &event_set ))!=PAPI_OK){
                ERROR_RETURN(retval)
            }

            PAPI_unregister_thread();
        }


        //Add the median off values to the csv header
        void addDataToCSVThread(CSVManager& csv_file, bool add_counter_value=true){
            //get Median
            int cnt = 0;
            for(auto& events: threads_events){
                if(events.empty()){
                    std::cout << "There are no events to record" << std::endl;
                    return;
                }
                cnt++;
                for(auto& e: events){
                    std::sort(e.counters_values.begin(), e.counters_values.end());
                    if(e.counters_values.size() == 1){
                        e.median_value = e.counters_values.front();
                    } else if(e.counters_values.size() > 1){
                        e.median_value = e.counters_values[(e.counters_values.size() / 2)];
                    } else {
                        e.median_value = 0;
                    }
                    auto headers = csv_file.getHeaderNames();
                    std::string name = "Thread" + std::to_string(cnt) + "_" + e.name;
                    if(std::find(headers.begin(), headers.end(), name) == headers.end()){
                        std::cout<< "The event\t" << name << "\t is not in the headers" << std::endl;
                        continue;
                    }
                    if(add_counter_value){
                        csv_file.addElementToRecord(e.median_value, name);
                    } else {
                        csv_file.addElementToRecord(-1, name);
                    }
                }
            }
        }

        void computeMedian(){
            for(auto& events: threads_events) {
                if(events.empty()){
                    std::cout << "There are no events to record" << std::endl;
                    return;
                }
                for (auto &e: events) {
                    std::sort(e.counters_values.begin(), e.counters_values.end());
                    if (e.counters_values.size() == 1) {
                        e.median_value = e.counters_values.front();
                    } else if (e.counters_values.size() > 1) {
                        e.median_value = e.counters_values[(e.counters_values.size() / 2)];
                    } else {
                        e.median_value = 0;
                    }
                }
            }
        }

        void resetThreadPAPICounters(){
            for(auto& events: threads_events){
                for(auto& e: events){
                    e.median_value = 0;
                    e.counters_values.clear();
                }
            }
        }
        ~PAPIParallelProfiler()=default;
    };


    class FusionDemoCustom{
    protected:
        int n_, nnz_;
        double *correct_x_;
        std::vector<double> x_, x_in_;
        std::string name_;
        PAPISerialProfiler*   serial_profiling_ptr;
        PAPIParallelProfiler* parallel_profiling_ptr;
        bool do_serial_profile;
        bool init_profile_ptr;
        bool init_parallel_profile_ptr;
        bool init_correct_x;
        timing_measurement analysis_time_;
        CSR *L1_csr_, *L2_csr_, *A_csr_;
        CSC *L1_csc_, *L2_csc_, *A_csc_;

        //Virtual Functions
        virtual timing_measurement fused_code() = 0;
        virtual void build_set() = 0;

        virtual void setting_up(){
            std::fill_n(x_in_.data(), n_, 1);
            std::fill_n(x_.data(), n_, 0.0);
        }

        virtual void testing(){
            if (correct_x_)
                if (!is_equal(0, n_, correct_x_, x_.data(), 1e-6))
                    PRINT_LOG(name_ + " code != reference solution.\n");
        }


    public:

        FusionDemoCustom(int n, int nnz, std::string name){
            this -> n_ = n;
            this -> name_ = name;
            this -> nnz_ = nnz;
            x_in_.resize(n, 0);
            x_.resize(n, 0);
            this -> do_serial_profile = false;
            this -> init_profile_ptr = false;
            this -> serial_profiling_ptr = nullptr;
        }

        /*
         * Set and get functions for profiling
         */
        void setProfilingPtr(PAPISerialProfiler* ptr){
            this->serial_profiling_ptr = ptr;
            if(serial_profiling_ptr != NULLPNTR){
                this->init_profile_ptr = true;
            } else {
                this->init_profile_ptr = false;
            }

        }

//        //Set and get functions
//        void setThreadProfilingPtr(PAPIThreadProfiler* ptr){
//            this->thread_profiling_ptr = ptr;
//            if(profiling_ptr != NULLPNTR){
//                this->init_thread_profile_ptr = true;
//            } else {
//                this->init_thread_profile_ptr = false;
//            }
//
//        }


        void doProfiling(bool profile){
            this->do_serial_profile = profile;
        }

        void initCorrectX(double* correct_x){
            this -> correct_x_ = correct_x;
            init_correct_x = true;
        }

        static int getNumberOftests(){return num_tests_;}

        //The main function
        timing_measurement evaluate(){
            timing_measurement median_t;
            std::vector<timing_measurement> time_array;
            analysis_time_.start_timer();
            build_set();
            analysis_time_.measure_elapsed_time();
            for (int i = 0; i < num_tests_; ++i) {
                if (do_serial_profile) {
                    //Check whether the profiling pointer is initialized
                    if (!init_profile_ptr) {
                        std::cout << "There is no initialized profiling pointer to use for profiling" << std::endl;
                        return median_t;
                    }
                    if(i == 0){
                        std::cout << "Start Profiling" << std::endl;
                    }
                    //Profiling routine
                    for(auto& event: serial_profiling_ptr->getEvents()){
                        setting_up();
                        serial_profiling_ptr->startProfiling(event.code);
                        timing_measurement t1 = fused_code();
                        serial_profiling_ptr->endProfiling(event.code);
                        time_array.emplace_back(t1);
                    }
                } else {
                    setting_up();
                    timing_measurement t1 = fused_code();
                    time_array.emplace_back(t1);
                }
            }
            testing();
            median_t = time_median(time_array);
            return median_t;
        }


        virtual ~FusionDemoCustom(){};
    };

}




#endif //LBC_LIB_PAPI_UTILS_H
