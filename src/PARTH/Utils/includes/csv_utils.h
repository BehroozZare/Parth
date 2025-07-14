//
// Created by behrooz on 12/2/20.
//

#ifndef LBC_LIB_CSV_UTILS_H
#define LBC_LIB_CSV_UTILS_H

#include <algorithm>
#include <cstring>
#include <fstream>
#include <ios>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace PARTH {
class CSVManager {
public:
  /*
   * @brief It create a CSV dataset for further analysis
   * @param output_name: the name of the output CSV
   * @param output_address: the place to save the CSV dataset (it is dummy
   * variable for now, the address is the workspace address of the file for now)
   * @param column_names: the name of each column of the dataset
   * @param force_creation: it will create a file. If a file with the same name
   * exist it will delete that file and create a new file.
   */
  CSVManager(std::string output_name, std::string output_address,
             std::vector<std::string> column_names, bool force_creation) {
    this->output_name = output_name;
    this->output_name += ".csv";
    this->output_addres = output_address;
    this->column_names = column_names;
    this->num_column = column_names.size();
    this->record_in_progres = false;
    this->defined_record_status.resize(num_column);
    this->record.resize(num_column);
    this->num_records = 0;
    // Create the CSV file or open the existing file
    std::ifstream file(this->output_name,
                       std::ios_base::out | std::ios_base::in);
    if (!file || force_creation) {
      std::cout << "Create a new data file " << this->output_name << std::endl;
      // If it doesn't exist or it is forced to create the new file create the
      // file and add header
      file.close();
      if (file)
        std::remove(&this->output_name[0]);

      dataset_file.open(this->output_name, std::ios::out | std::ios::app);
      if (dataset_file.fail()) {
        std::cout << "Cannot create the file" << std::endl;
      }
      // Add the header to CSV file
      for (auto iter = column_names.begin(); iter != column_names.end() - 1;
           iter++) {
        dataset_file << *iter << ",";
      }
      dataset_file << column_names.back() << "\n";
    } else { // Open the file and put the iterator in the end of the CSV
      std::cout << "Write in the existing file " << this->output_name
                << std::endl;
      file.close();
      if (dataset_file.fail()) {
        std::cout << "Cannot create the file" << std::endl;
      }
      dataset_file.open(this->output_name, std::ios::binary | std::ios::app);
    }
  }

  // This is for the case of if the input is the string itself
  template <typename T> std::string toString(const T &t) {
    return std::to_string(t);
  }

  std::string toString(const char *t) { return t; }

  std::string toString(const std::string &t) { return t; }

  static std::string toString(const double &t) {
    std::stringstream ss;
    ss << t;
    return ss.str();
  }
  template <class T>
  void addElementToRecord(T element, std::string column_name,
                          bool print = false) {
    if (!record_in_progres) {
      std::fill(defined_record_status.begin(), defined_record_status.end(),
                false);
      record_in_progres = true;
    }
    if (print) {
      std::cout << "The entry " << column_name << " has value " << element
                << std::endl;
    }
    // Find the column, add the entry with conversion if necessary
    auto iter =
        std::find(column_names.begin(), column_names.end(), column_name);
    if (iter == column_names.end()) {
      std::cout << "the column name " << column_name << " is not valid"
                << std::endl;
    } else {
      std::string element_string = toString(element);
      auto record_element_idx = std::distance(column_names.begin(), iter);
      defined_record_status[record_element_idx] = true;
      record[record_element_idx] = element_string;
    }
  };

  /*
   * @brief After creating an entry using addElementToEntry you can save the
   * entry in the CSV file using this function
   */
  void addRecord() {
    if (!record_in_progres) {
      std::cout << "There is no entry to add" << std::endl;
      return;
    } else {
      num_records++;
      for (int i = 0; i < num_column; i++) {
        if (defined_record_status[i] == false) {
          std::cout << "The entry column " << column_names[i]
                    << " is incomplete" << std::endl;
          return;
        }
      }
      // Add the entry to CSV file
      for (auto iter = record.begin(); iter != record.end() - 1; iter++) {
        dataset_file << *iter << ",";
      }
      dataset_file << record.back() << "\n" << std::flush;
      record_in_progres = false;
    }
  };

  /*
   * Not yet implemented
   */
  void deleteRecord();

  /*
   * Not yet implemented
   */
  void readRecord();
  /*
   * get headers
   */
  std::vector<std::string> getHeaderNames() { return this->column_names; }

  /*
   * get number of records
   */
  int getNumRecords() { return this->num_records; }

  ~CSVManager() {
    // close the file
    dataset_file.close();
  };

private:
  std::fstream dataset_file;
  int num_records;
  bool record_in_progres;
  int num_column;
  std::string output_name;
  std::string output_addres;
  std::vector<std::string> column_names;
  std::vector<std::string> record;
  std::vector<bool> defined_record_status;
};
} // namespace PARTH

#endif // LBC_LIB_CSV_UTILS_H
