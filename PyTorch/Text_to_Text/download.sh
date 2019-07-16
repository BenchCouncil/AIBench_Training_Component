#!/bin/bash
mkdir ../../DataSet/WMT-English-German/training
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz && tar -xf training.tar.gz -C ../../DataSet/WMT-English-German/training
mkdir ../../DataSet/WMT-English-German/validation
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz && tar -xf validation.tar.gz -C ../../DataSet/WMT-English-German/validation 
mkdir ../../DataSet/WMT-English-German/testing
wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/mmt16_task1_test.tar.gz && tar -xf mmt16_task1_test.tar.gz -C ../../DataSet/WMT-English-German/testing

