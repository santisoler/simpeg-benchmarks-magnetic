#!/bin/bash

command time -v python code/large_model.py choclo 2> results/memory-choclo
command time -v python code/large_model.py geoana 2> results/memory-geoana
command time -v python code/large_model.py dask 2> results/memory-dask
