## Assignment 1

This document provides a quick guide to the essential configuration variables and execution instructions for running the test suite in algorithms.py.

### Configuration Variables

The following variables, (top of the main script algorithms.py), control the behavior of the test run.

#### DISABLED_PROBLEMS

This list is used to explicitly exclude certain problem instances or sizes from the current test run. Any problem defined in the main problem set that has its identifier.

#### GENERATE_TIME_SERIES

This flag determines whether a detailed record of the algorithm's measurements over time should be saved to a file.

### Running the Test Suite

To execute the entire test suite, including all configured algorithms and problems (excluding those defined in DISABLED_PROBLEMS), simply run the main script from your terminal:

```
python algorithms.py
```