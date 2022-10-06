#include <iostream>
#include <vector>
#include <map>
#include <execution>
#include <algorithm>
#include <random>

#include <xtensor/xarray.hpp>
#include <pyxtensor/pyxtensor.hpp>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "cpp_lib/LinearSymbolicRegressor.hpp"


PYBIND11_MODULE(LinearSymbolicRegressor, m)
{ 
	// Evolver -> Linear Genetic Programming
	py::bind_vector<std::vector<double>>(m, "vector_double");
	py::bind_vector<std::vector<Gen>>(m, "vector_gen");
	py::bind_vector<std::vector<Program>>(m, "vector_program");
	py::bind_map<std::map<std::string, float>>(m, "stringfloatmap");

	py::class_<Gen>(m, "Gen")
		.def(py::init([](std::map<std::string, float> symreg)
					  { return new Gen(symreg); }))
		.def_readwrite("operation", &Gen::operation)
		.def_readwrite("value", &Gen::value)
		.def("mutate_gene", &Gen::mutate_gene);
 
	py::class_<Program>(m, "Program")
		.def(py::init([](std::map<std::string, float> symreg)
					  { return new Program(symreg); }))
		.def_readwrite("fitness", &Program::fitness)
		.def_readwrite("genes", &Program::genes) 
		.def("print_program", &Program::print_program)
		.def("compute_program", &Program::compute_program);
		//.def("save_program", &Program::save_program)
		//.def("load_program", &Program::load_program);

	py::class_<LinearSymbolicRegressor>(m, "LinearSymbolicRegressor")
		.def(py::init([](std::map<std::string, float> symreg)
					  { return new LinearSymbolicRegressor(symreg); }))
		.def_readwrite("symreg_params", &LinearSymbolicRegressor::symreg_params)
		.def_readwrite("programs", &LinearSymbolicRegressor::programs)
		.def_readwrite("best_programs", &LinearSymbolicRegressor::best_programs)
		.def_readwrite("fitness_average", &LinearSymbolicRegressor::fitness_average)
		.def("fit", &LinearSymbolicRegressor::fit);
}  