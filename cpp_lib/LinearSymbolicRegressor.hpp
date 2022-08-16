#include <iostream>
#include <vector>
#include <map>
#include <execution>
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <pyxtensor/pyxtensor.hpp>

#include <xtensor/xarray.hpp>

int randint(int min, int max)
{
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(min, max);
    return dist(rng);
}

float randfloat(float min, float max)
{
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<> dist(min, max);
    return dist(rng);
}

class Gen
{
public:
    int operation, arity;
    float value;
    std::map<std::string, float> symreg_params;

    Gen(std::map<std::string, float> symreg)
    {
        symreg_params = symreg;
        operation = randint(1, 8);
        value = randfloat(1, 100);
    }

    void mutate_gene()
    {
        operation = randint(1, 8);
        value = randfloat(1, 100);
    }
};

class Program
{
public:
    int init_length;
    float fitness;
    std::map<std::string, float> symreg_params;
    std::vector<Gen> genes;

    // Program(){};

    Program(std::map<std::string, float> symreg)
    {
        symreg_params = symreg;
        init_length = symreg["initial_depth"];

        for (int i = 0; i < init_length; i++)
        {
            std::cout<<"Init gen!!"<<std::endl;
            Gen g(symreg_params);
            genes.push_back(g);
        }
    }

    void mutate()
    {
        float p = randfloat(0, 1);
        float p_mutation = symreg_params["p_mutation"];
        float p_mutation_add_node = symreg_params["p_mutation_insert_node"];
        float p_mutation_delete_node = symreg_params["p_mutation_delete_node"];
        float p_mutation_replication = symreg_params["p_mutation_replication"];

        if (p < p_mutation)
        {
            int program_length = genes.size();
            int program_rand = randint(0, program_length);
            genes[program_rand].mutate_gene();
        }
        else if (p < p_mutation + p_mutation_add_node)
        {
            int program_length = genes.size();
            int program_rand = randint(0, program_length);
            Gen g(symreg_params);
            genes.insert(genes.begin() + program_rand, g);
        }
        else if (p < p_mutation + p_mutation_add_node + p_mutation_delete_node)
        {
            int program_length = genes.size();
            int program_rand = randint(0, program_length);
            genes.erase(genes.begin() + program_rand);
        }
    }

    void xover(Program father)
    {
        std::vector<Gen> child;
        int child_length = genes.size();
        int father_length = father.genes.size();
        int short_length = (child_length > father_length) ? father_length : child_length;
        int cut_point = randint(0, short_length);

        for (int i = 0; i < cut_point; i++)
        {
            child.push_back(genes[i]);
        }

        for (int i = cut_point; i < father_length; i++)
        {
            child.push_back(father.genes[i]);
        }
        genes = child;
    }

    void print_program()
    {
        std::string prefix = "y = ";

        std::map<int, std::string> operators{
            {0, "sin"},
            {1, "cos"},
            {2, "exp"},
            {3, "+"},
            {4, "+ x"},
            {5, "-"},
            {6, "- x"},
            {7, "*"},
            {8, "* x"},
        };
        std::string result = "";
        result += "x";
        for (auto& gen : genes)
        {
            switch (gen.operation)
            {
            case 0:
                result = operators[gen.operation] + "(" + result + ")";
                break;
            case 1:
                result = operators[gen.operation] + "(" + result + ")";
                break;
            case 2:
                result = operators[gen.operation] + "(" + result + ")";
                break;
            case 3:
                result = "(" + result + operators[gen.operation] + std::to_string(gen.value) + ")";
                break;
            case 4:
                result = "(" + result + operators[gen.operation] + ")";
                break;
            case 5:
                result = "(" + result + operators[gen.operation] + std::to_string(gen.value) + ")";
                break;
            case 6:
                result = "(" + result + operators[gen.operation] + ")";
                break;
            case 7:
                result = "(" + result + operators[gen.operation] + std::to_string(gen.value) + ")";
                break;
            case 8:
                result = "(" + result + operators[gen.operation] + ")";
                break;

            default:
                break;
            }
        }

        std::cout << prefix + result << std::endl;
    }

    xt::xarray<double> compute_program(xt::xarray<double> x_train)
    {
        xt::xarray<double> result = x_train;

        for (auto gen : genes)
        {
            switch (gen.operation)
            {
            case 0:
                result = xt::sin(result * gen.value);
                break;
            case 1:
                result = xt::cos(result * gen.value);
                break;
            case 2:
                result = xt::exp(result * gen.value);
            case 3:
                result = result + gen.value;
                break;
            case 4:
                result = result + x_train;
                break;
            case 5:
                result = result - gen.value;
                break;
            case 6:
                result = result - x_train;
                break;
            case 7:
                result = result * gen.value;
                break;
            case 8:
                // safe div
                result = result * x_train;
                break;
            default:
                break;
            }
        }
        return result;
    }

    void save_program(std::string _PATH_)
    {
        std::ofstream file;
        file.open(_PATH_ + "_.csv", std::ios::out);
        if (file.fail())
        {
            std::cout << "No se ha podido guardar el programa" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        /*SAVE FILE*/
        // file << "x" + "\n";

        for (int i = 0; i < genes.size(); i++)
        {
            file << std::to_string(genes[i].operation) + "," + std::to_string(genes[i].value);
            if (i < genes.size() - 1)
            {
                file << "\n";
            }
        }

        file.close();
    }

    void load_program(std::string _PATH_)
    {
        std::ifstream file;
        file.open(_PATH_, std::ios::in);
        if (file.fail())
        {
            std::cout << "No se ha podido cargar el programa" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        /*LOAD FILE*/
        std::string x;
        // std::getline(file, x, '\n');
        // init_col = std::stoi(x, nullptr, 10);
        // std::cout << x << std::endl;
        while (!file.eof())
        {
            Gen g(symreg_params);

            std::getline(file, x, ',');
            g.operation = std::stoi(x, nullptr, 10);
            std::cout << x << std::endl;

            std::getline(file, x, ',');
            g.value = std::stoi(x, nullptr, 10);
            std::cout << x << std::endl;

            genes.push_back(g);
        }

        file.close();
    }

    bool operator<(const Program &other)
    {
        return (fitness < other.fitness);
    }
};

class LinearSymbolicRegressor
{
public:
    std::map<std::string, float> symreg_params;
    std::vector<Program> programs;
    std::vector<Program> best_programs;
    std::vector<double> fitness_average;

    LinearSymbolicRegressor(std::map<std::string, float> symreg)
    {
        symreg_params = symreg;
        int n_programs = symreg["n_programs"];
        for (int i = 0; i < n_programs; i++)
        {
            std::cout<<"Init program!!"<<std::endl;
            Program p(symreg);
            programs.push_back(p);
        }
    }

    double MeanSquaredError(xt::xarray<double> y_train, xt::xarray<double> y_test)
    {
        int n = y_train.shape()[0];
        xt::xarray<double> diff = y_train - y_test;
        xt::xarray<double> squared = xt::pow(diff, 2);
        double sum = xt::sum(squared)[0];
        double fitness = sum * (double)(1 / n);
        std::cout<<" Fitness from function: "<< fitness<<std::endl;
        std::cout<<"VALUE OF SUM: " <<sum<<std::endl;
        std::cout<<"VALUE OF N: " <<n<<std::endl;
        return fitness;
    }

    void evaluate_programs(xt::xarray<double> y_train, xt::xarray<double> x_train)
    {
        std::for_each(
            std::execution::seq,
            std::begin(programs),
            std::end(programs),
            [&](Program &program)
            {
                xt::xarray<double> y_test = program.compute_program(x_train);
                program.print_program();
                program.fitness = MeanSquaredError(y_train, y_test);
                std::cout<<program.fitness<<std::endl;
                std::cout<<std::endl;
            });
    }

    void fit(xt::xarray<double> &y_train, xt::xarray<double> &x_train)
    {
        std::cout<<" Starting Fitness Process!!"<<std::endl;
        int n_generations = symreg_params["n_generations"];
        int fitness_counter = 0;

        for (int i = 0; i < n_generations; i++)
        {
            float fitness_sum = 0;
            evaluate_programs(y_train, x_train);
            evolve();
        }

        std::cout<<" Finishing Fitness Process!!"<<std::endl;
    }

    void selection()
    {
        int n_best = symreg_params["n_best"];

        for (auto program : programs)
        {
            int best_size = best_programs.size();

            if (best_size < n_best)
            {
                best_programs.push_back(program);
            }
            else if (program.fitness > best_programs[1].fitness)
            {
                best_programs.erase(best_programs.end());
                best_programs.push_back(program);
                std::sort(best_programs.begin(), best_programs.end());
            }
        }
    }

    void evolve()
    {
        selection();
        float p_crossover = symreg_params["p_xover"];
        float p_mutation = symreg_params["p_mutations"];
        float p_insertion = symreg_params["p_insertion"];
        float p_replication = symreg_params["p_replication"];

        for (auto &program : programs)
        {
            float p = randfloat(0, 1);

            if (p < p_crossover)
            {
                int k = randint(0, int(best_programs.size()) - 1);
                program.xover(best_programs[k]);
            }
            if (p < p_mutation)
            {
                program.mutate();
            }
        }
    }
};