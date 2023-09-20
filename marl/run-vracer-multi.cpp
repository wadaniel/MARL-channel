#include "_model/env.hpp"
#include "korali.hpp"
#include <filesystem>

int main(int argc, char *argv[])
{
  // Initialize MPI
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (provided != MPI_THREAD_FUNNELED)
  {
    printf("Error initializing MPI\n");
    exit(-1);
  }

  // retreiving number of task, agents, and ranks
  int nRanks  = atoi(argv[argc-1]);
  int rank;
  size_t NUMAGENTS = 4;
  size_t NUMACTIONS = 256;
  size_t NUMSTATES = 256/NUMAGENTS;


  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (rank == 0)
	  std::cout << "Running with nRanks: " << nRanks << " and with nAgents=" << NUMAGENTS << std::endl;

  // Storing parameters for environment
  _argc = argc;
  _argv = argv;

  // Getting number of workers
  int N = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &N);
  N = N - 1; // Minus one for Korali's engine
  N = (int)(N / nRanks); // Divided by the ranks per worker

  // Setting results path
  std::string trainingResultsPath = "./../_trainingResults/";
  std::string testingResultsPath = "./../_testingResults/";

  // Creating Experiment
  auto e = korali::Experiment();
  e["Problem"]["Type"] = "Reinforcement Learning / Continuous";

  // Check if existing results are there and continuing them
  auto found = e.loadState(trainingResultsPath + std::string("/latest"));
  if (found == true){
    // printf("[Korali] Continuing execution from previous run...\n");
    // Hack to enable execution after Testing.
    e["Solver"]["Termination Criteria"]["Max Generations"] = std::numeric_limits<int>::max();
  }

  // Configuring Experiment
  e["Problem"]["Environment Function"] = &runEnvironment;
  e["Problem"]["Agents Per Environment"] = NUMAGENTS;

  // Setting results path and dumping frequency in CUP
  //e["Problem"]["Custom Settings"]["Dump Frequency"] = 0.0;
  //e["Problem"]["Custom Settings"]["Dump Path"] = trainingResultsPath;
  //e["Problem"]["Actions Between Policy Updates"] = 1;

  // Setting up the state variables
  size_t curVariable = 0;
  for (; curVariable < NUMSTATES; curVariable++)
  {
    e["Variables"][curVariable]["Name"] = std::string("State") + std::to_string(curVariable);
    e["Variables"][curVariable]["Type"] = "State";
  }

  float maxV = 0.04285714285714286;

  for (; curVariable < NUMSTATES+NUMACTIONS; curVariable++)
  {
    e["Variables"][curVariable]["Name"] = "Control No. " + std::to_string(curVariable-NUMSTATES);
    e["Variables"][curVariable]["Type"] = "Action";
    e["Variables"][curVariable]["Lower Bound"] = -maxV;
    e["Variables"][curVariable]["Upper Bound"] = +maxV;
    e["Variables"][curVariable]["Initial Exploration Noise"] = maxV;
  }

  /// Defining Agent Configuration
  e["Solver"]["Type"] = "Agent / Continuous / VRACER";
  e["Solver"]["Mode"] = "Training";
  e["Solver"]["Episodes Per Generation"] = 1;
  e["Solver"]["Concurrent Workers"] = N;
  e["Solver"]["Experiences Between Policy Updates"] = 1;
  e["Solver"]["Learning Rate"] = 1e-4;
  e["Solver"]["Discount Factor"] = 0.95;
  e["Solver"]["Mini Batch"]["Size"] =  128;
  // e["Solver"]["Multi Agent Relationship"] = "Competition";

  /// Defining the configuration of replay memory
  e["Solver"]["Experience Replay"]["Start Size"] = 1024;
  e["Solver"]["Experience Replay"]["Maximum Size"] = 65536;
  e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8;
  e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 5.0;
  e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3;
  e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1;

  //// Defining Policy distribution and scaling parameters
  e["Solver"]["Policy"]["Distribution"] = "Clipped Normal";
  e["Solver"]["State Rescaling"]["Enabled"] = true;
  e["Solver"]["Reward"]["Rescaling"]["Enabled"] = true;

  //// Defining Neural Network
  e["Solver"]["Neural Network"]["Engine"] = "OneDNN";
  e["Solver"]["Neural Network"]["Optimizer"] = "Adam";

  e["Solver"]["L2 Regularization"]["Enabled"] = true;
  e["Solver"]["L2 Regularization"]["Importance"] = 1.0;

  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear";
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128;

  e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation";
  e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh";

  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear";
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128;

  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation";
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh";

  ////// Defining Termination Criteria
  e["Solver"]["Termination Criteria"]["Max Experiences"] = 5e5;

  ////// Setting Korali output configuration
  e["Console Output"]["Verbosity"] = "Normal";
  e["File Output"]["Enabled"] = true;
  e["File Output"]["Frequency"] = 1;
  e["File Output"]["Use Multiple Files"] = false;
  e["File Output"]["Path"] = trainingResultsPath;

  if (rank == 0)
  {
    std::cout << "rank 0 copying files to " << trainingResultsPath << std::endl;
    mkdir(trainingResultsPath.c_str(),0777);
    system("sed 's/SAMPLINGHEIGHT/{args.ycoords}/' {srcDir}bla_macro.i > {args.resDir}/bla.i");
    std::filesystem::copy("../bin/bla_16x65x16_1", trainingResultsPath);
  }


  ////// Running Experiment
  auto k = korali::Engine();

  // Configuring profiler output
  k["Profiling"]["Detail"] = "Full";
  k["Profiling"]["Path"] = trainingResultsPath + std::string("/profiling.json");
  k["Profiling"]["Frequency"] = 60;

  // Configuring conduit / communicator
  k["Conduit"]["Type"] = "Distributed";
  k["Conduit"]["Ranks Per Worker"] = nRanks;
  korali::setKoraliMPIComm(MPI_COMM_WORLD);

  // ..and run
  k.run(e);
}
