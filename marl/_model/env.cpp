//  Korali environment for CubismUP-2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "env.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>

int _argc;
char **_argv;
std::mt19937 _randomGenerator;

size_t NUMAGENTS = 4;
size_t NUMACTIONS = 256;
size_t NUMSTATES = 256/NUMAGENTS;

// Environment Function
void runEnvironment(korali::Sample &s)
{
  // Get MPI communicator
  MPI_Comm comm = *(MPI_Comm*) korali::getWorkerMPIComm();

  int root = 0;
  int maxprocs = 1;

  MPI_Info info;
  int MPI_Info_create( info );
  MPI_Info_set(info, "wdir", "."); //args.resdir
  //MPI_INFO_SET(info, "bind_to", value); //none

  int ok = MPI_Comm_spawn("./bla_16x65x16_1", MPI_ARGV_NULL, maxprocs, info, root,  MPI_COMM_SELF, &comm, MPI_ERRCODES_IGNORE);

  // Get rank and size of subcommunicator
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // Get rank in world
  int rankGlobal;
  MPI_Comm_rank(MPI_COMM_WORLD, &rankGlobal);

  // Setting seed
  const size_t sampleId = s["Sample Id"];
  //_randomGenerator.seed(sampleId);

  // Make sure folder / logfile is created before switching path
  MPI_Barrier(comm);

  std::vector<std::vector<double>> states(NUMAGENTS, std::vector<double>(NUMSTATES));
  std::vector<std::vector<double>> actions(NUMAGENTS, std::vector<double>(NUMACTIONS));
  std::vector<double> rewards(NUMAGENTS, 0.);
  s["State"] = states;

  bool done = false;
  int curStep = 0;
  int maxSteps = 1000;
  while ( curStep < maxSteps && done == false )
  {
      s.update();
      auto actionsJson = s["Action"];

      // Setting action for each agent
      for( int i = 0; i<NUMAGENTS; i++ )
      {
        actions[i] = actionsJson[i].get<std::vector<double>>();
      }

    s["State"]  = states;
    s["Reward"] = rewards;

    // Advancing to next step
    curStep++;
  }

  // Setting termination status
  s["Termination"] = done ? "Terminal" : "Truncated";
}
