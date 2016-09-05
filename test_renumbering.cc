#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>

#include <vector>
#include <fstream>
#include <iostream>

using namespace dealii;

class Test
{
private:
  parallel::shared::Triangulation<2> triangulation;
  FE_Q<2> fe;
  DoFHandler<2> dof_handler;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  TrilinosWrappers::SparseMatrix system_matrix;

public:
  Test() :
    triangulation(MPI_COMM_WORLD),
    fe(1),
    dof_handler(triangulation)
  {
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(2);
    dof_handler.distribute_dofs(fe);

    renumber();
    init_structures();
  }

private:
  void renumber()
  {
    deallog << "Starting renumbering..." << std::endl;
    // DoFRenumbering::Cuthill_McKee(dof_handler);

    locally_owned_dofs = dof_handler.locally_owned_dofs();

    std::vector<types::global_dof_index> new_number(dof_handler.n_dofs());
    for (decltype(new_number)::size_type i = 0; i < dof_handler.n_dofs(); ++i)
      {
        new_number[i] = dof_handler.n_dofs() - i - 1;
      }

    std::vector<types::global_dof_index> local_new_number;
    for (const auto dof : locally_owned_dofs)
      {
        local_new_number.push_back(new_number[dof]);
      }

    deallog << "n_dofs = " << dof_handler.n_dofs() << std::endl;
    deallog << "before renumbering:" << std::endl;
    locally_owned_dofs.print(deallog);
    dof_handler.renumber_dofs(local_new_number);

    deallog << "after renumbering:" << std::endl;
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_owned_dofs.print(deallog);
    deallog << "Done renumbering." << std::endl;
  }

  void init_structures()
  {
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    deallog << "locally owned dofs:" << std::endl;
    locally_owned_dofs.print(deallog);
    deallog << "locally relevant dofs:" << std::endl;
    locally_relevant_dofs.print(deallog);

    DynamicSparsityPattern sparsity_pattern (locally_relevant_dofs);
    DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);
    SparsityTools::distribute_sparsity_pattern (sparsity_pattern,
                                                dof_handler.n_locally_owned_dofs_per_processor(),
                                                MPI_COMM_WORLD,
                                                locally_relevant_dofs);

    system_matrix.reinit (locally_owned_dofs,
                          sparsity_pattern,
                          MPI_COMM_WORLD);
    // This range is incorrect if we use the hand-rolled renumbering.
    const auto local_range = system_matrix.local_range();
    deallog << "sparse matrix row range: "
            << local_range.first
            << ", "
            << local_range.second
            << std::endl;

    const auto domain_indices = system_matrix.locally_owned_domain_indices();
    deallog << "domain indices (i.e., nonzero columns in the matrix): ";
    domain_indices.print(deallog);
    deallog << std::endl;

    const auto range_indices = system_matrix.locally_owned_range_indices();
    deallog << "range indices (i.e., nonzero rows in the matrix): ";
    range_indices.print(deallog);
    deallog << std::endl;
  }
};

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv);

  const auto rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  std::ofstream logfile("debug_output_" + Utilities::to_string(rank));
  deallog << std::setprecision(4);
  deallog.attach(logfile);
  deallog.depth_console(0);
  deallog.threshold_double(1.e-10);

  // In this minimal example only the failing mode is run.
  Test test2;

  return 0;
}
