#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <mpi.h>

#include <vector>
#include <iostream>

using std::vector;
using std::cout;
using std::endl;

using namespace dealii;

class Test
{
private:
  MPI_Comm mpi_communicator;

  const unsigned int rank;
  const unsigned int n_ranks;

  parallel::shared::Triangulation<2> triangulation;

  DoFHandler<2> dof_handler;

  FE_Q<2> fe;
  QGauss<2> quadrature;

  ConstraintMatrix constraints;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  TrilinosWrappers::SparseMatrix system_matrix;
  TrilinosWrappers::MPI::Vector system_rhs, solution;

  ConditionalOStream   pcout;

public:
  Test(const bool do_renumber) :
    mpi_communicator(MPI_COMM_WORLD),
    rank(Utilities::MPI::this_mpi_process(mpi_communicator)),
    n_ranks(Utilities::MPI::n_mpi_processes(mpi_communicator)),
    triangulation(mpi_communicator),
    dof_handler(triangulation),
    fe(1),
    quadrature(2),
    pcout (std::cout, rank == 0)
  {
    pcout << "Start";

    if (do_renumber)
      pcout << " with renumbering" << endl;
    else
      pcout << " without renumbering" << endl;

    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(1);

    dof_handler.distribute_dofs(fe);

    constraints.clear();
    constraints.close();

    if (do_renumber) renumber();

    init_structures();

    assemble();

    solve();

    pcout << "Finished";

    if (do_renumber)
      pcout << " with renumbering" << endl;
    else
      pcout << " without renumbering" << endl;
  }

  ~Test ()
  {
    dof_handler.clear();
  }

private:

  void init_structures()
  {
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    solution.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    DynamicSparsityPattern sparsity_pattern (locally_relevant_dofs);
    DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern,
                                     constraints, /*keep constrained dofs*/ false);
    SparsityTools::distribute_sparsity_pattern (sparsity_pattern,
                                                dof_handler.n_locally_owned_dofs_per_processor(),
                                                MPI_COMM_WORLD,
                                                locally_relevant_dofs);

    system_matrix.reinit (locally_owned_dofs,
                          locally_owned_dofs,
                          sparsity_pattern,
                          MPI_COMM_WORLD);
  }

  void renumber()
  {
    //DoFRenumbering::Cuthill_McKee(dof_handler);

    locally_owned_dofs = dof_handler.locally_owned_dofs();

    vector<unsigned int> new_number(dof_handler.n_dofs());
    for (unsigned int i = 0; i < dof_handler.n_dofs(); i++)
      new_number[i] = dof_handler.n_dofs() - i - 1;

    vector<unsigned int> local_new_number;
    for (unsigned int dof : locally_owned_dofs)
      local_new_number.push_back(new_number[dof]);

    deallog << "n_dofs = " << dof_handler.n_dofs() << std::endl;
    deallog << "before renumbering:" << std::endl;
    locally_owned_dofs.print(dealii::deallog);
    dof_handler.renumber_dofs(local_new_number);

    deallog << "after renumbering:" << std::endl;
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_owned_dofs.print(dealii::deallog);
  }

  void assemble()
  {
    FEValues<2> fe_values(fe, quadrature,
                          update_gradients | update_values | update_JxW_values);
    system_matrix = 0;
    system_rhs = 0;
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
      {
        if ( !cell->is_locally_owned()) continue;

        fe_values.reinit(cell);

        Vector<double> local_rhs(fe.dofs_per_cell);
        local_rhs = 0;

        FullMatrix<double> local_matrix(fe.dofs_per_cell, fe.dofs_per_cell);
        local_matrix = 0;

        for (unsigned int q = 0; q < fe_values.n_quadrature_points; q++)
          {
            for (unsigned int i = 0; i < fe.dofs_per_cell; i++)
              {
                for (unsigned int j = 0; j < fe.dofs_per_cell; j++)
                  {
                    local_matrix(i, j) += fe_values.shape_value(i, q) * fe_values.shape_value(j, q) * fe_values.JxW(q);
                  }

                local_rhs(i) += (fe_values.shape_value(i, q) * fe_values.JxW(q));
              }
          }

        vector<unsigned int> local_dofs(fe.dofs_per_cell);
        cell->get_dof_indices(local_dofs);
        constraints.distribute_local_to_global(local_matrix, local_rhs, local_dofs, system_matrix, system_rhs);
      }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }

  void solve()
  {
    pcout << "l1(mat) = " << system_matrix.l1_norm() << std::endl;
    pcout << "fr(mat) = " << system_matrix.frobenius_norm() << std::endl;
    cout  << "l2(rhs) = " << system_rhs.l2_norm()    << std::endl;

    // TODO:
    //typedef TrilinosWrappers::SolverDirect SOLVER; // breaks!
    typedef TrilinosWrappers::SolverCG SOLVER; // gives different results with 1 and 2 cores

    TrilinosWrappers::PreconditionIdentity preconditioner;

    SolverControl sc;
    SOLVER::AdditionalData data;
    SOLVER solver(sc, data);

    solver.solve(system_matrix, solution, system_rhs, preconditioner);

    pcout << "l2(sol) = " << solution.l2_norm() << std::endl;
  }
};

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv);

  std::ofstream logfile("debug_output_"+Utilities::int_to_string(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD),3));
  dealii::deallog << std::setprecision(4);
  dealii::deallog.attach(logfile);
  dealii::deallog.depth_console(0);
  dealii::deallog.threshold_double(1.e-10);


  Test test1(false);

  MPI_Barrier(MPI_COMM_WORLD);

  Test test2(true);

  return 0;
}
