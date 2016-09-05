#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/sparsity_tools.h>

#include <vector>
#include <iostream>
#include <fstream>

using namespace dealii;

class Test
{
private:
  MPI_Comm mpi_communicator;

  const unsigned int rank;
  const unsigned int n_ranks;

  // PETSc (correctly) crashes when one uses parallel::shared::Triangulation
  // here due to a renumbering bug with that class
  parallel::distributed::Triangulation<2> triangulation;
  FE_Q<2> fe;
  DoFHandler<2> dof_handler;
  QGauss<2> quadrature;

  ConstraintMatrix constraints;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  PETScWrappers::MPI::SparseMatrix system_matrix;
  PETScWrappers::MPI::Vector system_rhs, solution;

  ConditionalOStream pcout;

public:
  Test(const bool do_renumber) :
    mpi_communicator(MPI_COMM_WORLD),
    rank(Utilities::MPI::this_mpi_process(mpi_communicator)),
    n_ranks(Utilities::MPI::n_mpi_processes(mpi_communicator)),
    triangulation(mpi_communicator),
    fe(1),
    dof_handler(triangulation),
    quadrature(2),
    pcout (std::cout, rank == 0)
  {
    pcout << "Start";

    if (do_renumber)
      pcout << " with renumbering" << std::endl;
    else
      pcout << " without renumbering" << std::endl;

    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(1);

    dof_handler.distribute_dofs(fe);

    constraints.clear();
    constraints.close();

    if (do_renumber)
      {
        renumber();
      }

    init_structures();
    assemble();
    solve();

    pcout << "Finished";

    if (do_renumber)
      pcout << " with renumbering" << std::endl;
    else
      pcout << " without renumbering" << std::endl;
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

    solution.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    DynamicSparsityPattern sparsity_pattern (locally_relevant_dofs);
    DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern,
                                     constraints, /*keep constrained dofs*/ false);
    deallog << "sparsity pattern (before distribution):" << std::endl;
    sparsity_pattern.print(deallog.get_file_stream());
    deallog << std::endl;

    SparsityTools::distribute_sparsity_pattern (sparsity_pattern,
                                                dof_handler.n_locally_owned_dofs_per_processor(),
                                                MPI_COMM_WORLD,
                                                locally_relevant_dofs);
    deallog << "sparsity pattern (after distribution):" << std::endl;
    sparsity_pattern.print(deallog.get_file_stream());
    deallog << std::endl;

    system_matrix.reinit (locally_owned_dofs, locally_owned_dofs,
                          sparsity_pattern, MPI_COMM_WORLD);
    const auto local_range = system_matrix.local_range();
    deallog << "sparse matrix row range: "
            << local_range.first
            << ", "
            << local_range.second
            << std::endl;
    // This range is incorrect if we use the hand-rolled renumbering.
  }

  void assemble()
  {
    const auto &mapping = MappingQ1<2>{};
    FEValues<2> fe_values(fe, quadrature, update_values | update_JxW_values
                          | update_quadrature_points);
    system_matrix = 0;
    system_rhs = 0;
    Vector<double> local_rhs(fe.dofs_per_cell);
    FullMatrix<double> local_matrix(fe.dofs_per_cell, fe.dofs_per_cell);
    std::vector<unsigned int> local_dofs(fe.dofs_per_cell);

    for (auto cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            cell->get_dof_indices(local_dofs);

            local_rhs = 0;
            local_matrix = 0;

            for (unsigned int q = 0; q < fe_values.n_quadrature_points; q++)
              {
                for (unsigned int i = 0; i < fe.dofs_per_cell; i++)
                  {
                    for (unsigned int j = 0; j < fe.dofs_per_cell; j++)
                      {
                        local_matrix(i, j) += fe_values.shape_value(i, q) * fe_values.shape_value(j, q)
                          * fe_values.JxW(q);
                      }

                    local_rhs(i) += (fe_values.shape_value(i, q) * fe_values.JxW(q));
                  }
              }

            deallog << "Assembling on cell with dofs: ";
            for (unsigned int i = 0; i < local_dofs.size(); ++i)
              {
                deallog << local_dofs[i] << (i == local_dofs.size() - 1 ? "" : ", ");
              }
            deallog << std::endl;

            deallog << "dof coordinates: ";
            for (unsigned int i = 0; i < local_dofs.size(); ++i)
              {
                deallog << "("
                        << mapping.transform_unit_to_real_cell(cell, fe.unit_support_point(i))
                        << ")"
                        << (i == local_dofs.size() - 1 ? "" : ", ");
              }
            deallog << std::endl;

            deallog << "cell vertices: ";
            for (unsigned int i = 0; i < GeometryInfo<2>::vertices_per_cell; ++i)
              {
                deallog << "("
                        << cell->vertex(i)
                        << ")"
                        << (i == GeometryInfo<2>::vertices_per_cell - 1 ? "" : ", ");
              }
            deallog << std::endl;
            // conclusion from these three: the mesh information (topological
            // and coordinate) is correct.


            constraints.distribute_local_to_global(local_matrix,
                                                   local_rhs,
                                                   local_dofs,
                                                   system_matrix,
                                                   system_rhs);
          }
      }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }

  void solve()
  {
    deallog << "matrix entries: " << std::endl;
    system_matrix.print(deallog.get_file_stream());
    deallog.get_file_stream().flush();
    pcout << "l1(mat) = " << system_matrix.l1_norm() << std::endl;
    pcout << "fr(mat) = " << system_matrix.frobenius_norm() << std::endl;
    pcout << "l2(rhs) = " << system_rhs.l2_norm()    << std::endl;

    PETScWrappers::PreconditionNone preconditioner(system_matrix);

    SolverControl sc(100, 1e-10);
    PETScWrappers::SolverCG solver(sc, MPI_COMM_WORLD);

    // solver.solve(system_matrix, solution, system_rhs);
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    deallog << "rhs vector: " << std::endl;
    system_rhs.print(deallog.get_file_stream());
    deallog << std::endl;

    deallog << "solution vector: " << std::endl;
    solution.print(deallog.get_file_stream());
    deallog << std::endl;

    pcout << "l2(sol) = " << solution.l2_norm() << std::endl;
  }
};

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, /*max_num_threads=*/1);

  const auto rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  std::ofstream logfile("debug_output_" + Utilities::to_string(rank));
  deallog << std::setprecision(4);
  deallog.attach(logfile);
  deallog.depth_console(0);
  deallog.threshold_double(1.e-10);

  Test test1(false);

  MPI_Barrier(MPI_COMM_WORLD);
  for (unsigned int i = 0; i < 10; ++i)
    {
      deallog << "---------------------------------------"
              << "---------------------------------------"
              << std::endl;
    }

  Test test2(true);

  return 0;
}
