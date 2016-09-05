# Forum Trilinos Bug
## Overview
This is a repository with test code for #dealii/dealii/2966: briefly, manually
renumbering the degrees of freedom on a `parallel::shared::Triangulation<2>` has
unexpected consequences. This shows up with two MPI processes: one is okay.

## Branches
### `master`
Modified version of the test code with a lot more logging. This intermittently
segmentation faults in debug mode (possibly yet another bug).

### `minimal`
A stripped copy of `master` where I remove everything after the first
questionable piece of output. More specifically: after renumbering the DoFs the
Trilinos matrices both claim to own the wrong sets of rows, so stop execution
there for the sake of a minimal test case.

### `no-trilinos`
There is a related problem with the Trilinos wrappings: they write into entries
they do not own without complaint. The PETSc bindings correctly crash when we
tell them to do this.

### `original`
The original (submitted on the forum) test case. One can check that the problem
is with the shared triangulation by switching it out for a
`parallel::distributed::Triangulation<2>`.
