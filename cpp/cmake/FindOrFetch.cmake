#.rst:
# FindOrFetch
# ----------------------
#
# Find or fetch a package in order to satisfy target dependencies.
#
#   FindOrFetch([USE_SYSTEM_PACKAGE [ON/OFF]]
#               [PACKAGE_NAME [name]]
#               [LIBRARY_NAME [name]]
#               [GIT_REPO [repo]]
#               [GIT_TAG [tag]]
#               [TARGETS [targets]])
#
# The command has the following parameters:
#
# Arguments:
#  - ``USE_SYSTEM_PACKAGE`` one-value argument on whether to search for the
#    package in the system (ON) or whether to fetch the library from a git
#    repository (OFF).
#  - ``PACKAGE_NAME`` name of the system-package. Ignored if
#    ``USE_SYSTEM_PACKAGE`` is ``OFF``.
#  - ``LIBRARY_NAME`` name of the library. Ignored if
#    ``USE_SYSTEM_PACKAGE`` is ``ON``.
#  - ``GIT_REPO`` git repository to fetch the library from. Ignored if
#    ``USE_SYSTEM_PACKAGE`` is ``ON``.
#  - ``GIT_TAG`` tag reference when fetching the library from the git
#    repository. Ignored if ``USE_SYSTEM_PACKAGE`` is ``ON``.
#  - ``TARGETS`` list of targets to be satisfied. If any of these targets are
#    not currently defined, this macro will attempt to either find or fetch the
#    package.

if (COMMAND FindOrFetch)
  return()
endif()

macro(FindOrFetch)
  if (NOT FetchContent)
    include(FetchContent)
  endif()

  # Parse arguments.
  set(options)
  set(one_value_args
    USE_SYSTEM_PACKAGE
    PACKAGE_NAME
    LIBRARY_NAME
    GIT_REPO
    GIT_TAG
  )
  set(multi_value_args TARGETS)
  cmake_parse_arguments(_ARGS
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  # Check if all targets are found.
  if (NOT _ARGS_TARGETS)
    message(FATAL_ERROR "dm_robotics::FindOrFetch: TARGETS must be specified.")
  endif()
  set(targets_found TRUE)
  foreach(target ${_ARGS_TARGETS})
    if (NOT TARGET ${target})
      message(STATUS "dm_robotics::FindOrFetch: target `${target}` not defined.")
      set(targets_found FALSE)
      break()
    endif()
  endforeach()

  # If targets are not found, use `find_package` or `FetchContent...` to get
  # it.
  if (NOT targets_found)
    if (${_ARGS_USE_SYSTEM_PACKAGE})
      message(STATUS
        "dm_robotics::FindOrFetch: Attempting to find `${_ARGS_PACKAGE_NAME}` in system packages..."
      )
      find_package(${_ARGS_PACKAGE_NAME} REQUIRED)
      message(STATUS
        "dm_robotics::FindOrFetch: Found `${_ARGS_PACKAGE_NAME}` in system packages."
      )
    else()
      message(STATUS
        "dm_robotics::FindOrFetch: Attempting to fetch `${_ARGS_LIBRARY_NAME}` from `${_ARGS_GIT_REPO}`..."
      )
      FetchContent_Declare(
        ${_ARGS_LIBRARY_NAME}
        GIT_REPOSITORY ${_ARGS_GIT_REPO}
        GIT_TAG ${_ARGS_GIT_TAG}
      )
      FetchContent_MakeAvailable(${_ARGS_LIBRARY_NAME})
      message(STATUS
        "dm_robotics::FindOrFetch: Fetched `${_ARGS_LIBRARY_NAME}` from `${_ARGS_GIT_REPO}`."
      )
    endif()
  else()
    message(STATUS
      "dm_robotics::FindOrFetch: `${_ARGS_PACKAGE_NAME}` targets found."
    )
  endif()
endmacro()
