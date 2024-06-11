# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Package building script."""

import setuptools


def _get_requirements(requirements_file):  # pylint: disable=g-doc-args
  """Returns a list of dependencies for setup() from requirements.txt.

  Currently a requirements.txt is being used to specify dependencies. In order
  to avoid specifying it in two places, we're going to use that file as the
  source of truth.

  Lines starting with -r will be ignored. If the requirements are split across
  multiple files, call this function multiple times instead and sum the results.
  """

  def line_should_be_included(line):
    return line and not line.startswith("-r")

  with open(requirements_file) as f:
    return [_parse_line(line) for line in f if line_should_be_included(line)]


def _parse_line(s):
  """Parses a line of a requirements.txt file."""
  requirement, *_ = s.split("#")
  return requirement.strip()


setuptools.setup(
    name="dm_robotics-manipulation",
    package_dir={"dm_robotics.manipulation": ""},
    packages=[
        "dm_robotics.manipulation",
        "dm_robotics.manipulation.props",
        "dm_robotics.manipulation.props.parametric_object",
        "dm_robotics.manipulation.props.parametric_object.rgb_objects",
        "dm_robotics.manipulation.props.rgb_objects",
        "dm_robotics.manipulation.props.utils",
        "dm_robotics.manipulation.standard_cell",
    ],
    version="0.8.0",
    license="Apache 2.0",
    author="DeepMind",
    description="Parametrically defined mesh objects",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/deepmind/dm_robotics/tree/main/py/manipulation",
    python_requires=">=3.7, <3.11",
    setup_requires=["wheel >= 0.31.0"],
    install_requires=(_get_requirements("requirements.txt") +
                      _get_requirements("requirements_external.txt")),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering",
    ],
    zip_safe=True,
    include_package_data=True,
)
