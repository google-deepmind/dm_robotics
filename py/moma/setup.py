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
    name="dm_robotics-moma",
    package_dir={"dm_robotics.moma": ""},
    packages=[
        "dm_robotics.moma",
        "dm_robotics.moma.effectors",
        "dm_robotics.moma.models",
        "dm_robotics.moma.utils",
        "dm_robotics.moma.sensors",
        "dm_robotics.moma.tasks",
        "dm_robotics.moma.tasks.example_task",
        "dm_robotics.moma.models.arenas",
        "dm_robotics.moma.models.end_effectors.robot_hands",
        "dm_robotics.moma.models.end_effectors.wrist_sensors",
        "dm_robotics.moma.models.robots.robot_arms",
    ],
    version="0.2.0",
    license="Apache 2.0",
    author="DeepMind",
    description="Tools for authoring robotic manipulation tasks.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/deepmind/dm_robotics/tree/main/py/moma",
    python_requires=">=3.7, <3.10",
    install_requires=(_get_requirements("requirements.txt") +
                      _get_requirements("requirements_external.txt")),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    zip_safe=True,
    include_package_data=True
)
