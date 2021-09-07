# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities to process different mesh files formats."""
import itertools
import struct
from typing import Any, Sequence

# Internal imports.


def _flatten(list_of_lists: Sequence[Sequence[Any]]) -> Sequence[Any]:
  return list(itertools.chain.from_iterable(list_of_lists))


def _relabel_obj_to_mj(vertices, faces, texcoords, normals):
  """Remapping elemets from obj to mujoco compatible format.

  In normal obj we specify a list of 3D coordinates and texture coordinates.
  Then when defining faces we can choose a different index for each one. This
  way a single 3D location which has 2 different texture coordinates on 2
  different faces can in obj be representated by defining a single 3D location
  and 2 texture coords. Than when defining faces we match these up. However in
  mujoco this is not possible as when face indexes into the array it uses the
  same index for all. Therefore here we need to create a new vertex for every
  used combination of position, texture coordinare and normal.

  Args:
    vertices: (vertex, 3) float list
    faces: (faces, 3) int  list
    texcoords: (texcoords, 2) float list
    normals: (normals, 3) float list

  Returns:
    A tuple of:
      * vertices: (nvertex, 3) float list
      * faces: (faces, 3) int  list
      * texcoords: (nvertex, 2) float list
      * normals: (nvertex, 3) float list, or an empty list
  """
  unique_triples_mapping = {}
  remapped_vertices = []
  remapped_faces = []
  remapped_texcoords = []
  remapped_normals = []
  for face in faces:
    this_face = []
    for vertex in face:
      if vertex not in unique_triples_mapping:
        unique_triples_mapping[vertex] = len(remapped_vertices)
        remapped_vertices.append(vertices[vertex[0]])
        remapped_texcoords.append(texcoords[vertex[1]])
        if normals:
          remapped_normals.append(normals[vertex[2]])
      this_face.append(unique_triples_mapping[vertex])
    remapped_faces.append(this_face)
  flat_remapped_vertices = _flatten(remapped_vertices)
  flat_remapped_faces = _flatten(remapped_faces)
  flat_remapped_texcoords = _flatten(remapped_texcoords)
  flat_remapped_normals = _flatten(remapped_normals)
  return (flat_remapped_vertices, flat_remapped_faces, flat_remapped_texcoords,
          flat_remapped_normals)


def _parse_obj(path):
  """Parses obj from a path into a list of meshes."""
  with open(path) as f:
    obj_lines = f.readlines()

  parsed_objects = []
  vertices = []
  faces = []
  texcoords = []
  normals = []

  for l in obj_lines:
    l = l.strip()
    token = l.split(' ')
    k = token[0]
    v = token[1:]
    if k == 'o':
      if vertices:
        parsed_objects.append((vertices, faces, texcoords, normals))
      vertices = []
      faces = []
      texcoords = []
      normals = []
    elif k == 'v':
      vertices.append(tuple(float(x) for x in v))
    elif k == 'f':
      v = [tuple(int(n) - 1 for n in b.split('/')) for b in v]
      if len(v) == 4:
        faces.append([v[2], v[3], v[0]])
        v = v[:3]
      faces.append(v)
    elif k == 'vt':
      raw_texcoords = tuple(float(x) for x in v)
      # There seems to be an inconsistency between Katamari and MuJoco in the
      # way the texture coordinates are defined
      texcoords.append((raw_texcoords[0], 1 - raw_texcoords[1]))
    elif k == 'vn':
      normals.append(tuple(float(x) for x in v))

  parsed_objects.append((vertices, faces, texcoords, normals))
  return parsed_objects


def object_to_msh_format(vertices, faces, texcoords, normals):
  """Coverts a mesh from lists to a binary MSH format."""
  nvertex = len(vertices) // 3
  nnormal = len(normals) // 3
  ntexcoord = len(texcoords) // 2
  nface = len(faces) // 3

  # Convert to binary format according to:
  # # http://mujoco.org/book/XMLreference.html#mesh
  msh_string = bytes()
  msh_string += struct.pack('4i', nvertex, nnormal, ntexcoord, nface)
  msh_string += struct.pack(str(3 * nvertex) + 'f', *vertices)
  if nnormal:
    msh_string += struct.pack(str(3 * nnormal) + 'f', *normals)
  if ntexcoord:
    msh_string += struct.pack(str(2 * ntexcoord) + 'f', *texcoords)
  msh_string += struct.pack(str(3 * nface) + 'i', *faces)
  return msh_string


def obj_file_to_mujoco_msh(mesh_file):
  msh_strings = [
      object_to_msh_format(*_relabel_obj_to_mj(*parsed_object))
      for parsed_object in _parse_obj(mesh_file)
  ]
  return msh_strings
